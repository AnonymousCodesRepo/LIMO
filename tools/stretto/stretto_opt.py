"""Faithful port of Stretto's global gradient-based plan optimizer (the part that
matters for our cascade baseline).

A semantic operator = a cost-sorted cascade of physical operators o_1 -> ... -> o_K
-> gold. Each physical operator emits accept / reject / unsure per tuple from the
log-odds of its Yes/No tokens, using two thresholds theta_upper / theta_lower:

    accept (predict Yes)  if log_odds >  theta_upper
    reject (predict No)   if log_odds <  theta_lower
    unsure (escalate)     otherwise            -> passed to the next, costlier op

Only "unsure" tuples flow on; the final gold operator (the 70B, no compression)
takes all residual unsure tuples and emits its own (hard) label -- it never says
"unsure". Precision / recall are measured against the GOLD operator's outputs, not
human labels (the 70B is the cascade ceiling).

The optimizer picks, per physical operator, a pick factor sigma in [0,1] and the two
thresholds, to satisfy a query-level (precision, recall) target -- as a Bayesian
credible lower bound at level `confidence` -- while minimising expected token cost.
It uses a continuous relaxation with temperature annealing and Adam, exactly as in
reasondb/optimizer/gd_optimizer.py.

Track B uses a single non-gold operator (the 0.8B, no compression). The code is
written for the general K-operator case so Track A (the KV-cache ladder) can reuse it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import beta as _beta

THRESHOLD_MIN, THRESHOLD_MAX = -10.0, 10.0  # reasondb text_qa_filter THRESHOLD_RANGE


# --------------------------------------------------------------------------- #
# Bayesian credible lower bound, differentiable via a straight-through trick.
# value  = exact scipy Beta(1+successes, 1+failures) lower quantile
# gradient = normal-approximation gradient (as in reasondb beta_bounds)
# --------------------------------------------------------------------------- #
def _beta_lower_bound(successes: torch.Tensor, failures: torch.Tensor,
                      confidence: float) -> torch.Tensor:
    a = successes + 1.0
    b = failures + 1.0
    exact = torch.tensor(
        float(_beta.ppf(1.0 - confidence, a.detach().cpu().numpy(),
                        b.detach().cpu().numpy())),
        dtype=successes.dtype,
    )
    if not torch.isfinite(exact):
        exact = torch.zeros((), dtype=successes.dtype)
    mu = a / (a + b)
    sigma = torch.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1.0)))
    z = torch.tensor(float(_beta.ppf(1.0 - confidence, 1.0, 1.0)))  # placeholder, replaced below
    # normal quantile of (1-confidence) via icdf
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - confidence))
    ci_normal = mu + z * sigma
    return ci_normal + (exact - ci_normal).detach()


@dataclass
class Plan:
    """A fitted physical plan for one semantic operator (one query)."""
    theta_lower: np.ndarray      # (K,) reject threshold per non-gold operator
    theta_upper: np.ndarray      # (K,) accept threshold per non-gold operator
    pick: np.ndarray             # (K,) 1.0 if the operator is used, else 0.0
    target_prec: float
    target_rec: float
    # diagnostics on the training sample (hard decisions):
    train_esc: float
    train_prec_lb: float
    train_rec_lb: float
    feasible: bool


# --------------------------------------------------------------------------- #
# Hard-decision simulation of a plan on a set of tuples (used for scoring and at
# eval time). log_odds_list: list of (N,) arrays, one per non-gold operator, in
# cost order. gold: (N,) hard 0/1 labels from the gold (70B) operator.
# Returns: final hard prediction (0/1), and an "escalated" mask = reached gold.
# --------------------------------------------------------------------------- #
def simulate_hard(log_odds_list, theta_lower, theta_upper, pick, gold):
    n = len(gold)
    unresolved = np.ones(n, dtype=bool)          # still "unsure" -> flows on
    pred = np.full(n, -1, dtype=np.int64)
    for k in range(len(log_odds_list)):
        if pick[k] < 0.5:
            continue                              # operator not selected -> pass through
        lo = log_odds_list[k]
        acc = unresolved & (lo > theta_upper[k])
        rej = unresolved & (lo < theta_lower[k])
        pred[acc] = 1
        pred[rej] = 0
        unresolved = unresolved & ~acc & ~rej     # the rest stay unsure
    escalated = unresolved.copy()                 # residual unsure -> gold decides
    pred[escalated] = gold[escalated]
    return pred, escalated


def _hard_bounds(pred, gold, confidence):
    """Precision / recall Bayesian lower bounds of `pred` vs `gold` (both 0/1)."""
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    prec_lb = float(_beta.ppf(1.0 - confidence, tp + 1, fp + 1))
    rec_lb = float(_beta.ppf(1.0 - confidence, tp + 1, fn + 1))
    return prec_lb, rec_lb


# --------------------------------------------------------------------------- #
# Fit one plan for one (precision, recall) target, on the training sample.
# --------------------------------------------------------------------------- #
def fit_plan(log_odds_list, gold, op_costs, gold_cost,
             target_prec, target_rec, *,
             confidence=0.95, steps=300, lr=0.02, restarts=6,
             tau_begin=1.0, tau_end=0.02, optimize_pick=False, seed=0) -> Plan:
    """
    log_odds_list : list of K numpy (N,) arrays (training sample), cost order.
    gold          : numpy (N,) hard 0/1 gold (70B) labels on the training sample.
    op_costs      : list of K per-tuple token costs (USD) for each non-gold op.
    gold_cost     : per-tuple token cost of the gold (70B) operator.
    Returns the best feasible min-cost Plan (or min-violation if none feasible).
    """
    K = len(log_odds_list)
    N = len(gold)
    lo_t = [torch.tensor(x, dtype=torch.float64) for x in log_odds_list]
    gold_t = torch.tensor(gold, dtype=torch.float64)
    op_cost_t = torch.tensor(op_costs, dtype=torch.float64)
    max_cost = float(N) * (float(op_cost_t.sum()) + float(gold_cost))
    if max_cost <= 0:
        max_cost = 1.0

    # violation multiplier sweep across restarts (geometric 1 -> 100), like reasondb
    betas = np.geomspace(1.0, 100.0, num=max(restarts, 1))

    best = None
    best_key = None
    for r in range(max(restarts, 1)):
        g = torch.Generator().manual_seed(seed * 1000 + r)
        # raw params -> theta = sigmoid(raw)*(max-min)+min ; init wide unsure band
        theta_u_raw = torch.empty(K, dtype=torch.float64).uniform_(-2, 2, generator=g).requires_grad_(True)
        theta_l_raw = torch.empty(K, dtype=torch.float64).uniform_(-2, 2, generator=g).requires_grad_(True)
        if optimize_pick:
            pick_raw = torch.zeros(K, dtype=torch.float64).requires_grad_(True)
            params = [theta_u_raw, theta_l_raw, pick_raw]
        else:
            pick_raw = None
            params = [theta_u_raw, theta_l_raw]
        opt = torch.optim.Adam(params, lr=lr)
        beta_pen = float(betas[r])

        for i in range(steps):
            frac = i / max(steps - 1, 1)
            tau = tau_begin * (tau_end / tau_begin) ** frac
            opt.zero_grad()
            theta_u = torch.sigmoid(theta_u_raw) * (THRESHOLD_MAX - THRESHOLD_MIN) + THRESHOLD_MIN
            theta_l = torch.sigmoid(theta_l_raw) * (THRESHOLD_MAX - THRESHOLD_MIN) + THRESHOLD_MIN
            pick = torch.sigmoid(pick_raw / 0.05) if optimize_pick else torch.ones(K, dtype=torch.float64)

            m = torch.ones(N, dtype=torch.float64)        # unsure mass entering op 0
            pred_yes = torch.zeros(N, dtype=torch.float64)
            proc_cost = torch.zeros((), dtype=torch.float64)
            for k in range(K):
                keep = (lo_t[k] - theta_u[k]) / tau
                disc = (theta_l[k] - lo_t[k]) / tau
                zero = torch.zeros(N, dtype=torch.float64)
                soft = torch.softmax(torch.stack([keep, disc, zero], dim=-1), dim=-1)
                pk, pd = soft[:, 0], soft[:, 1]
                sig = pick[k]
                pred_yes = pred_yes + sig * m * pk
                proc_cost = proc_cost + m.sum() * op_cost_t[k]   # op k sees all unsure mass entering it
                m = m * (1.0 - sig * (pk + pd))                  # remaining unsure flows on
            pred_yes = pred_yes + m * gold_t                     # gold resolves residual unsure
            proc_cost = proc_cost + m.sum() * gold_cost

            tp = torch.sum(pred_yes * gold_t)
            fp = torch.sum(pred_yes * (1.0 - gold_t))
            fn = torch.sum((1.0 - pred_yes) * gold_t)
            prec_lb = _beta_lower_bound(tp, fp, confidence)
            rec_lb = _beta_lower_bound(tp, fn, confidence)

            l_cost = proc_cost / max_cost
            l_prec = torch.relu(torch.tensor(target_prec, dtype=torch.float64) - prec_lb)
            l_rec = torch.relu(torch.tensor(target_rec, dtype=torch.float64) - rec_lb)
            loss = beta_pen * (l_prec + l_rec) + l_cost
            loss.backward()
            opt.step()

        with torch.no_grad():
            theta_u = (torch.sigmoid(theta_u_raw) * (THRESHOLD_MAX - THRESHOLD_MIN) + THRESHOLD_MIN).numpy()
            theta_l = (torch.sigmoid(theta_l_raw) * (THRESHOLD_MAX - THRESHOLD_MIN) + THRESHOLD_MIN).numpy()
            pick = (torch.sigmoid(pick_raw / 0.05).numpy() > 0.5).astype(float) if optimize_pick else np.ones(K)

        pred, esc = simulate_hard(log_odds_list, theta_l, theta_u, pick, gold)
        prec_lb, rec_lb = _hard_bounds(pred, gold, confidence)
        esc_rate = float(esc.mean())
        feasible = (prec_lb >= target_prec) and (rec_lb >= target_rec)
        # rank: feasible first, then least escalation; else least total shortfall.
        violation = max(0.0, target_prec - prec_lb) + max(0.0, target_rec - rec_lb)
        key = (0 if feasible else 1, esc_rate if feasible else violation)
        if best_key is None or key < best_key:
            best_key = key
            best = Plan(theta_lower=theta_l, theta_upper=theta_u, pick=pick,
                        target_prec=target_prec, target_rec=target_rec,
                        train_esc=esc_rate, train_prec_lb=prec_lb, train_rec_lb=rec_lb,
                        feasible=feasible)
    return best
