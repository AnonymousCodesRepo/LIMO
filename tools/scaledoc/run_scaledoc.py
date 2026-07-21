"""Faithful ScaleDoc orchestrator for the LIMO baseline port.

COPY THIS INTO ScaleDoc/src/ AND RUN FROM THERE (it imports ScaleDoc's own
model.py / utils/train.py / utils/sample.py / cascade.py).

Per (dataset, query) it runs ScaleDoc's real pipeline, only swapping:
  * embeddings  -> all-mpnet-base-v2 (768d), the same model our other methods use
  * oracle      -> our large model (Llama-3.1-70B), read from the cached labels
                   (so train labels, escalated answers all come from the 70B)
  * eval        -> accuracy / F1 vs EXPERT GOLD (not vs the 70B)
  * knob        -> sweeps target_acc to trace a cost/escalation frontier

Final per-document prediction over all docs of a query:
  proxy-confident-positive band -> 1
  proxy-confident-negative band -> 0
  escalated (7% train sample  U  5% calib sample  U  uncertain middle band) -> 70B label
Escalation rate = |escalated| / |docs|. USD = |escalated| * per-70B-call cost.

Outputs one CSV row per (dataset, query, target_acc).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# ScaleDoc's own code (cwd must be ScaleDoc/src)
from model import Encoder, ContrastiveLoss
from utils.train import train_cl_encoder, eval_cl_encoder
from utils.sample import aug_split
from cascade import (calibrate_sampling, smooth_distr,
                     select_sim_filter_AT, select_sim_filterB)

# Threshold selector: `B` (select_sim_filterB) is the F1-based selector matching
# the paper's Algorithm 2 ("Acc(l,r)" = F1), which produces the two-threshold
# cascade; ScaleDoc ships `AT` active. Override with SCALEDOC_SELECTOR=AT|B.
SELECTOR = {"AT": select_sim_filter_AT, "B": select_sim_filterB}[
    os.environ.get("SCALEDOC_SELECTOR", "B")]

DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 64
# Wide target-accuracy grid, dense at the low end: AT's constraint is the
# predicted-positive rate (~class prior on imbalanced data), so only small alpha
# is feasible -- sweeping from very small values is what makes AT vary the
# escalation rate. Override with SCALEDOC_ALPHAS="0.01,0.05,...".
_DEFAULT_ALPHAS = ("0.05,0.10,0.13,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,"
                   "0.85,0.88,0.90,0.92,0.94,0.95,0.97,0.99")
ALPHAS = [float(x) for x in os.environ.get("SCALEDOC_ALPHAS", _DEFAULT_ALPHAS).split(",")]
# MODE=sweep (default): directly enumerate the two cascade thresholds to trace
# ScaleDoc's achievable (escalation, accuracy) frontier at each escalation
# budget. MODE=alpha: target-accuracy selection.
MODE = os.environ.get("SCALEDOC_MODE", "sweep")
# Escalation-budget grid for the threshold sweep. Floor ~0.12 (7% train + 5%
# calib are always sent to the 70B); points below the floor collapse to it.
E_GRID = [round(0.10 + 0.05 * k, 3) for k in range(19)]  # 0.10 .. 1.00
TRAIN_RATIO = 0.07
CALIB_RATIO = 0.05
TRAIN_VALID_SPLIT = 0.7
LARGE_IN_COST = 0.52e-6
LARGE_OUT_COST = 0.75e-6


def seed_all(s=43):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_encoder(sample_data, data_test):
    """Mirror ScaleDoc train_cl.model_train_eval (two contrastive phases)."""
    train_points = max(1, int(TRAIN_VALID_SPLIT * len(sample_data)))
    data_train, data_valid, _ = aug_split(sample_data, train_points, 3)
    train_ld = DataLoader(data_train, 128, shuffle=True)
    valid_ld = DataLoader(data_valid, 128, shuffle=False)

    enc = Encoder(DIM)
    # phase 1: qsim only
    loss = ContrastiveLoss(b1=0.0, b2=0.0, b3=1.0)
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad],
                           lr=1e-3, weight_decay=1e-4)
    train_cl_encoder(enc, loss, opt, 50, DEVICE, 128, "_tmp", 0, train_ld, valid_ld,
                     early_stop=True)
    if os.path.exists("../models/_tmp_q0_enc.pt"):
        enc.load_state_dict(torch.load("../models/_tmp_q0_enc.pt", weights_only=True))
    # phase 2: supcon + polar
    loss = ContrastiveLoss(b1=0.2, b2=0.8, b3=0.0)
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad],
                           lr=1e-3, weight_decay=1e-4)
    train_cl_encoder(enc, loss, opt, 30, DEVICE, 128, "_tmp", 0, train_ld, valid_ld,
                     early_stop=True)

    test_ld = DataLoader(data_test, 128, shuffle=False)
    q_ref, doc_ref = eval_cl_encoder(enc, DEVICE, test_ld, "_tmp", 0)
    return q_ref, doc_ref


def cascade_distributions(q_ref, doc_ref):
    """Refined cosine sim + reconstructed pos/neg score curves (ScaleDoc Sec.4)."""
    d_emb = torch.cat([d[0] for d in doc_ref], dim=0)
    d_gt = torch.cat([d[1] for d in doc_ref], dim=0).cpu().numpy().ravel().astype(int)
    d_idx = torch.cat([d[2] for d in doc_ref], dim=0).cpu().numpy().ravel().astype(int)
    q = torch.nn.functional.normalize(q_ref.unsqueeze(0), p=2, dim=1)
    d = torch.nn.functional.normalize(d_emb, p=2, dim=1)
    cos = torch.nn.functional.cosine_similarity(q, d, dim=1).cpu().numpy()

    pos_idx = np.where(d_gt == 1)[0]
    neg_idx = np.where(d_gt == 0)[0]
    hist, bins = np.histogram(cos, bins=NUM_BINS)
    try:
        calib = calibrate_sampling(CALIB_RATIO, hist, bins, cos, pos_idx, neg_idx)
    except ValueError:
        # every bin sampled 0 -> fall back to a small random calib set
        k = max(1, int(CALIB_RATIO * len(cos)))
        calib = np.random.choice(len(cos), min(k, len(cos)), replace=False)

    bins_center = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    pos_ = np.array([j for j in calib if j in set(pos_idx.tolist())])
    neg_ = np.array([j for j in calib if j in set(neg_idx.tolist())])
    pos_cos = cos[pos_] if pos_.size else np.array([])
    neg_cos = cos[neg_] if neg_.size else np.array([])
    hist_pos, _ = np.histogram(pos_cos, bins=bins)
    hist_neg, _ = np.histogram(neg_cos, bins=bins)
    # jitter + smooth (verbatim ScaleDoc)
    hist_pos = hist_pos + np.random.choice([0, 0.1, 0.2], size=hist_pos.shape, p=[0.6, 0.3, 0.1])
    hist_neg = hist_neg + np.random.choice([0, 0.1, 0.2], size=hist_neg.shape, p=[0.6, 0.3, 0.1])
    _, neg_ma = smooth_distr(bins, hist_neg, window_size=5)
    _, pos_ma = smooth_distr(bins, hist_pos, window_size=5)
    return cos, d_gt, d_idx, bins, bins_center, pos_ma, neg_ma, calib


def bounds_to_mask(cos, bins, steps, le, re):
    """ScaleDoc apply_bounds masking, without its vs-oracle F1."""
    le_idx = max(0, np.where(steps == le)[0][0] * 2 - 1)
    re_idx = min(len(bins) - 1, np.where(steps == re)[0][0] * 2 + 1)
    le_b, re_b = bins[le_idx], bins[re_idx]
    return np.where(cos < le_b, 0, np.where(cos >= re_b, 1, -1))


def confusion(y_pred, y_true):
    return (int(((y_pred == 1) & (y_true == 1)).sum()),
            int(((y_pred == 1) & (y_true == 0)).sum()),
            int(((y_pred == 0) & (y_true == 1)).sum()),
            int(((y_pred == 0) & (y_true == 0)).sum()))


def degenerate_rows(ds, qname, qid, n, n_train, valid_ids, gold, label70, avg_large_in, status):
    """All-escalate fallback: ScaleDoc's proxy cannot be trained (too few / single
    class samples), so every doc is answered by the 70B (escalation 100%,
    accuracy = 70B accuracy). Emits one row per alpha (alpha mode) or per
    escalation budget (sweep mode)."""
    y_true = np.array([gold[str(d)] for d in valid_ids])
    y_pred = np.array([label70[d] for d in valid_ids])
    tp, fp, fn, tn = confusion(y_pred, y_true)
    base = {"dataset": ds, "query_name": qname, "qid": qid,
            "n_valid": n, "n_train_sample": n_train, "n_calib": 0, "n_uncertain": n,
            "esc_rate": 1.0,
            "acc": float(np.mean(y_pred == y_true)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "acc_vs_70b": 1.0,
            "usd": n * (avg_large_in * LARGE_IN_COST + LARGE_OUT_COST),
            "status": status}
    if MODE == "sweep":
        return [{**base, "budget_e": e, "mode": "sweep"} for e in E_GRID]
    return [{**base, "target_acc": a} for a in ALPHAS]


def run_query(ds, qid, qname, data, avg_large_in, fout):
    seed_all(43)
    gt = json.loads((data / "gt" / f"{ds}_res_{qid}.json").read_text())
    gold = json.loads((data / "gold" / f"{ds}_q{qid}.json").read_text())
    valid_ids = [r[0] for r in gt]
    label70 = {r[0]: (1 if r[1]["label"] == "Yes" else 0) for r in gt}
    n = len(valid_ids)

    demb_full = torch.load(data / "embeds" / f"{ds}_embeds_mpnet.pt", weights_only=True)
    qemb_full = torch.load(data / "embeds" / f"{ds}_qemb_mpnet.pt", weights_only=True)
    d_e = demb_full[valid_ids].float()
    q_e = qemb_full[qid].float().unsqueeze(0).expand(n, -1)
    # ScaleDoc's aug_split assumes the feature tensors live on CUDA (its original
    # embeddings came back from the GPU encoder); keep that invariant here.
    emb = torch.stack([d_e, q_e], dim=1).to(DEVICE)    # [n,2,768]
    labels = torch.tensor([float(label70[i]) for i in valid_ids]).unsqueeze(1).to(DEVICE)
    idx_doc = torch.tensor(valid_ids).unsqueeze(1).to(DEVICE)  # carries doc_id

    pos = np.arange(n)
    sample_pos = np.random.RandomState(43).choice(n, max(1, int(TRAIN_RATIO * n)), replace=False)
    test_pos = np.array([p for p in pos if p not in set(sample_pos.tolist())])
    sample_doc_ids = [valid_ids[p] for p in sample_pos]

    def emit(rows):
        for r in rows:
            fout.write(json.dumps(r) + "\n")
        fout.flush()

    # too few labeled samples or single-class -> ScaleDoc cannot train; record degenerate
    lab_sample = [label70[d] for d in sample_doc_ids]
    if len(test_pos) < 2 or len(set(lab_sample)) < 2:
        emit(degenerate_rows(ds, qname, qid, n, len(sample_doc_ids), valid_ids, gold,
                             label70, avg_large_in, "degenerate-too-small"))
        return

    timed = os.environ.get("SCALEDOC_TIME")

    def _sync():
        if timed and torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        sp = torch.as_tensor(sample_pos, dtype=torch.long, device=DEVICE)
        tp = torch.as_tensor(test_pos, dtype=torch.long, device=DEVICE)
        sample_data = TensorDataset(emb[sp], labels[sp], idx_doc[sp])
        data_test = TensorDataset(emb[tp], labels[tp], idx_doc[tp])
        _sync(); t0 = time.perf_counter()
        q_ref, doc_ref = train_encoder(sample_data, data_test)
        _sync(); t1 = time.perf_counter()
        cos, d_gt, d_idx, bins, bins_center, pos_ma, neg_ma, calib = \
            cascade_distributions(q_ref, doc_ref)
        _sync(); t2 = time.perf_counter()
    except Exception as e:
        # ScaleDoc's aug_split / cascade can divide-by-zero on tiny single-class
        # samples; fall back to the honest all-escalate point.
        emit(degenerate_rows(ds, qname, qid, n, len(sample_doc_ids), valid_ids, gold,
                             label70, avg_large_in, f"degenerate-train-fail:{type(e).__name__}"))
        return
    steps = bins[::2]
    calib_doc_ids = set(int(d_idx[p]) for p in calib)

    if os.environ.get("SCALEDOC_DEBUG"):
        cp = cos[d_gt == 1]; cn = cos[d_gt == 0]
        print(f"  [dbg] {ds} q{qid} n_test={len(cos)} pos={len(cp)} neg={len(cn)} "
              f"cos_pos={cp.mean():.3f}±{cp.std():.3f} cos_neg={cn.mean():.3f}±{cn.std():.3f} "
              f"cos_range=[{cos.min():.3f},{cos.max():.3f}] calib={len(calib_doc_ids)}", flush=True)

    if MODE == "diag":
        # Proxy-ONLY performance on the full (test) set: no escalation, no 70B.
        # Best single-threshold accuracy / F1 of the trained proxy's cosine
        # scores vs gold -- i.e. how good the small model is on its own.
        gt_test = np.array([gold[str(int(d))] for d in d_idx]).astype(int)
        y70 = d_gt.astype(int)
        n_t = len(cos)
        order = np.argsort(-cos)                      # highest similarity first
        g = gt_test[order]
        cum = np.concatenate([[0], np.cumsum(g)])     # positives among top-k
        k = np.arange(0, n_t + 1)
        TP = cum; FP = k - TP; FN = int(g.sum()) - TP; TN = (n_t - k) - FN
        acc = (TP + TN) / n_t
        denom = 2 * TP + FP + FN
        f1 = np.where(denom > 0, 2 * TP / denom, 0.0)
        emit([{
            "dataset": ds, "query_name": qname, "qid": qid, "n_test": n_t,
            "proxy_acc": float(acc.max()), "proxy_f1": float(f1.max()),
            "large_acc": float((y70 == gt_test).mean()),
            "large_f1": float(f1_score(gt_test, y70, zero_division=0)),
            "gold_pos_rate": float(gt_test.mean()), "status": "ok",
        }])
        return

    if MODE == "sweep":
        # Directly enumerate the two cascade thresholds (le<=re) over the 64-bin
        # step grid to trace the achievable (escalation, accuracy) frontier. The
        # cascade operating point depends only on (le, re), not on which selector
        # picked them -- so this is method-neutral. 7% train + 5% calib are always
        # escalated (calib is charged per the run config).
        gold_test = np.array([gold[str(int(d))] for d in d_idx])
        y70_test = d_gt.astype(int)
        calib_mask = np.zeros(len(cos), dtype=bool)
        calib_mask[np.asarray(calib, dtype=int)] = True
        gold_train = np.array([gold[str(int(d))] for d in sample_doc_ids])
        y70_train = np.array([label70[int(d)] for d in sample_doc_ids])
        n_train = len(sample_doc_ids)
        pts = []  # (esc, acc, tp, fp, fn)
        for i in range(len(steps)):
            le = steps[i]
            for j in range(i, len(steps)):
                re = steps[j]
                band = np.where(cos < le, 0, np.where(cos >= re, 1, -1))
                esc_test = (band == -1) | calib_mask
                pred_test = np.where(esc_test, y70_test, band)
                yp = np.concatenate([pred_test, y70_train])
                yt = np.concatenate([gold_test, gold_train])
                tp, fp, fn, _ = confusion(yp, yt)
                acc = float((yp == yt).mean())
                esc = (int(esc_test.sum()) + n_train) / n
                pts.append((esc, acc, tp, fp, fn))
        rows = []
        min_esc = min(p[0] for p in pts)
        for e in E_GRID:
            cand = [p for p in pts if p[0] <= e + 1e-9]
            if not cand:  # budget below the ~12% floor -> best point AT the floor
                cand = [p for p in pts if abs(p[0] - min_esc) < 1e-9]
            esc_b, acc_b, tp_b, fp_b, fn_b = max(cand, key=lambda p: p[1])
            rows.append({
                "dataset": ds, "query_name": qname, "qid": qid, "budget_e": e,
                "n_valid": n, "n_train_sample": n_train, "n_calib": len(calib_doc_ids),
                "esc_rate": esc_b, "acc": acc_b, "f1": tp_b / (tp_b + 0.5 * (fp_b + fn_b))
                if (tp_b + fp_b + fn_b) > 0 else 0.0,
                "tp": tp_b, "fp": fp_b, "fn": fn_b, "mode": "sweep", "status": "ok",
            })
        if timed:
            _sync(); t3 = time.perf_counter()
            print(f"  [time] {ds} q{qid} n={n} | encoder_train={t1-t0:.2f}s "
                  f"cascade_dist={t2-t1:.2f}s sweep({len(pts)} pairs)={t3-t2:.2f}s "
                  f"| total={t3-t0:.2f}s", flush=True)
        emit(rows)
        return

    rows = []
    for a in ALPHAS:
        try:
            le, re, _ = SELECTOR(steps, bins_center, pos_ma, neg_ma, bins[0], bins[-1], a)
            mask = bounds_to_mask(cos, bins, steps, le, re)
            if os.environ.get("SCALEDOC_DEBUG"):
                print(f"  [dbg] a={a} le={le:.3f} re={re:.3f} bands(neg/pos/unc)="
                      f"{int((mask==0).sum())}/{int((mask==1).sum())}/{int((mask==-1).sum())}",
                      flush=True)
        except Exception as e:
            if os.environ.get("SCALEDOC_DEBUG"):
                print(f"  [dbg] a={a} EXC {type(e).__name__}: {e}", flush=True)
            mask = np.full(len(cos), -1)  # degenerate -> escalate all test docs

        pred, escalated = {}, set()
        for p in range(len(cos)):
            did = int(d_idx[p])
            if mask[p] == -1:
                pred[did] = label70[did]; escalated.add(did)
            else:
                pred[did] = int(mask[p])
        for did in calib_doc_ids:                       # calib docs were oracle-labeled
            pred[did] = label70[did]; escalated.add(did)
        for did in sample_doc_ids:                      # 7% train sample -> oracle
            pred[did] = label70[did]; escalated.add(did)

        y_true = np.array([gold[str(d)] for d in valid_ids])
        y_pred = np.array([pred[d] for d in valid_ids])
        y70 = np.array([label70[d] for d in valid_ids])
        n_unc = int(np.sum(mask == -1))
        tp, fp, fn, tn = confusion(y_pred, y_true)
        rows.append({
            "dataset": ds, "query_name": qname, "qid": qid, "target_acc": a,
            "n_valid": n, "n_train_sample": len(sample_doc_ids),
            "n_calib": len(calib_doc_ids), "n_uncertain": n_unc,
            "esc_rate": len(escalated) / n,
            "acc": float(np.mean(y_pred == y_true)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acc_vs_70b": float(np.mean(y_pred == y70)),
            "usd": len(escalated) * (avg_large_in * LARGE_IN_COST + LARGE_OUT_COST),
            "status": "ok",
        })
    if timed:
        _sync(); t3 = time.perf_counter()
        print(f"  [time] {ds} q{qid} n={n} | encoder_train={t1-t0:.2f}s "
              f"cascade_dist(eval+cosine+calib)={t2-t1:.2f}s "
              f"alpha_sweep(6x thresh)={t3-t2:.3f}s | query_total={t3-t0:.2f}s",
              flush=True)
    emit(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", default="cuad,contract,opp115,hoc,sembench_medical")
    ap.add_argument("--qids", default="", help="optional comma list to filter qids (debug)")
    ap.add_argument("--out", required=True, help="results jsonl")
    args = ap.parse_args()
    qid_filter = {int(x) for x in args.qids.split(",") if x.strip()} if args.qids else None
    data = Path(args.data)
    Path("../models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    with open(args.out, "w") as fout:
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            meta = json.loads((data / "meta" / f"{ds}.json").read_text())
            avg_large_in = float(meta["avg_large_in_tokens"])
            qid_to_name = {int(k): v for k, v in meta["qid_to_name"].items()}
            for qid in sorted(qid_to_name):
                if qid_filter is not None and qid not in qid_filter:
                    continue
                qname = qid_to_name[qid]
                try:
                    run_query(ds, qid, qname, data, avg_large_in, fout)
                    print(f"[ok] {ds} q{qid} {qname}", flush=True)
                except Exception as e:
                    print(f"[ERR] {ds} q{qid} {qname}: {e}", flush=True)
                    traceback.print_exc()
                    fout.write(json.dumps({"dataset": ds, "query_name": qname, "qid": qid,
                                           "status": f"error: {e}"}) + "\n")
                    fout.flush()


if __name__ == "__main__":
    main()
