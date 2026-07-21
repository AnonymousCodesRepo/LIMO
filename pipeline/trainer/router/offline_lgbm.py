"""Batch fit a LightGBM-shaped router head from offline rollouts.

Lifts the online buffer-fit logic from ``pipeline/router/lightgbm_router.py``
into a one-shot batch fit. The resulting checkpoint is consumed by
``LightGBMRouter.load_offline_checkpoint`` so the eval pipeline can
warm-start the router and either freeze it or keep updating online.

Unlike the online refit (streaming buffer, no IPS, fixed cadence under a
lock), the offline trainer takes the *full* synthetic rollout buffer at
once, applies importance-sampling weights ``1 / q`` (acquisition was
non-uniform), and holds out a synthetic-validation split for diagnostics.

Checkpoint format
-----------------
A pickled dict (the ``LightGBMRouterCheckpoint`` dataclass dumped as
plain dict for forward compatibility):

    {
        "model": <fitted CalibratedClassifierCV or HGB>,
        "feature_dim": 34,
        "use_care_features": True,
        "threshold": 0.5,
        "trained_on": {"n_pos": ..., "n_neg": ...},
        "metrics": {...},
    }

The eval-time loader instantiates a fresh LightGBMRouter, attaches this
model, and flips the trained flag.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.label_synth.types import Rollout, load_rollouts


@dataclass
class LightGBMRouterCheckpoint:
    """Plain-data wrapper for a pickled offline-fit router."""

    model: Any                                 # fitted sklearn estimator
    feature_dim: int = 34
    use_care_features: bool = True
    threshold: float = 0.5
    trained_on: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def _split_train_val(
    n: int, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * float(val_frac)))
    if n_val < 1:
        n_val = max(1, n // 10)
    val = idx[:n_val]
    tr = idx[n_val:]
    return tr, val


def _binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _build_base_estimator(
    *, backend: str, n_estimators: int, learning_rate: float,
    num_leaves: int, min_child_samples: int, reg_lambda: float,
    class_weight: str,
    mlp_hidden: tuple[int, ...] = (256, 256),
    mlp_alpha: float = 1e-4,
    mlp_max_iter: int = 300,
):
    """Build the GBDT base estimator. Mirrors LightGBMRouter._fit's
    backend selection so the offline-fit checkpoint plugs into the
    runtime router cleanly.
    """
    cw = "balanced" if class_weight == "balanced" else None
    if backend == "logreg":
        # Linear head: standardized logistic regression (hand-composed so
        # the IPS sample_weight path survives — see pipeline/router/_logreg).
        from pipeline.router._logreg import ScaledLogisticRegression
        return ScaledLogisticRegression(
            C=1.0 / max(float(reg_lambda), 1e-12),
            class_weight=cw,
        )
    if backend == "mlp":
        # Neural-net router head: a standardized MLP. MLPClassifier has no
        # sample_weight, so the caller's try/except falls back to an
        # unweighted fit (the IPS weights do not apply to this backend).
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(int(h) for h in mlp_hidden),
                alpha=float(mlp_alpha),
                max_iter=int(mlp_max_iter),
                early_stopping=True,
                n_iter_no_change=20,
                random_state=0,
            ),
        )
    if backend == "rf":
        # Random Forest — bagging ensemble; supports class_weight natively
        # for the agreement-label imbalance.
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_child_samples),
            class_weight=cw,
            n_jobs=1,
            random_state=0,
        )
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        kwargs = dict(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child_samples),
            reg_lambda=float(reg_lambda),
            verbose=-1,
            # Pin to single-threaded fits: on many-core machines LightGBM's
            # default one-thread-per-core spawn overhead dominates for this
            # small problem (few rows, few features).
            n_jobs=1,
            num_threads=1,
        )
        if cw == "balanced":
            kwargs["class_weight"] = "balanced"
        return LGBMClassifier(**kwargs)
    # Default: sklearn HistGradientBoostingClassifier.
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=int(n_estimators),
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(num_leaves),
        min_samples_leaf=int(min_child_samples),
        l2_regularization=float(reg_lambda),
        class_weight=cw,
    )


def _fit_calibrated_hgb(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, *,
    n_estimators: int, learning_rate: float, num_leaves: int,
    min_child_samples: int, reg_lambda: float,
    calibration_method: str, calibration_cv: int,
    class_weight: str,
    backend: str = "sklearn_hgb",
    mlp_hidden: tuple[int, ...] = (256, 256),
    mlp_alpha: float = 1e-4,
    mlp_max_iter: int = 300,
) -> Any:
    """Fit a calibrated GBDT classifier. Mirrors LightGBMRouter._fit."""
    from sklearn.calibration import CalibratedClassifierCV

    base = _build_base_estimator(
        backend=backend,
        n_estimators=n_estimators, learning_rate=learning_rate,
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        reg_lambda=reg_lambda, class_weight=class_weight,
        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, mlp_max_iter=mlp_max_iter,
    )
    if calibration_method == "auto":
        method = "sigmoid" if len(y) < 200 else "isotonic"
    elif calibration_method == "none":
        method = None
    else:
        method = calibration_method

    if method is None:
        try:
            base.fit(X, y, sample_weight=w)
        except Exception:
            base.fit(X, y)
        return base

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    cv = max(2, min(int(calibration_cv), n_pos, n_neg))
    try:
        m = CalibratedClassifierCV(base, method=method, cv=cv)
        m.fit(X, y, sample_weight=w)
        return m
    except Exception:
        # Fall back to uncalibrated HGB rather than failing the run.
        try:
            base.fit(X, y, sample_weight=w)
        except Exception:
            base.fit(X, y)
        return base


def train_router_offline(
    *,
    rollouts_path: str | Path,
    output_path: str | Path,
    val_frac: float = 0.15,
    seed: int = 0,
    threshold: float = 0.5,
    # Hyperparameters mirror LightGBMRouter defaults.
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 15,
    min_child_samples: int = 5,
    reg_lambda: float = 1.0,
    calibration_method: str = "auto",
    calibration_cv: int = 3,
    class_weight: str = "none",
    ips: bool = True,
    ips_weight_clip: tuple[float, float] = (0.1, 50.0),
    # GBDT backend. "lightgbm" uses the LightGBM library; the default
    # "sklearn_hgb" uses sklearn's HistGradientBoostingClassifier
    # (functionally equivalent here, avoids LightGBM's matplotlib import
    # path under a NumPy 1.x/2.x ABI mismatch).
    backend: str = "sklearn_hgb",
    # Feature-slicing knob. label_synth produces 34-d vectors (14 point +
    # 20 CARE-aggregate) matching the online router. When the synth
    # pipeline retrieved no experiences (the default), the CARE block is
    # identically zero; a head trained on those columns generalises poorly
    # to nonzero eval-time blocks. feature_dim=14 slices to the point block,
    # pairing cleanly with --lgbm-no-care-features at eval time.
    feature_dim: int | None = None,
    # MLP-router backend HPs (used only when backend == "mlp").
    mlp_hidden: tuple[int, ...] = (256, 256),
    mlp_alpha: float = 1e-4,
    mlp_max_iter: int = 300,
) -> dict[str, Any]:
    """Fit a LightGBM-shaped router head from a rollout JSONL.

    Returns a metrics dict including held-out synthetic log-loss and the
    decision-threshold diagnostic at the configured threshold. The
    pickled checkpoint is written to ``output_path``.
    """
    rollouts = load_rollouts(rollouts_path)
    rollouts = [r for r in rollouts if r.z is not None]
    if not rollouts:
        raise RuntimeError(f"no labeled rollouts in {rollouts_path}")
    X = np.asarray([r.features for r in rollouts], dtype=np.float64)
    if feature_dim is not None and feature_dim != X.shape[1]:
        if feature_dim > X.shape[1]:
            raise ValueError(
                f"feature_dim={feature_dim} but rollouts only have "
                f"{X.shape[1]} features"
            )
        X = X[:, :feature_dim]
    y = np.asarray([int(r.z) for r in rollouts], dtype=np.int64)
    # IPS weight construction. The recorded ``q`` is the per-pool
    # selection probability (sums to 1 over the candidate pool). The
    # unbiased active-learning estimator weights each example by
    # ``p(x) / q(x)`` with a uniform prior ``p(x) = 1 / N``, i.e. the
    # weight ``1 / (N * q)`` — O(1) when q is roughly uniform and larger
    # when q under-sampled the example.
    n_pool = max(len(rollouts), 1)
    if ips:
        raw_w = np.asarray(
            [1.0 / max(r.q * n_pool, 1e-6) for r in rollouts], dtype=np.float64,
        )
        # Center to mean 1 so the absolute weight scale doesn't shift HGB's
        # learning rate, and so the weighted log-loss/threshold diagnostics
        # stay interpretable.
        raw_w = raw_w / max(float(raw_w.mean()), 1e-12)
        w = np.clip(raw_w, ips_weight_clip[0], ips_weight_clip[1])
    else:
        # ips=False: fit the head with uniform example weights (no
        # inverse-propensity correction for the active-sampling bias).
        w = np.ones(len(rollouts), dtype=np.float64)
    feat_dim = X.shape[1]

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "rollouts collapsed to a single class — refusing to fit "
            "(would produce a degenerate router). Increase budget or "
            "rebalance synthetic-query distribution."
        )

    tr_idx, val_idx = _split_train_val(len(rollouts), val_frac, seed)
    t0 = time.time()
    model = _fit_calibrated_hgb(
        X[tr_idx], y[tr_idx], w[tr_idx],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        reg_lambda=reg_lambda,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        class_weight=class_weight,
        backend=backend,
        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, mlp_max_iter=mlp_max_iter,
    )
    fit_seconds = time.time() - t0

    # Held-out diagnostics on the synthetic validation split.
    p_val = model.predict_proba(X[val_idx])[:, 1]
    val_loss = _binary_log_loss(p_val, y[val_idx].astype(np.float64))
    pred_at_thr = (p_val >= threshold).astype(np.int64)
    val_acc = float((pred_at_thr == y[val_idx]).mean())
    val_pos_rate = float((y[val_idx] == 1).mean())

    metrics = {
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "n_pos_train": int((y[tr_idx] == 1).sum()),
        "n_neg_train": int((y[tr_idx] == 0).sum()),
        "val_log_loss": val_loss,
        "val_accuracy_at_threshold": val_acc,
        "val_pos_rate": val_pos_rate,
        "fit_seconds": round(fit_seconds, 2),
        "ips_weight_min": float(w.min()),
        "ips_weight_max": float(w.max()),
        "ips_weight_mean": float(w.mean()),
    }
    config = dict(
        feature_dim=int(feat_dim),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        reg_lambda=float(reg_lambda),
        calibration_method=calibration_method,
        calibration_cv=int(calibration_cv),
        class_weight=class_weight,
        backend=backend,
        ips_weight_clip=list(ips_weight_clip),
        val_frac=float(val_frac),
        seed=int(seed),
        threshold=float(threshold),
    )
    ckpt = LightGBMRouterCheckpoint(
        model=model,
        feature_dim=int(feat_dim),
        use_care_features=(feat_dim > 14),
        threshold=float(threshold),
        trained_on={
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
        },
        metrics=metrics,
        config=config,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(ckpt, f)
    # Sidecar metrics JSON for easy inspection without unpickling.
    with open(out.with_suffix(".metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "config": config,
                   "trained_on": ckpt.trained_on,
                   "feature_dim": ckpt.feature_dim,
                   "use_care_features": ckpt.use_care_features,
                   "threshold": ckpt.threshold}, f, indent=2)
    return metrics


def load_checkpoint(path: str | Path) -> LightGBMRouterCheckpoint:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, LightGBMRouterCheckpoint):
        return obj
    if isinstance(obj, dict):
        return LightGBMRouterCheckpoint(**obj)
    raise TypeError(f"unrecognized checkpoint type: {type(obj).__name__}")
