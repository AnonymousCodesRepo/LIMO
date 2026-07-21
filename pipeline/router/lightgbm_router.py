"""LightGBM router — the paper's learned router.

Implements the router contract (`route`, `should_escalate`,
`on_escalation_observed`) with a calibrated LightGBM classifier head.

Design choices
--------------
- **Escalation-outcome supervision (same as CARE):** every escalation
  produces one observation (φ_t, z_t = 1[small_pred == large_pred]).
  Ground truth is never read by this router.
- **14-d point features** (via `_point_features._featurize`): small-model
  output signals, length signals, and the data-point / query escalation
  history signals (read from the experience retriever's counters).
  Additionally mean-pools per-experience CARE features when an
  `experience_retriever` exposing `set_features_for(point, exps)` is
  provided.
- **Bootstrap → trained switch.** Until both class minima are met, the
  router routes via a confidence-threshold rule (`bootstrap_threshold`).
  Once met, fits a calibrated LightGBM and switches to its threshold.
- **Calibration is non-negotiable.** `CalibratedClassifierCV(method='sigmoid'
  if n<200 else 'isotonic', cv=5)` wraps every fit; without it the GBDT's
  `predict_proba` is overconfident and the threshold rule is useless.
- **Refit cadence.** Every `refit_every` PROCESSED DATA POINTS (paper §7.1
  semantics; points are counted in `should_escalate`). A due refit is
  skipped — and rescheduled to the next cadence point — when fewer than
  `refit_min_new_obs` new escalation observations arrived since the last
  actual refit. Training data is capped to the most recent
  `online_buffer_max` observations to bound memory and let drift propagate.

Cold-start: until enough class-balanced observations accumulate, the head
falls back to the confidence threshold.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Literal

import numpy as np

from pipeline.common.types import DataPoint, RunState

from pipeline.experience_retriever.feature_linucb_v3 import _FEATURE_DIM_V3

from ._base import BaseRouter, EscalationDecision
from ._point_features import _featurize as _point_featurize


class _CompiledHGB:
    """Pure-Python single-row predictor compiled from a fitted binary
    ``HistGradientBoostingClassifier``.

    sklearn's ``predict_proba`` spends tens of milliseconds per single-row
    call on validation plumbing while the tree walk is microseconds; this
    extracts each tree's node arrays into plain Python lists once per (re)fit
    and walks them directly.

    Compile-time self-check: predictions on fixed random rows must match
    sklearn's ``predict_proba`` to 1e-9, else ``__init__`` raises and the
    caller keeps the sklearn path — a mismatch degrades to slow-but-correct,
    never wrong.
    """

    def __init__(self, model: Any) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        if not isinstance(model, HistGradientBoostingClassifier):
            raise TypeError("not an uncalibrated HistGradientBoostingClassifier")
        if list(getattr(model, "classes_", [])) != [0, 1]:
            raise ValueError("binary {0,1} classifier required")
        self._baseline = float(np.ravel(model._baseline_prediction)[0])
        trees = []
        for predictors in model._predictors:
            if len(predictors) != 1:
                raise ValueError("multiclass not supported")
            nodes = predictors[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("categorical splits not supported")
            trees.append((
                nodes["feature_idx"].tolist(),
                nodes["num_threshold"].tolist(),
                nodes["left"].tolist(),
                nodes["right"].tolist(),
                nodes["is_leaf"].tolist(),
                nodes["value"].tolist(),
                nodes["missing_go_to_left"].tolist(),
            ))
        self._trees = trees
        self.n_features = int(model.n_features_in_)

        rng = np.random.default_rng(12345)
        # Mix wide-normal rows with [0, 1] rows: features blend probabilities/
        # cosines in [0, 1] with log/length values, so exercise both regimes
        # against sklearn before trusting the compiled walk.
        X = np.vstack([
            rng.standard_normal((24, self.n_features)),
            rng.random((24, self.n_features)),
        ])
        want = model.predict_proba(X)[:, 1]
        got = np.array([self.predict_p1(x) for x in X])
        if not np.allclose(want, got, atol=1e-9):
            raise ValueError("compiled predictor disagrees with sklearn")

    def predict_p1(self, x: np.ndarray) -> float:
        """P(class == 1) for one feature row. Microseconds per call."""
        raw = self._baseline
        xs = x.tolist()
        for feat, thr, left, right, leaf, val, miss in self._trees:
            i = 0
            while not leaf[i]:
                v = xs[feat[i]]
                if v != v:  # NaN
                    i = left[i] if miss[i] else right[i]
                elif v <= thr[i]:
                    i = left[i]
                else:
                    i = right[i]
            raw += val[i]
        if raw >= 0.0:
            return 1.0 / (1.0 + math.exp(-raw))
        e = math.exp(raw)
        return e / (1.0 + e)


class LightGBMRouter(BaseRouter):
    """Calibrated LightGBM router. See module docstring."""

    def __init__(
        self,
        *,
        experience_retriever: Any = None,
        bootstrap_threshold: float = 0.9,
        bootstrap_min_positives: int = 30,
        bootstrap_min_negatives: int = 30,
        threshold: float = 0.5,
        refit_every: int = 500,
        refit_min_new_obs: int = 10,
        online_buffer_max: int = 5000,
        # GBDT hyperparameters.
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 5,
        reg_lambda: float = 1.0,
        # Calibration knob (auto-selects per fit unless overridden).
        calibration_method: Literal["auto", "sigmoid", "isotonic", "none"] = "auto",
        calibration_cv: int = 5,
        # Backend: "lightgbm" requires the package and behaves like
        # "sklearn_hgb" (HistGradientBoostingClassifier) within rounding
        # error; sklearn_hgb is the default. "catboost" is a third GBDT
        # family with native categorical handling.
        backend: Literal["sklearn_hgb", "lightgbm", "catboost", "rf", "mlp", "logreg"] = "sklearn_hgb",
        # MLP-router backend HPs (used only when backend == "mlp").
        # Standardized sklearn MLPClassifier.
        mlp_hidden: tuple[int, ...] = (256, 256),
        mlp_alpha: float = 1e-4,
        mlp_max_iter: int = 300,
        # Feature ablation: when False, drop the mean-pooled CARE φ +
        # mean p_calib blocks and run on the 14-d point block alone.
        use_care_features: bool = True,
        # Class-weighting for the head. "balanced" uses sklearn's standard
        # n_samples / (n_classes * np.bincount(y)) — useful when the cascade
        # signal z = 1[small==large] is heavily skewed.
        class_weight: Literal["none", "balanced"] = "none",
        confidence_veto: float | None = None,
        # Optional offline pretraining: when set, load a checkpoint produced
        # by `pipeline.trainer.router.offline_lgbm.train_router_offline` and
        # treat the head as already trained. Online refits continue by
        # default; set `freeze_after_load=True` to disable them and keep the
        # offline-fit head fixed for the run.
        pretrained_checkpoint: str | None = None,
        freeze_after_load: bool = False,
    ) -> None:
        if not 0.0 <= bootstrap_threshold <= 1.0:
            raise ValueError(f"bootstrap_threshold out of range: {bootstrap_threshold}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold out of range: {threshold}")
        if confidence_veto is not None and not 0.0 <= confidence_veto <= 1.0:
            raise ValueError(f"confidence_veto out of range: {confidence_veto}")

        self.experience_retriever = experience_retriever
        self.bootstrap_threshold = float(bootstrap_threshold)
        self.bootstrap_min_pos = int(bootstrap_min_positives)
        self.bootstrap_min_neg = int(bootstrap_min_negatives)
        self.threshold = float(threshold)
        self.confidence_veto = (
            None if confidence_veto is None else float(confidence_veto)
        )
        self.refit_every = max(1, int(refit_every))
        self.refit_min_new_obs = max(0, int(refit_min_new_obs))
        self.online_buffer_max = int(online_buffer_max)

        self._lgbm_kwargs = dict(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child_samples),
            reg_lambda=float(reg_lambda),
            verbose=-1,
        )
        self.calibration_method = calibration_method
        self.calibration_cv = int(calibration_cv)
        self.backend = backend
        self._mlp_hidden = tuple(int(h) for h in mlp_hidden)
        self._mlp_alpha = float(mlp_alpha)
        self._mlp_max_iter = int(mlp_max_iter)
        self.use_care_features = bool(use_care_features)
        self.class_weight = class_weight

        # Online buffer. Entries are (features, label). Bounded by
        # online_buffer_max — older entries dropped on refit.
        self._buf_X: list[np.ndarray] = []
        self._buf_y: list[int] = []
        self._buf_lock = threading.Lock()

        # Trained model + bookkeeping. `_fast_model` is the compiled
        # single-row predictor for the CURRENT `_model` (None whenever the
        # model shape doesn't support compilation — calibrated wrappers,
        # non-HGB backends); readers fall back to sklearn predict_proba.
        self._model: Any = None
        self._fast_model: _CompiledHGB | None = None
        self._trained: bool = False
        self._n_obs: int = 0
        # Processed-data-point counter (incremented in should_escalate) —
        # the refit cadence runs on this, per the paper's §7.1 semantics.
        self._n_points: int = 0
        self._last_refit_at_points: int = 0
        self._obs_at_last_refit: int = 0
        self._n_refits: int = 0
        self._first_train_at_obs: int | None = None

        # Per-(doc_id, query_name) feature cache so on_escalation_observed
        # reuses the exact vector should_escalate built (no re-featurise).
        self._feat_cache: dict[tuple[int, str], np.ndarray] = {}
        self._feat_cache_lock = threading.Lock()

        # Diagnostic counters to debug routing behaviour.
        self._debug_n_called = 0
        self._debug_n_returned_true = 0
        self._debug_n_exceptions = 0
        self._debug_first_exception: str | None = None

        # Online refits are gated by this flag. When freeze_after_load is
        # True and a pretrained checkpoint is loaded, on_escalation_observed
        # still appends to the buffer (for diagnostics) but `_fit` is never
        # called — the offline-fit head stays fixed.
        self._freeze_online_fit: bool = bool(freeze_after_load)
        self._loaded_from_checkpoint: bool = False
        if pretrained_checkpoint is not None:
            self.load_offline_checkpoint(pretrained_checkpoint)

    # ---- Stage interface --------------------------------------------------

    def route(
        self, state: RunState, point: DataPoint
    ) -> Literal["small", "large"]:
        # Always small first; escalation handled in should_escalate.
        return "small"

    def should_escalate(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str | None,
        small_raw: str,
        small_confidence: float | None,
        *,
        small_features: dict | None = None,
        retrieved_experiences: list | None = None,
        **_kwargs,
    ) -> EscalationDecision:
        try:
            self._debug_n_called += 1
            # Every should_escalate call is one processed data point — the
            # refit cadence counts these, not escalation observations.
            with self._buf_lock:
                self._n_points += 1
            bootstrap = not self._trained
            # Match existing routers: forced escalation on UNKNOWN / Unsure.
            if small_pred in (None, "UNKNOWN", "Unsure"):
                self._debug_n_returned_true += 1
                return EscalationDecision(escalate=True, bootstrap=bootstrap)

            feats = self._build_features(point, small_features, retrieved_experiences)
            with self._feat_cache_lock:
                self._feat_cache[(point.doc_id, point.query_name)] = feats

            if bootstrap:
                # Bootstrap — confidence threshold.
                if small_confidence is None:
                    self._debug_n_returned_true += 1
                    return EscalationDecision(escalate=True, bootstrap=True)
                out = small_confidence < self.bootstrap_threshold
                if out:
                    self._debug_n_returned_true += 1
                return EscalationDecision(escalate=out, bootstrap=True)

            # Trained path: calibrated P(z=1 | x). z=1 ⇒ small agrees with large
            # (escalation is unnecessary). Escalate when P(z=1) < threshold.
            try:
                fast = self._fast_model
                if fast is not None:
                    p_z1 = fast.predict_p1(feats)
                else:
                    p_z1 = float(self._model.predict_proba(feats.reshape(1, -1))[0, 1])
            except Exception:
                # Defensive fallback: behave like bootstrap.
                fallback = small_confidence is None or small_confidence < self.bootstrap_threshold
                if fallback:
                    self._debug_n_returned_true += 1
                return EscalationDecision(escalate=fallback, bootstrap=False)
            out = p_z1 < self.threshold
            if (out and self.confidence_veto is not None
                    and small_confidence is not None
                    and small_confidence >= self.confidence_veto):
                out = False
            if out:
                self._debug_n_returned_true += 1
            return EscalationDecision(escalate=out, p_z1=p_z1, bootstrap=False)
        except Exception as e:
            import traceback
            self._debug_n_exceptions += 1
            if self._debug_first_exception is None:
                self._debug_first_exception = (
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
                import sys
                print(
                    f"[lightgbm_router] FIRST EXCEPTION in should_escalate: "
                    f"{self._debug_first_exception}",
                    file=sys.stderr, flush=True,
                )
            # Conservative fallback: escalate on error so the cascade still works.
            return EscalationDecision(escalate=True)

    def on_escalation_observed(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str | None,
        large_pred: str | None,
        *,
        small_features: dict | None = None,
        small_confidence: float | None = None,
        n_retrieved_experiences: int = 0,
        retrieved_experiences: list | None = None,
        **_kwargs,
    ) -> None:
        # Ignore unparseable outcomes (matches CARE convention).
        if small_pred in (None, "UNKNOWN") or large_pred in (None, "UNKNOWN"):
            return

        # Reuse the cached feature vector if should_escalate already built it.
        with self._feat_cache_lock:
            feats = self._feat_cache.pop(
                (point.doc_id, point.query_name), None
            )
        if feats is None:
            feats = self._build_features(point, small_features, retrieved_experiences)

        z = 1 if small_pred == large_pred else 0
        with self._buf_lock:
            self._buf_X.append(feats)
            self._buf_y.append(int(z))
            # Cap buffer.
            if len(self._buf_X) > self.online_buffer_max:
                drop = len(self._buf_X) - self.online_buffer_max
                self._buf_X = self._buf_X[drop:]
                self._buf_y = self._buf_y[drop:]
            self._n_obs += 1
            # Cadence in processed data points (paper §7.1). A due refit is
            # skipped — and rescheduled — when too few NEW escalation
            # observations arrived since the last actual refit, or when the
            # buffer lacks class balance.
            due = (
                self._n_points - self._last_refit_at_points >= self.refit_every
            )
            if not due:
                return
            enough_new = (
                self._n_obs - self._obs_at_last_refit >= self.refit_min_new_obs
            )
            balanced = (
                sum(self._buf_y) >= self.bootstrap_min_pos
                and (len(self._buf_y) - sum(self._buf_y)) >= self.bootstrap_min_neg
            )
            if not (enough_new and balanced):
                # Skip this scheduled refit; try again after another
                # refit_every processed points.
                self._last_refit_at_points = self._n_points
                return
            X = np.stack(self._buf_X)
            y = np.asarray(self._buf_y, dtype=np.int64)
            self._last_refit_at_points = self._n_points
            self._obs_at_last_refit = self._n_obs

        self._fit(X, y)

    # ---- Internals --------------------------------------------------------

    def _build_features(
        self,
        point: DataPoint,
        small_features: dict | None,
        retrieved_experiences: list | None,
    ) -> np.ndarray:
        """Build a fixed-dimensional feature vector for the GBDT head.

        Layout — fixed-width so the online buffer's feature dim never
        changes between calls:
            [ point_features (14, incl. doc/query history signals) ]
            [ mean-pooled CARE φ (phi_dim),  zeros when no exps ]
            [ mean estimated utility (1),    zero when no exps ]

        Mean pooling of the per-experience features plus the mean estimated
        utility follows the paper (§5.4). The set blocks are populated when
        the retriever is CARE-shaped and retrieved a non-empty set;
        otherwise they are zero, so the head sees a consistent shape from
        the first observation.
        """
        er = self.experience_retriever
        # Doc/query history signals come from the retriever's counters
        # (single source of truth, updated on every escalation observation).
        history = None
        if er is not None and hasattr(er, "history_stats"):
            try:
                history = er.history_stats(point)
            except Exception:
                history = None
        point_feats = _point_featurize(point, small_features, history=history)
        # Feature ablation: skip CARE blocks entirely.
        if not self.use_care_features:
            return point_feats
        # Default zero blocks (always present so feature dim is fixed).
        phi_dim = _FEATURE_DIM_V3  # CARE's v3 dim
        zero_phi_block = np.zeros(phi_dim, dtype=np.float64)
        zero_pcal_block = np.zeros(1, dtype=np.float64)

        if er is not None and retrieved_experiences:
            sf = er.set_features_for(point, retrieved_experiences)
            phi = sf.get("phi")
            p_cal = sf.get("p_calib")
            if phi is not None and phi.size > 0:
                phi = np.asarray(phi, dtype=np.float64)
                if phi.ndim == 1:
                    phi = phi.reshape(1, -1)
                if phi.shape[1] == phi_dim:
                    phi_block = phi.mean(axis=0)
                else:
                    phi_block = zero_phi_block
                if p_cal is not None and p_cal.size > 0:
                    p_cal = np.asarray(p_cal, dtype=np.float64).ravel()
                    pcal_block = np.array(
                        [float(p_cal.mean())], dtype=np.float64
                    )
                else:
                    pcal_block = zero_pcal_block
                return np.concatenate([point_feats, phi_block, pcal_block])
        return np.concatenate([point_feats, zero_phi_block, zero_pcal_block])

    def load_offline_checkpoint(self, path: str) -> None:
        """Load a checkpoint produced by trainer.router.offline_lgbm.

        Sets the model and trained flag so should_escalate() takes the
        trained path immediately. Online refits remain enabled unless
        ``freeze_after_load=True`` was passed at construction.
        """
        # Local import to avoid a circular dep at module load.
        from pipeline.trainer.router.offline_lgbm import load_checkpoint

        ckpt = load_checkpoint(path)
        self._model = ckpt.model
        try:
            self._fast_model = _CompiledHGB(ckpt.model)
        except Exception:
            self._fast_model = None
        self._trained = True
        self._loaded_from_checkpoint = True
        if self._first_train_at_obs is None:
            self._first_train_at_obs = 0
        # Honor the checkpoint's threshold if the caller didn't override it
        # at construction time. We can't tell construction-time intent here
        # cheaply, so we only adopt the checkpoint's threshold when ours is
        # the default 0.5 — this preserves explicit user overrides.
        if abs(self.threshold - 0.5) < 1e-9 and ckpt.threshold is not None:
            self.threshold = float(ckpt.threshold)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._freeze_online_fit:
            # Frozen offline-fit head: skip the refit but keep the buffer
            # populated for diagnostics in fit_info().
            return
        if len(np.unique(y)) < 2 or len(y) < 20:
            self._fast_model = None
            self._model = None
            self._fallback = float(np.mean(y)) if len(y) else 0.5
            return
        # Build the base estimator. `sklearn_hgb` is the default to avoid
        # lightgbm's matplotlib import path under threaded refits.
        cw = "balanced" if self.class_weight == "balanced" else None
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier

            kwargs = dict(self._lgbm_kwargs)
            if cw == "balanced":
                kwargs["class_weight"] = "balanced"
            base = LGBMClassifier(**kwargs)
        elif self.backend == "catboost":
            from catboost import CatBoostClassifier

            cb_kwargs = dict(
                iterations=self._lgbm_kwargs["n_estimators"],
                learning_rate=self._lgbm_kwargs["learning_rate"],
                depth=6,  # CatBoost uses tree depth, not leaf count
                l2_leaf_reg=self._lgbm_kwargs["reg_lambda"],
                min_data_in_leaf=self._lgbm_kwargs["min_child_samples"],
                thread_count=1,
                verbose=False,
            )
            if cw == "balanced":
                cb_kwargs["auto_class_weights"] = "Balanced"
            base = CatBoostClassifier(**cb_kwargs)
        elif self.backend == "rf":
            # Random Forest — bagging-based tree ensemble.
            from sklearn.ensemble import RandomForestClassifier

            base = RandomForestClassifier(
                n_estimators=self._lgbm_kwargs["n_estimators"],
                min_samples_leaf=self._lgbm_kwargs["min_child_samples"],
                class_weight=cw,
                n_jobs=1,
                random_state=0,
            )
        elif self.backend == "logreg":
            # Linear head: standardized logistic regression. C mirrors the
            # GBDT's reg_lambda knob (C = 1/λ).
            from ._logreg import ScaledLogisticRegression

            base = ScaledLogisticRegression(
                C=1.0 / max(float(self._lgbm_kwargs["reg_lambda"]), 1e-12),
                class_weight=cw,
            )
        elif self.backend == "mlp":
            # Neural-net head: standardized MLP. No sample_weight support;
            # online refit is unweighted (consistent with the offline fit).
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            base = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=self._mlp_hidden,
                    alpha=self._mlp_alpha,
                    max_iter=self._mlp_max_iter,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=0,
                ),
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier as _HGB

            base = _HGB(
                max_iter=self._lgbm_kwargs["n_estimators"],
                learning_rate=self._lgbm_kwargs["learning_rate"],
                max_leaf_nodes=self._lgbm_kwargs["num_leaves"],
                min_samples_leaf=self._lgbm_kwargs["min_child_samples"],
                l2_regularization=self._lgbm_kwargs["reg_lambda"],
                class_weight=cw,
            )

        from sklearn.calibration import CalibratedClassifierCV

        if self.calibration_method == "auto":
            method = "sigmoid" if len(y) < 200 else "isotonic"
        elif self.calibration_method == "none":
            method = None
        else:
            method = self.calibration_method

        # Fit into a local, then swap references — fitting in place on
        # `self._model` would let concurrent should_escalate calls hit a
        # partially-fitted estimator during the refit.
        if method is None:
            model = base
            model.fit(X, y)
        else:
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            cv = max(2, min(self.calibration_cv, n_pos, n_neg))
            try:
                model = CalibratedClassifierCV(base, method=method, cv=cv)
                model.fit(X, y)
            except Exception:
                # Fall back to the uncalibrated base.
                model = base
                model.fit(X, y)

        try:
            fast = _CompiledHGB(model)
        except Exception:
            fast = None
        # Swap order matters: null the fast predictor first so a reader
        # never pairs the NEW model with the OLD compiled trees.
        self._fast_model = None
        self._model = model
        self._fast_model = fast

        if not self._trained:
            self._first_train_at_obs = self._n_obs
        self._trained = True
        self._n_refits += 1

    def fit_info(self) -> dict[str, Any]:
        """Diagnostic dump for the run JSON."""
        with self._buf_lock:
            n_obs = self._n_obs
            n_pos = int(sum(self._buf_y))
            n_neg = len(self._buf_y) - n_pos
        return {
            "router": "lightgbm",
            "trained": self._trained,
            "fast_predictor": self._fast_model is not None,
            "n_observations": n_obs,
            "n_points": self._n_points,
            "refit_every_points": self.refit_every,
            "refit_min_new_obs": self.refit_min_new_obs,
            "first_train_at_obs": self._first_train_at_obs,
            "n_refits": self._n_refits,
            "buffer_pos": n_pos,
            "buffer_neg": n_neg,
            "threshold": self.threshold,
            "bootstrap_threshold": self.bootstrap_threshold,
            "calibration_method": self.calibration_method,
            "lgbm_kwargs": self._lgbm_kwargs,
            "debug_n_called": self._debug_n_called,
            "debug_n_returned_true": self._debug_n_returned_true,
            "debug_n_exceptions": self._debug_n_exceptions,
            "debug_first_exception": self._debug_first_exception,
        }
