"""Scaled logistic-regression estimator for the router head.

A tiny sklearn-compatible classifier: StandardScaler + LogisticRegression
composed by hand instead of a Pipeline so ``sample_weight`` survives — the
offline trainer fits with IPS weights via ``fit(X, y, sample_weight=w)``,
and sklearn's Pipeline drops a bare ``sample_weight`` kwarg. Used by
``LightGBMRouter`` (backend="logreg") and
``pipeline.trainer.router.offline_lgbm``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ScaledLogisticRegression(BaseEstimator, ClassifierMixin):
    """StandardScaler + LogisticRegression with sample_weight support."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | None = None,
        max_iter: int = 2000,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X)
        self.lr_ = LogisticRegression(
            C=float(self.C),
            class_weight=self.class_weight,
            max_iter=int(self.max_iter),
        )
        self.lr_.fit(self.scaler_.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.lr_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.lr_.predict(self.scaler_.transform(X))
