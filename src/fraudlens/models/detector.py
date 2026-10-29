"""Hybrid detector: supervised XGBoost + unsupervised IsolationForest.

XGBoost (ref [16]) gives the primary fraud probability; IsolationForest gives an
anomaly score useful for unseen fraud patterns (ref [4] hybrid taxonomy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from fraudlens.features.engineering import FEATURE_NAMES


@dataclass
class FraudDetector:
    threshold: float = 0.5
    anomaly_weight: float = 0.2
    seed: int = 42
    clf: XGBClassifier | None = field(default=None, repr=False)
    iso: IsolationForest | None = field(default=None, repr=False)
    _iso_range: tuple[float, float] = (0.0, 1.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudDetector":
        pos = max(int(y.sum()), 1)
        neg = max(int(len(y) - y.sum()), 1)
        self.clf = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9,
            scale_pos_weight=neg / pos, eval_metric="aucpr",
            random_state=self.seed, n_jobs=2,
        )
        self.clf.fit(X[FEATURE_NAMES], y)
        self.iso = IsolationForest(n_estimators=200, contamination="auto", random_state=self.seed, n_jobs=2)
        self.iso.fit(X[FEATURE_NAMES])
        raw = -self.iso.score_samples(X[FEATURE_NAMES])
        self._iso_range = (float(raw.min()), float(raw.max()))
        return self

    def _check(self) -> None:
        if self.clf is None or self.iso is None:
            raise RuntimeError("detector not fitted")

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation-forest anomaly score scaled to [0, 1] (higher = more anomalous)."""
        self._check()
        lo, hi = self._iso_range
        raw = -self.iso.score_samples(X[FEATURE_NAMES])
        return np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

    def fraud_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check()
        return self.clf.predict_proba(X[FEATURE_NAMES])[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Blended risk score in [0, 1]."""
        w = self.anomaly_weight
        return (1 - w) * self.fraud_proba(X) + w * self.anomaly_score(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FraudDetector":
        return joblib.load(path)
