"""SHAP explanations for flagged transactions (refs [17], [9], [18], [19])."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from fraudlens.features.engineering import FEATURE_NAMES
from fraudlens.models.detector import FraudDetector


@dataclass
class Contribution:
    feature: str
    value: float
    shap: float  # log-odds contribution toward fraud (+) or legit (-)

    @property
    def direction(self) -> str:
        return "raises risk" if self.shap > 0 else "lowers risk"


class ShapExplainer:
    def __init__(self, detector: FraudDetector):
        self.detector = detector
        self._explainer = shap.TreeExplainer(detector.clf)

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[list[Contribution]]:
        X = X[FEATURE_NAMES]
        vals = np.asarray(self._explainer.shap_values(X))
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(vals[i]))[:top_k]
            out.append([Contribution(FEATURE_NAMES[j], float(X.iloc[i, j]), float(vals[i, j])) for j in order])
        return out

    def base_value(self) -> float:
        ev = self._explainer.expected_value
        return float(np.ravel(ev)[0])
