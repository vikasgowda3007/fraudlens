"""LIME tabular cross-check for SHAP (ref [10]). Slow: use for consistency studies, not every alert."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from fraudlens.features.engineering import FEATURE_NAMES
from fraudlens.models.detector import FraudDetector


class LimeExplainer:
    def __init__(self, detector: FraudDetector, background: pd.DataFrame, seed: int = 0):
        self.detector = detector
        self.seed = seed
        self._ex = LimeTabularExplainer(
            background[FEATURE_NAMES].to_numpy(), feature_names=FEATURE_NAMES,
            class_names=["legit", "fraud"], discretize_continuous=True, random_state=seed)

    def explain(self, row: pd.DataFrame, top_k: int = 5, num_samples: int = 1000) -> list[tuple[str, float]]:
        def predict(a: np.ndarray) -> np.ndarray:
            p = self.detector.fraud_proba(pd.DataFrame(a, columns=FEATURE_NAMES))
            return np.column_stack([1 - p, p])
        exp = self._ex.explain_instance(row[FEATURE_NAMES].to_numpy()[0], predict,
                                        num_features=top_k, num_samples=num_samples)
        return exp.as_map()[1] and [(FEATURE_NAMES[i], float(w)) for i, w in exp.as_map()[1]]


def topk_overlap(shap_features: list[str], lime_features: list[str]) -> float:
    """Fraction of SHAP's top-k features also in LIME's top-k (explanation consistency metric)."""
    if not shap_features:
        return 0.0
    return len(set(shap_features) & set(lime_features)) / len(shap_features)
