import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fraudlens.data.paysim import generate_synthetic
from fraudlens.features.engineering import FEATURE_NAMES, build_features
from fraudlens.models.detector import FraudDetector
from fraudlens.xai.shap_explainer import ShapExplainer


@pytest.fixture(scope="module")
def setup():
    df = generate_synthetic(6000, fraud_rate=0.03, seed=11)
    X, y = build_features(df), df.isFraud
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    det = FraudDetector().fit(Xtr, ytr)
    return det, Xte


def test_shap_local_accuracy(setup):
    """SHAP values + base value must reconstruct the model's raw margin (additivity)."""
    det, Xte = setup
    ex = ShapExplainer(det)
    sample = Xte.head(10)
    vals = np.asarray(ex._explainer.shap_values(sample[FEATURE_NAMES]))
    margin = det.clf.predict(sample[FEATURE_NAMES], output_margin=True)
    assert np.allclose(vals.sum(axis=1) + ex.base_value(), margin, atol=1e-3)


def test_top_features_for_flagged_are_sensible(setup):
    det, Xte = setup
    ex = ShapExplainer(det)
    flagged = Xte[det.predict(Xte) == 1].head(5)
    assert len(flagged) > 0
    for contribs in ex.explain(flagged, top_k=3):
        assert len(contribs) == 3
        assert contribs[0].shap > 0          # strongest driver pushes toward fraud
        assert contribs[0].direction == "raises risk"
