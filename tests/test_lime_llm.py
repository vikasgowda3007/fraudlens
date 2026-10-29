import pytest
from sklearn.model_selection import train_test_split

from fraudlens.data.paysim import generate_synthetic
from fraudlens.features.anonymize import deidentified_summary
from fraudlens.features.engineering import build_features
from fraudlens.llm.rationale import (PrivacyError, TemplateClient, build_packet,
                                     generate_rationale, privacy_scan)
from fraudlens.models.detector import FraudDetector
from fraudlens.xai.lime_explainer import LimeExplainer, topk_overlap
from fraudlens.xai.shap_explainer import ShapExplainer


@pytest.fixture(scope="module")
def env():
    df = generate_synthetic(6000, fraud_rate=0.03, seed=21)
    X, y = build_features(df), df.isFraud
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    det = FraudDetector().fit(Xtr, ytr)
    return df, det, Xtr, Xte


def flagged_packet(env):
    df, det, _, Xte = env
    idx = Xte[det.predict(Xte) == 1].index[0]
    X1 = Xte.loc[[idx]]
    contribs = ShapExplainer(det).explain(X1, top_k=4)[0]
    return build_packet(float(det.score(X1)[0]), deidentified_summary(df.loc[idx]), contribs)


def test_lime_returns_topk_and_overlaps_shap(env):
    df, det, Xtr, Xte = env
    row = Xte[det.predict(Xte) == 1].head(1)
    lime = LimeExplainer(det, Xtr).explain(row, top_k=5, num_samples=500)
    shap_top = [c.feature for c in ShapExplainer(det).explain(row, top_k=5)[0]]
    assert len(lime) == 5
    assert topk_overlap(shap_top, [f for f, _ in lime]) >= 0.2   # sanity: not disjoint


def test_packet_has_no_identifiers_and_template_rationale(env):
    pkt = flagged_packet(env)
    r = generate_rationale(pkt)
    assert r.source == "template" and "Analyst review" in r.text
    privacy_scan(pkt.to_json())


def test_privacy_scan_blocks_leaks():
    for bad in ["acct C123456789", "mail a@b.com", "num 1234567890123"]:
        with pytest.raises(PrivacyError):
            privacy_scan(bad)


class Bad:
    def __init__(self, out): self.out = out
    def generate(self, s, u):
        if isinstance(self.out, Exception): raise self.out
        return self.out


@pytest.mark.parametrize("out", ["Block the account C123456789 now.", "Please freeze the account immediately.",
                                 "Amount was 999999 total.", "", RuntimeError("down")])
def test_guard_falls_back(env, out):
    r = generate_rationale(flagged_packet(env), Bad(out))
    assert r.source == "template_fallback"


def test_good_llm_output_accepted(env):
    r = generate_rationale(flagged_packet(env), Bad("Flagged because the origin account was emptied late at night."))
    assert r.source == "llm"
