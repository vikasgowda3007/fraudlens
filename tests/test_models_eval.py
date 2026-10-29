import numpy as np
import pytest

from fraudlens.data.paysim import generate_synthetic
from fraudlens.evaluation.metrics import best_f1_threshold, classification_report
from fraudlens.features.engineering import build_features
from fraudlens.models.detector import FraudDetector
from fraudlens.models.train import run


def test_report_math():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.6, 0.4, 0.9])
    r = classification_report(y, s, 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert 0 < best_f1_threshold(y, s) <= 0.9


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        FraudDetector().fraud_proba(build_features(generate_synthetic(50)))


def test_end_to_end_beats_random_and_roundtrips(tmp_path):
    df = generate_synthetic(8000, fraud_rate=0.02, seed=7)
    rep = run(df, out_dir=tmp_path)
    assert rep["blended"]["pr_auc"] > 0.7          # base rate is 0.02
    assert rep["blended"]["recall"] > 0.6
    assert rep["isolation_forest_only"]["pr_auc"] > 0.02  # beats chance
    assert rep["latency"]["p95_ms"] < 200
    det = FraudDetector.load(tmp_path / "detector.joblib")
    X = build_features(df.head(20))
    assert det.score(X).shape == (20,)
    assert ((det.score(X) >= 0) & (det.score(X) <= 1)).all()


def test_temporal_split_orders_time():
    from fraudlens.models.train import temporal_split
    df = generate_synthetic(3000, seed=9)
    X = build_features(df)
    Xtr, Xva, Xte, ytr, yva, yte = temporal_split(df, X, df.isFraud)
    assert df.loc[Xtr.index, "step"].max() <= df.loc[Xva.index, "step"].min()
    assert df.loc[Xva.index, "step"].max() <= df.loc[Xte.index, "step"].min()
    assert len(Xtr) + len(Xva) + len(Xte) == len(df)


def test_ablation_runs_and_reports_both():
    from fraudlens.models.train import ablation
    r = ablation(generate_synthetic(6000, fraud_rate=0.03, seed=10))
    assert set(r) == {"full", "no_balance_features"}
    assert r["full"]["pr_auc"] >= r["no_balance_features"]["pr_auc"] - 0.05
