"""Train/evaluate CLI: python -m fraudlens.models.train [--csv PaySim.csv]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from fraudlens.data.paysim import generate_synthetic, load_paysim_csv
from fraudlens.evaluation.metrics import best_f1_threshold, classification_report, measure_latency
from fraudlens.features.engineering import build_features
from fraudlens.models.detector import FraudDetector


LEAKY = ["orig_balance_error", "dest_balance_error", "orig_drained", "amount_to_balance",
         "newbalanceOrig", "oldbalanceOrg", "newbalanceDest", "oldbalanceDest"]


def temporal_split(df, X, y, test_frac=0.3, val_frac=0.2):
    """Split by PaySim ``step`` (time): train < val < test. Avoids look-ahead leakage."""
    order = df["step"].to_numpy().argsort(kind="stable")
    n = len(order)
    n_te, n_va = int(n * test_frac), int(n * (1 - test_frac) * val_frac)
    te, va, tr = order[n - n_te:], order[n - n_te - n_va:n - n_te], order[:n - n_te - n_va]
    return X.iloc[tr], X.iloc[va], X.iloc[te], y.iloc[tr], y.iloc[va], y.iloc[te]


def run(df, seed: int = 42, out_dir: str | Path | None = None, temporal: bool = True,
        drop_features: list[str] | None = None) -> dict:
    X, y = build_features(df), df["isFraud"]
    if temporal:
        Xtr, Xva, Xte, ytr, yva, yte = temporal_split(df, X, y)
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    if drop_features:  # ablation: neutralize columns (constant 0) so the schema stays fixed
        Xtr, Xva, Xte = (a.assign(**{c: 0.0 for c in drop_features}) for a in (Xtr, Xva, Xte))
    det = FraudDetector(seed=seed).fit(Xtr, ytr)
    det.threshold = best_f1_threshold(yva, det.score(Xva))
    report = {
        "blended": classification_report(yte, det.score(Xte), det.threshold),
        "xgboost_only": classification_report(yte, det.fraud_proba(Xte), best_f1_threshold(yva, det.fraud_proba(Xva))),
        "isolation_forest_only": classification_report(yte, det.anomaly_score(Xte), best_f1_threshold(yva, det.anomaly_score(Xva))),
        "latency": measure_latency(det.score, Xte),
        "split": "temporal" if temporal else "stratified", "dropped": drop_features or [],
        "n_train": len(Xtr), "n_test": len(Xte), "test_fraud": int(yte.sum()),
    }
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        det.save(out / "detector.joblib")
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def ablation(df, seed: int = 42) -> dict:
    """Full features vs. without balance-delta/leak-prone features (RESEARCH.md Sec. 6)."""
    return {"full": run(df, seed)["xgboost_only"], "no_balance_features": run(df, seed, drop_features=LEAKY)["xgboost_only"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default="artifacts"); ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    df = load_paysim_csv(a.csv) if a.csv else generate_synthetic(a.n)
    print(json.dumps(ablation(df) if a.ablation else run(df, out_dir=a.out), indent=2))


if __name__ == "__main__":
    main()
