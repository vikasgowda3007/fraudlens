"""Scoring service: ties detector, SHAP, privacy boundary, LLM rationale, phishing signal, audit log."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

import pandas as pd

from fraudlens.data.paysim import generate_synthetic
from fraudlens.features.anonymize import anonymize_frame, deidentified_summary
from fraudlens.features.engineering import build_features
from fraudlens.llm.rationale import LLMClient, build_packet, generate_rationale
from fraudlens.models.detector import FraudDetector
from fraudlens.models.train import temporal_split
from fraudlens.evaluation.metrics import best_f1_threshold
from fraudlens.phishing.classifier import PhishingClassifier, generate_messages
from fraudlens.xai.shap_explainer import ShapExplainer

PHISH_ALERT = 0.7


class FraudService:
    def __init__(self, db_path: str = ":memory:", llm: LLMClient | None = None,
                 secret: bytes | None = None, n_train: int = 20000):
        self.secret = secret or os.environ.get("FRAUDLENS_SECRET", "dev-only-secret").encode()
        self.llm = llm
        df = generate_synthetic(n_train)
        X, y = build_features(df), df.isFraud
        Xtr, Xva, _, ytr, yva, _ = temporal_split(df, X, y)
        self.detector = FraudDetector().fit(Xtr, ytr)
        self.detector.threshold = best_f1_threshold(yva, self.detector.score(Xva))
        self.explainer = ShapExplainer(self.detector)
        msgs = generate_messages()
        self.phishing = PhishingClassifier().fit(msgs.text, msgs.is_phishing)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS alerts(
            id TEXT PRIMARY KEY, ts REAL, score REAL, flagged INTEGER, phishing_score REAL,
            payload TEXT, feedback TEXT)""")

    def score_transaction(self, tx: dict, message: str | None = None) -> dict:
        t0 = time.perf_counter()
        raw = pd.DataFrame([tx])
        X = build_features(raw)
        score = float(self.detector.score(X)[0])
        phish = float(self.phishing.score([message])[0]) if message else None
        # phishing context nudges triage priority only; it never overrides the detector
        flagged = score >= self.detector.threshold
        contribs = self.explainer.explain(X, top_k=4)[0]
        anon = anonymize_frame(raw, self.secret).iloc[0]
        packet = build_packet(score, deidentified_summary(anon), contribs)
        rationale = generate_rationale(packet, self.llm)
        result = {
            "alert_id": uuid.uuid4().hex[:12], "risk_score": round(score, 4), "flagged": bool(flagged),
            "risk_band": packet.risk_band, "phishing_score": None if phish is None else round(phish, 3),
            "phishing_context": bool(phish is not None and phish >= PHISH_ALERT),
            "drivers": [{"feature": c.feature, "shap": round(c.shap, 3), "direction": c.direction} for c in contribs],
            "rationale": rationale.text, "rationale_source": rationale.source,
            "account_pseudonym": anon["nameOrig"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        self.db.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,NULL)",
                        (result["alert_id"], time.time(), score, int(flagged), phish, json.dumps(result)))
        self.db.commit()
        return result

    def list_alerts(self, flagged_only: bool = True, limit: int = 50) -> list[dict]:
        q = "SELECT payload, feedback FROM alerts " + ("WHERE flagged=1 " if flagged_only else "") + "ORDER BY ts DESC LIMIT ?"
        return [{**json.loads(p), "feedback": f} for p, f in self.db.execute(q, (limit,))]

    def add_feedback(self, alert_id: str, verdict: str) -> bool:
        cur = self.db.execute("UPDATE alerts SET feedback=? WHERE id=?", (verdict, alert_id))
        self.db.commit()
        return cur.rowcount == 1
