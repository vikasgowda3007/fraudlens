# FraudLens
Explainable LLM-assisted detection of fraudulent real-time banking transactions (CSE 543, Group 6).

Pipeline: PaySim-schema data -> features + anonymization -> XGBoost + IsolationForest -> SHAP/LIME -> constrained LLM rationale -> FastAPI/dashboard.

## Status
| Module | Path | State |
|---|---|---|
| Data (PaySim loader + synthetic generator) | `data/paysim.py` | done, tested |
| Feature engineering | `features/engineering.py` | done, tested |
| Anonymization / de-identified summaries | `features/anonymize.py` | done, tested |
| Hybrid detector (XGBoost + IsolationForest) | `models/` | done, tested |
| Evaluation (PR-AUC, F1, FPR, latency) | `evaluation/` | done, tested |
| SHAP explanations | `xai/shap_explainer.py` | done, tested |
| LIME cross-check | `xai/lime_explainer.py` | done, tested |
| Constrained LLM rationale (+ privacy scan, output guard, fallback) | `llm/rationale.py` | done, tested (template + HF local client; HF client not exercised offline) |
| Phishing text track (synthetic corpus, TF-IDF+LR) | `phishing/` | done, tested (synthetic only) |
| FastAPI + dashboard + SQLite feedback | `api/` | done, tested, live-run verified |

## Run
    python3 -m venv .venv && . .venv/bin/activate && pip install -e . pytest
    # macOS: brew install libomp (needed by xgboost)
    pytest
    python -m fraudlens.models.train --n 30000 --out artifacts   # or --csv PaySim.csv

    uvicorn fraudlens.api.app:app --port 8000     # dashboard at /
    python -m fraudlens.models.train --ablation   # full vs. no balance features

Evaluation uses a temporal split by `step`. Ablation shows PR-AUC falls ~0.995 -> ~0.63 without balance features: PaySim fraud is largely a deterministic drain pattern, so treat headline scores with caution.
The phishing track runs on a template-generated corpus (PaySim has no message data); its scores prove plumbing, not real-world accuracy.

Synthetic data is a stand-in for the real PaySim CSV (Kaggle); numbers on it are not comparable to published results.
Research notes: `docs/RESEARCH.md` (written by background research agent).
