# FraudLens Research Notes

Project: Explainable LLM-Assisted Detection of Fraudulent Real-Time Banking Transactions, incl. Account-Takeover (ATO) and Phishing Fraud (CSE543 Group 6).
Date: 2026-10-29.

Citation key:
- `[Proposal]` = "05 - ... Full Project Proposal.docx.txt" (Sept 10, 2026).
- `[IDR n]` = "03 - In-depth References Study Report" for reference n (reviewer-approved unless noted).
- `[WR8]` = "07 - ... Group Weekly Report 8".
- Everything with a URL was fetched on 2026-10-29. Items marked **UNVERIFIED** could not be confirmed from a primary source and need a local check once data is downloaded.

---

## 1. Project requirements (from Proposal)

**Scope [Proposal]:** prototype that scores transactions from a public/synthetic dataset, explains flagged ones, and turns structured evidence into analyst-facing text.

Hard requirements:
1. Python pipeline: preprocess + anonymize BEFORE scoring.
2. Baseline classifier plus an anomaly detector (scikit-learn named as the baseline tool).
3. SHAP or LIME feature-level explanations for flagged cases.
4. Constrained LLM prompt receives ONLY de-identified summaries + model explanations; raw PII stays local. LLM is "Hugging Face Transformers or a hosted LLM API".
5. Lightweight API/dashboard: near-real-time scoring, explanations, analyst feedback.
6. Evaluation: precision, recall, F1/PR-AUC, false-positive behavior, explanation consistency/usefulness, latency, end-to-end case demos.

Explicit non-goals [Proposal]: autonomous account blocking, production bank integration, real customer data.

Fraud-type framing [Proposal]: ATO and phishing are treated as a connected chain (phishing yields credentials, ATO drains the account). Phishing detection of message text is covered in the literature (refs 11, 20) but **PaySim has no message/phishing data** (IDR 21 reviewer: "do not infer phishing coverage"). Decision needed (see Risks): treat PaySim as the ATO-style transaction track only, and demonstrate phishing via a small, clearly-labeled synthetic or public text sample through the LLM path, or scope it out honestly.

Team context: 8 members; references 1-21 read in depth; Week 8 status says implementation not yet started ([WR8]: "identified the remaining readings before implementation").

---

## 2. Per-reference takeaways relevant to implementation

### Data and privacy
- **[21] PaySim (Lopez-Rojas et al., EMSS 2016)** [IDR 21, source checked: https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf]: multi-agent simulation (MASON) calibrated on a private one-month log from an African mobile-money service; run PS41840 ~23M records; five types CASH-IN, CASH-OUT, TRANSFER, PAYMENT, DEBIT. Paper evaluates simulator realism (SSE vs original), NOT fraud classifiers; fraud agents were out of the paper's scope. Takeaway: the labeled Kaggle PaySim1 release is what we use, and its fraud is simulator logic, not real attacker behavior (see Section 3.1).
- **[12] Kong et al., "Separation of Concerns for Privacy-Preserving LLM Adoption" (FinanSE '26)** [IDR 12]: BPPD architecture = local LoRA-tuned DetectorLLM (Llama-3.2-3B) detects PII risk, Format-Preserving Encryption anonymizes while keeping structure, a proxy mediates calls to external LLMs. Reported 99.9% safety accuracy, 87.3% privacy hiding rate vs 73.6%/14.4% baseline, on 4,170 synthetic samples only (in-distribution; typos/adversarial PII untested). Implementation takeaway: copy the *architecture pattern* (local detector/sanitizer -> gateway -> external LLM), not the fine-tuning. Gate = deterministic allowlist of fields, plus a PII scan of the final prompt.
- **[13] Byun et al., "Risk In Context" (arXiv:2507.17066)** [IDR 13]: foundation-model synthetic tabular data shows competitive utility but higher privacy leakage; only membership inference measured. Takeaway: "synthetic" does not equal "safe"; keep the same de-identification discipline even for PaySim, and do not use an LLM to generate/augment data without a leakage check. Low direct implementation impact.

### Models
- **[16] XGBoost (Chen & Guestrin, KDD '16)** [IDR 16, https://arxiv.org/html/1603.02754]: sparsity-aware split finding, regularized objective; names fraud as a use case. Reviewer caveat: paper benchmarks training speed, not real-time inference latency or imbalanced fraud quality -> we must measure these ourselves.
- **[3] Eswar Prasad et al. 2023** [IDR 3]: DT/SVM/ANN with SMOTE+PCA on Kaggle credit-card data (284,807 rows, 492 frauds, 0.17%); ANN 98.69% accuracy. Takeaway: *accuracy is meaningless at this imbalance*; paper has internal inconsistencies (unreported Naive Bayes, contradictory confusion matrix). Do not copy SMOTE+PCA blindly: PCA destroys feature interpretability, which hurts SHAP/LIME.
- **[4] Nassif et al. anomaly-detection review (IEEE Access 2021)** [IDR 4, IDR 2 Cristian]: 290 studies, supports multi-metric evaluation, no universally best detector. Takeaway: report PR-AUC + F1 + precision/recall at an alert budget, not one number.
- **[19] Narasapuram et al., XAI for suspicious bank transactions (IJMLC 2026)** [IDR 19, https://link.springer.com/article/10.1007/s13042-026-02994-w]: RF/SVM/GB on author-generated synthetic data, SHAP/LIME/Shapley-Lorenz with RFE top-10 features; fraud F1 0.66-0.78 (GB best 0.78). Takeaway: realistic F1 expectations for synthetic data; cross-check SHAP vs LIME agreement (our "explanation consistency" metric); reviewer flags label-derived feature leakage risk.

### XAI
- **[17] Lundberg & Lee, SHAP (NeurIPS 2017)** [IDR 17, https://arxiv.org/html/1705.07874]: unique additive attribution satisfying local accuracy, missingness, consistency; Kernel SHAP and model-specific estimators. Reviewer: assess correlated-feature handling and explanation runtime in our layer. Use TreeExplainer (exact-ish, fast for trees).
- **[10] Ribeiro et al., LIME (KDD '16)** [IDR 10]: local sparse linear surrogate on perturbed samples. Limitation: only sparse linear explanations; stochastic. Use as secondary explainer / consistency check.
- **[9] Adadi & Berrada XAI survey (IEEE Access 2018)** [IDR 9]: taxonomy (global/local, model-agnostic/specific); no agreed explainability metric; predates LLMs. Takeaway: define our own operational explanation metrics (Section 6).
- **[18] Pranggono et al. XAI framework for real-time fraud (2026)** [IDR 18]: conceptual only (no dataset); components = real-time monitoring, ML, XAI, LLM explanations, human-in-the-loop, decision logging, regulatory validation, continuous refinement. Takeaway: use as the *checklist* for architecture (audit log, analyst feedback, human-in-loop), not as evidence of performance. Authors admit no large-scale validation.

### LLM-for-fraud
- **[20] Chan & Chan, LLM-Assisted Authentication and Fraud Detection (arXiv:2601.19684)** [IDR 20]: RAG-grounded scam detector cut scam false positives 17.2% -> 3.5%; semantic authentication accepted 99.5% non-exact legit answers at 0.1% false acceptance; synthetic, preliminary. Takeaway: ground the LLM in retrieved/structured evidence (our SHAP table + fixed policy snippets) rather than free recall; adversarial-edit fragility cited (Chang et al.).
- **[15] Gadzinski & Liuzzi, FTX (Finance Research Letters 2024)** [IDR 15 Sujay]: unguided ChatGPT is conservative/unreliable; improves when given evidence and domain context. Takeaway: LLM is a *narrator over supplied evidence*, never the decision-maker.
- **[11] Patel et al., LLM phishing (IEEE HSI 2024)** [IDR 11]: 15 LLMs, only 15 random 419-scam emails; Proposal itself labels methodology weak. Takeaway: do not cite as evidence of phishing-detection accuracy.
- **[14] FinGPT**, **[5],[6],[7],[8]** LLM surveys/application papers: background only; no design constraint extracted beyond "open-weight financial LLMs exist" (not read in depth for this document).

### ATO background (not deeply extracted)
- **[1] Ghelani, [2] Aziz & Andriansyah**: ATO signals include geospatial impossible-travel and behavioral biometrics [Proposal]. PaySim has no geo/device/session data, so these are out of reach; mention as future work.

---

## 3. Verified stack facts (primary sources)

### 3.1 PaySim
| Fact | Source |
|---|---|
| Repo describes PaySim as mobile-money simulator, first synthetic, later calibrated with aggregated real-service data; 2019 contribution added money-laundering cases | https://github.com/EdgarLopezPhD/PaySim |
| Default config: `nbFraudsters=1000`, `fraudProbability=0.001`, `nbClients=20000`, `nbMerchants=34749`, `nbBanks=5`, `nbSteps=720`, `transferLimit=20,000,000,000` | https://raw.githubusercontent.com/EdgarLopezPhD/PaySim/master/PaySim.properties |
| Fraud injection: each step a Fraudster acts with probability `fraudProbability`; picks a random client as victim; if balance > 0, creates mule account(s), TRANSFERs balance (in chunks of `transferLimit`) to mule, then mule does a fraudulent CASH-OUT of full balance. So fraud = drain-and-cash-out pattern. | https://raw.githubusercontent.com/EdgarLopezPhD/PaySim/master/src/main/java/org/paysim/paysim/actors/Fraudster.java |
| Transaction types: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER | [IDR 21]; secondary mirrors |
| Kaggle PaySim1 columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud; 6,362,620 rows x 11 cols (~268 MB) | Kaggle page not renderable by fetch (https://www.kaggle.com/datasets/ealaxi/paysim1); row/column count from mirror https://baselight.app/u/kaggle/dataset/ealaxi_paysim1 (search result). **Secondary source.** |
| `isFlaggedFraud` = business-rule flag for single transfers over 200,000; `isFraud` = transactions by fraudulent agents | Secondary mirrors only (see web search results: https://github.com/BBQtime/Synthetic-Financial-Datasets-For-Fraud-Detection). **UNVERIFIED against original Kaggle text.** |
| Class imbalance ~0.13% (8,213 fraud rows) and fraud only in TRANSFER and CASH_OUT | Widely reported in community analyses but **UNVERIFIED**: search could not confirm 8,213 or the type restriction. The Fraudster.java logic is consistent with the type restriction. Confirm with `value_counts` on first load. |
| Fraudulent transactions are "annulled", so balance columns after fraud are unreliable; mule `isFraud` flag is never reset | Fraudster.java above; mirror note. |

Implications:
- Balance-delta features (e.g., `oldbalanceOrg - amount - newbalanceOrig`, dest balance zero-before flags) are near-perfect fraud proxies because of the simulator's mechanics. This is the **label-leakage/simulator-artifact risk** also raised in [IDR 19]. Report results both with and without such features.
- `nameOrig`/`nameDest` are synthetic IDs (prefix C = client, M = merchant), not real PII; still treat as pseudonymization targets to exercise the privacy pipeline.
- `step` = 1 hour; 743-744 steps ~ 30 days. Use a **temporal split** (train on early steps, test on later), not random split.
- `isFlaggedFraud` must NOT be a feature (it is a post-hoc rule output); use it only as a rule-baseline comparison.

### 3.2 XGBoost (https://xgboost.readthedocs.io/en/stable/parameter.html)
- `scale_pos_weight`: "Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances)."
- `eval_metric="aucpr"`: area under PR curve, uses continuous interpolation (differs from `map`).
- `tree_method="hist"`: faster histogram-optimized approximate greedy algorithm.
- `max_delta_step`: usually unneeded, may help logistic regression when classes extremely imbalanced.
- Tuning guide (https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html): for imbalanced data, if you care about ranking (AUC) use `scale_pos_weight`; if you need well-calibrated probabilities, do NOT rebalance, use finite `max_delta_step`. The page does not discuss AUCPR guidance. Consequence for us: reweighting distorts probabilities, so thresholds must be chosen on a validation set (PR curve), and any "risk score" shown to analysts should be labeled as a rank score unless calibrated (e.g., isotonic) afterwards.

### 3.3 scikit-learn
- `IsolationForest` (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html): defaults `n_estimators=100`, `max_samples='auto'` (=min(256, n_samples)), `contamination='auto'`; float contamination must be in (0, 0.5]. `score_samples` is the *opposite* of the paper's anomaly score: **lower = more abnormal**. `decision_function` = `score_samples - offset_` (negative = outlier). `predict` returns -1 outlier / +1 inlier. Consequence: negate scores before computing AP/PR-AUC; fraud rate (~0.1%) is below the useful contamination granularity, so use scores + threshold on validation, not `predict`.
- `average_precision_score` (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): AP = sum_n (R_n - R_{n-1}) P_n, **not interpolated**; differs from trapezoidal `auc(recall, precision)` which "can be too optimistic". Therefore XGBoost's `aucpr` (interpolated) and sklearn AP will not match exactly; choose AP as the single reported "PR-AUC" and say so.

### 3.4 SHAP TreeExplainer (https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- Supports XGBoost, LightGBM, CatBoost, PySpark, most sklearn trees.
- Default `model_output="raw"`: for XGBoost binary classifier values are in **log-odds**; `shap_values` returns one array per sample plus scalar `expected_value`. Apply `expit` to base + sum to compare with `predict_proba`.
- `feature_perturbation` default "auto": interventional if `data` given, else tree_path_dependent; `model_output="probability"` only works with interventional (cost grows with background size).
- Consequence: explain in log-odds, present top-k signed contributions; for the LLM prompt, pass the top-k features with direction and magnitude, not raw SHAP arrays.

### 3.5 LIME tabular (https://lime-ml.readthedocs.io/en/latest/lime.html)
- `LimeTabularExplainer(training_data, feature_names, class_names, categorical_features, mode, discretize_continuous)`; `explain_instance(data_row, predict_fn, num_features=10, num_samples=5000)`; `predict_fn` must accept a batch and return class probabilities.
- Continuous features are discretized (quartiles default) -> explanations are in bins. 5000 predictions per explanation: much slower than TreeExplainer, so run LIME only on flagged cases / offline for the consistency study, not on the hot path.

### 3.6 FastAPI
- Request bodies via Pydantic `BaseModel`: automatic parsing, validation, JSON Schema in OpenAPI, interactive docs at `/docs` (https://fastapi.tiangolo.com/tutorial/body/).
- `BackgroundTasks` run after the response is sent; docs recommend Celery-like tools for heavy computation but say BackgroundTasks suffice for small tasks sharing app objects (https://fastapi.tiangolo.com/tutorial/background-tasks/). Consequence: score + SHAP synchronously (fast), generate LLM rationale asynchronously (slow) via a background task or separate endpoint; store result keyed by alert id.

### 3.7 LLM options
- Local Transformers (https://huggingface.co/docs/transformers/main/en/llm_tutorial): `generate()` returns only 20 tokens by default -> always set `max_new_tokens`; default decoding is greedy; `temperature` needs `do_sample=True` (low <0.4 suggested for reasoning-type tasks); chat models need `apply_chat_template` with system/user roles; 4-bit load via bitsandbytes `BitsAndBytesConfig(load_in_4bit=True)` (note: bitsandbytes backend support depends on hardware; on a Mac use a small model, fp16/MPS or a GGUF runtime instead - not verified here). Constrained generation libraries listed: Outlines, SynCode (JSON/grammar).
- Hosted: Hugging Face Inference Providers - OpenAI-compatible endpoint `https://router.huggingface.co/v1`, HF token auth, chat-completions only, provider policies `:fastest`/`:cheapest`/`:preferred`; pay-as-you-go with monthly credits for PRO/Team (https://huggingface.co/docs/inference-providers/index). Any other OpenAI-compatible or Anthropic API could be swapped behind the same interface.
- Because hosted calls leave the machine, they are gated by the sanitizer (Section 4).

### 3.8 PII handling (Presidio, https://presidio.dataprivacystack.org/)
- Analyzer detects PII in **text** via NER + regex + rules + checksums; Anonymizer de-identifies with operators; a separate Presidio Structured module covers structured/semi-structured data. The page does not enumerate operators (replace/redact/hash/mask/encrypt appear in the full docs; not fetched here - verify before relying).
- Official disclaimer: "no guarantee that Presidio will find all sensitive information"; add other protections.
- Consequence for tabular PaySim: PII is *structured IDs*, so deterministic handling beats NER. Use keyed hashing (HMAC with secret kept local) or sequential pseudonyms for `nameOrig`/`nameDest`, bucketing for amount/balances if needed, and use Presidio only as a **last-line text scanner** on the assembled prompt (and on any free-text analyst notes). Hashing/pseudonymization per se is not documented in the pages I fetched; HMAC choice is a design decision, not a cited fact (note: unkeyed hash of a low-entropy ID is reversible by dictionary attack).

---

## 4. Recommended architecture

```
PaySim CSV
  -> ingest/validate
  -> split (temporal)
  -> PII boundary (local only): pseudonymize IDs (HMAC, key in env), drop isFlaggedFraud as feature
  -> feature engineering (no PCA; keep names interpretable)
  -> scorers: XGBoost (supervised, p_fraud) + IsolationForest (unsupervised anomaly score)
  -> alert decision: fused rank score vs validation-chosen threshold
  -> explain: SHAP TreeExplainer (every alert), LIME (sampled/offline consistency check)
  -> evidence builder: whitelisted fields only -> EvidencePacket (JSON)
  -> prompt guard: schema validation + Presidio scan + regex deny-list on final prompt
  -> LLM gateway: local HF model OR hosted API (same interface); fixed system prompt, low temperature, max_new_tokens
  -> output guard: must cite only features from the packet; reject numbers not in packet; fallback to template text
  -> FastAPI: /score, /alerts, /alerts/{id}, /feedback, /health, /metrics
  -> dashboard (static page or small Streamlit/HTMX) + SQLite audit log (decisions, prompts hashed, feedback, latencies)
```

Design decisions:
1. **Detector is the decision-maker; LLM only narrates.** Supported by [IDR 15] (LLM alone unreliable), [IDR 20] (grounding cuts false positives), [Proposal] (human oversight, no auto-blocking).
2. **Two-stage privacy boundary** modeled on [IDR 12]: raw data never leaves the local process; the LLM sees an EvidencePacket containing pseudonymous alert id, transaction type, bucketed amount ratio, top-k SHAP features (name, direction, magnitude, human-readable value band), model scores. No account IDs, no exact balances.
3. **Fallback rationale template** (deterministic, from SHAP) so the system works with the LLM disabled and so the LLM output can be compared against a no-LLM baseline in evaluation.
4. **Phishing track (decision pending):** minimal option = a `text_signal` input (analyst-pasted message) passed through the same sanitizer and LLM with a RAG-style policy snippet [IDR 20]; evaluate on a small clearly-synthetic set. Do not claim phishing detection performance from PaySim.
5. **Latency budget:** target p95 under 100 ms for score + SHAP on CPU (to be measured, not assumed); LLM rationale reported separately as seconds-scale async.

### Module list (proposed package `fraudlens/`)
| Module | Responsibility |
|---|---|
| `data/loader.py` | load PaySim, schema checks, temporal split, class stats |
| `privacy/pseudonymize.py` | HMAC IDs, bucketing, field allowlist |
| `privacy/guard.py` | prompt-level scan (regex + Presidio), deny-list, size limits |
| `features/build.py` | feature engineering, leakage switch (with/without balance-delta features) |
| `models/xgb_model.py` | train (scale_pos_weight, aucpr early stopping), save/load |
| `models/iforest_model.py` | train on (mostly) normal data, score negation, normalization |
| `models/fusion.py` | rank-average / logistic stacking, threshold selection on validation |
| `explain/shap_explainer.py` | TreeExplainer, top-k, log-odds -> readable |
| `explain/lime_explainer.py` | LimeTabularExplainer, agreement metrics |
| `llm/evidence.py` | EvidencePacket (pydantic), prompt template |
| `llm/client.py` | `LocalHFClient`, `HostedClient` behind one interface |
| `llm/validate.py` | output checks, fallback template |
| `api/main.py` | FastAPI app, pydantic schemas, background rationale generation |
| `api/store.py` | SQLite alerts, feedback, audit, latencies |
| `dashboard/` | alert queue, detail view (SHAP bars + rationale), feedback buttons |
| `eval/metrics.py` | AP, PR curve, F1, precision@k, latency, explanation metrics |
| `eval/run_eval.py` | reproducible experiment runner, writes tables/figures |
| `tests/` | privacy-leak tests, API contract tests, metric sanity tests |
| `docs/` | RESEARCH.md (this), threat model, eval report |

Suggested stack: Python 3.11+, pandas, scikit-learn, xgboost, shap, lime, fastapi + uvicorn, pydantic, presidio-analyzer (optional), transformers (local) / `huggingface_hub` or `openai` client (hosted), sqlite3, pytest.

---

## 5. Threat assumptions (for the design spec)

- Honest-but-curious external LLM provider: must never receive identifiers or exact balances.
- Prompt injection via free text (analyst notes, pasted phishing messages): treat as untrusted data; LLM has no tools and cannot change a decision; output is advisory text only [IDR 20 cites adversarial edits degrading LLM scam detection].
- Hallucinated rationale: mitigated by the output guard and by showing SHAP bars next to the text.
- Model-evasion and concept drift: out of scope for experiments, listed as limitation.

---

## 6. Evaluation plan

**Data protocol**
- Temporal split by `step` (e.g., train steps 1-500, val 501-620, test 621-end); stratification not possible with time split, so report fraud counts per split.
- Optionally restrict to TRANSFER/CASH_OUT only as an ablation (if fraud is confined to them; verify).
- Never use `isFlaggedFraud` as a feature; report it as a rule baseline.

**Detection metrics** (rationale: [IDR 3] accuracy misleading; [IDR 4] multi-metric)
- Primary: sklearn `average_precision_score` on test set (= "PR-AUC"); also state XGBoost `aucpr` used for early stopping only.
- Precision, recall, F1 at validation-chosen threshold; precision@k / recall at an alert budget (e.g., top 0.1% / 0.5%).
- False-positive behavior: FP count per 10k transactions and per-type breakdown.
- Baselines: majority/rule (`isFlaggedFraud`), logistic regression, IsolationForest alone, XGBoost alone, fusion. Expect IsolationForest weaker than supervised; [IDR 19] reports fraud F1 0.66-0.78 on its synthetic data as a sanity band (different dataset, not a target).
- Ablation: with vs without balance-delta features (simulator-artifact risk).
- Repeat with several seeds; report mean and spread; bootstrap CI for AP.

**Explanation metrics** ([IDR 9]: no standard metric, so define operationally)
- SHAP additivity check: base + sum(SHAP) -> `expit` equals `predict_proba` (sanity).
- SHAP vs LIME top-k overlap (Jaccard / rank correlation) on a sample of flagged cases; stability across LIME reseeds.
- Fidelity: delete/mask top-k SHAP features and measure score drop vs random-k.
- Explanation latency: SHAP vs LIME per case.

**LLM rationale metrics**
- Faithfulness (automatic): fraction of rationale claims (features, directions, numbers) present in the EvidencePacket; target 100% after output guard.
- Privacy leak test: seeded canary identifiers must never appear in any outbound prompt (unit test + scan of audit log).
- Comparison: LLM rationale vs deterministic template on a small rubric (clarity, actionability, correctness) rated by team members (n≥3 raters, ~30 cases); report agreement. Caveat: small, non-expert raters.
- Local model vs hosted model: latency, cost, rubric score.

**System metrics**
- p50/p95/p99 latency for `/score` (including SHAP), measured with warm model, single worker, N≥1000 requests; separate LLM latency. [IDR 16] warns paper speed does not equal inference latency, so measure ourselves.
- Throughput under modest concurrency.
- Analyst feedback loop: record accept/reject/uncertain; report agreement with ground truth on demo set (feedback is not used to retrain in the prototype unless time permits).

**Demonstrations:** 3-5 end-to-end cases: true positive (drain via TRANSFER then CASH_OUT), false positive, borderline, plus one phishing-text case if that track is kept.

---

## 7. Risks and open questions

1. **Phishing coverage gap.** PaySim has no phishing/message/device/geo data. Options: (a) scope phishing as a text-assist demo with a small synthetic set, (b) drop and revise title (Proposal title was already changed once after feedback). Needs team + TA decision. Do not claim ATO/phishing detection accuracy from PaySim; frame it as "ATO-like drain pattern" at most.
2. **Simulator artifacts and leakage.** Fraud logic is deterministic drain + cash-out; balance features may give near-perfect scores, making comparisons uninformative. Mitigate with ablations and a temporal split; state limitations prominently ([IDR 19] raises the same concern for its own data).
3. **Unverified dataset facts.** Exact Kaggle row/fraud counts (6,362,620 rows is from a mirror; 8,213 fraud and TRANSFER/CASH_OUT-only are NOT confirmed), `isFlaggedFraud` definition and "annulled fraud" behavior are from secondary sources. Check on the real CSV and record in the eval report. Kaggle requires login/terms acceptance for download; confirm license/terms for course use.
4. **Calibration vs imbalance weighting.** `scale_pos_weight` distorts probabilities (XGBoost tuning guide). Decide: weight for ranking and treat output as a score, or calibrate afterwards.
5. **IsolationForest value.** Unsupervised scorer may add little over XGBoost; frame honestly as complementary signal for novel patterns. Also its contamination parameter cap (0.5) and score sign convention are easy bugs.
6. **SHAP/LIME instability and cost.** LIME is stochastic and slow; SHAP interventional mode costs scale with background; correlated features (e.g., oldbalance/newbalance) make attributions split arbitrarily ([IDR 17] reviewer note).
7. **LLM hallucination and prompt injection.** Mitigated, not eliminated, by the guards; faithfulness is measured, not assumed. Evidence from [IDR 15], [IDR 20] is preliminary/synthetic.
8. **Privacy of hosted LLM.** Even de-identified summaries can leak patterns; Presidio gives no guarantee (official disclaimer); real banking data is out of scope [Proposal]. Keep the hosted path optional with a local default.
9. **Compute constraints.** 6.3M rows is fine for XGBoost `hist` on a laptop; local LLM size must fit available memory; bitsandbytes 4-bit may not be available on macOS (not verified). Plan: small instruct model (about 1-3B) or hosted fallback.
10. **Low-quality references.** [11] (15 emails), [1], [2], [18] (conceptual) are weak evidence; cite carefully in the final report.
11. **Presidio operator details and HMAC pseudonymization guidance** were not verified from a primary page in this pass; check the Presidio docs/anonymizer page and a standard reference before the privacy section is finalized.
12. **Refs not read in this pass:** 1, 2 (ATO), 5-8 (LLM surveys), 14 (FinGPT) were only skimmed via the Proposal; 8 per-student reports exist if deeper detail is needed.
13. **Evaluation subjectivity.** "Explanation usefulness" will rest on a small internal rubric; be explicit that it is not a user study.

---

## 8. Suggested build order
1. Data loader + temporal split + class stats (resolves item 3 of Risks).
2. Features + XGBoost + IsolationForest + metrics (AP baseline table, leakage ablation).
3. SHAP layer + additivity test; LIME consistency study.
4. Privacy module + EvidencePacket + canary tests.
5. LLM client (template fallback first, then local/hosted) + output guard.
6. FastAPI + SQLite + dashboard + feedback.
7. Latency benchmark, full evaluation run, case demos, report.
