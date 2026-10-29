"""Evaluation: PR-AUC, precision/recall/F1, false-positive behavior, latency (proposal Sec. Components)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score)


def best_f1_threshold(y_true, scores) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def classification_report(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
    }


def measure_latency(score_fn, X: pd.DataFrame, n: int = 200) -> dict:
    """Single-transaction scoring latency in milliseconds."""
    n = min(n, len(X))
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        score_fn(X.iloc[[i]])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"n": n, "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}
