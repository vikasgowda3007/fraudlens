"""Pseudonymization and de-identified summaries (privacy boundary before the LLM).

Design follows refs [12] (separation of concerns: raw PII stays local) and
[13] (foundation models can leak tabular data, so send only minimal aggregates).
"""
from __future__ import annotations

import hashlib
import hmac
import re

import pandas as pd

ID_COLUMNS = ["nameOrig", "nameDest"]
_ID_PATTERN = re.compile(r"\b[CM]\d{6,}\b")


def pseudonymize_id(value: str, secret: bytes) -> str:
    """Keyed, deterministic pseudonym: same account -> same token, not reversible without key."""
    digest = hmac.new(secret, str(value).encode(), hashlib.sha256).hexdigest()[:10]
    return f"{'MERCH' if str(value).startswith('M') else 'ACCT'}-{digest}"


def anonymize_frame(df: pd.DataFrame, secret: bytes) -> pd.DataFrame:
    out = df.copy()
    for col in ID_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(lambda v: pseudonymize_id(v, secret))
    return out


def contains_raw_identifier(text: str) -> bool:
    """True if text still contains a PaySim-style raw account id (guard before LLM calls)."""
    return bool(_ID_PATTERN.search(text))


def _bucket(amount: float) -> str:
    for limit, label in [(1_000, "<1k"), (10_000, "1k-10k"), (100_000, "10k-100k"), (1_000_000, "100k-1M")]:
        if amount < limit:
            return label
    return ">=1M"


def deidentified_summary(row: pd.Series) -> dict:
    """Minimal, coarse, identifier-free facts about one transaction."""
    return {
        "type": str(row["type"]),
        "amount_bucket": _bucket(float(row["amount"])),
        "hour_of_day": int((int(row["step"]) - 1) % 24),
        "origin_account_drained": bool(row["newbalanceOrig"] == 0 and row["oldbalanceOrg"] > 0),
        "destination_is_merchant": str(row["nameDest"]).startswith(("M", "MERCH")),
        "destination_previously_empty": bool(row["oldbalanceDest"] == 0),
    }
