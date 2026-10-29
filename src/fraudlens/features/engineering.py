"""Feature engineering shared by training and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]

FEATURE_NAMES = [
    "amount", "log_amount", "hour_of_day", "is_night",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "orig_balance_error", "dest_balance_error", "orig_drained", "amount_to_balance",
    "dest_is_merchant", "dest_was_empty",
] + [f"type_{t}" for t in TYPES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature frame (column order == FEATURE_NAMES)."""
    out = pd.DataFrame(index=df.index)
    amt = df["amount"].astype(float)
    out["amount"] = amt
    out["log_amount"] = np.log1p(amt)
    out["hour_of_day"] = (df["step"].astype(int) - 1) % 24
    out["is_night"] = out["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        out[c] = df[c].astype(float)
    # Balance-consistency errors are a strong PaySim signal.
    out["orig_balance_error"] = out["newbalanceOrig"] + amt - out["oldbalanceOrg"]
    out["dest_balance_error"] = out["oldbalanceDest"] + amt - out["newbalanceDest"]
    out["orig_drained"] = ((out["newbalanceOrig"] == 0) & (out["oldbalanceOrg"] > 0)).astype(int)
    out["amount_to_balance"] = amt / (out["oldbalanceOrg"] + 1.0)
    out["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    out["dest_was_empty"] = (out["oldbalanceDest"] == 0).astype(int)
    for t in TYPES:
        out[f"type_{t}"] = (df["type"] == t).astype(int)
    return out[FEATURE_NAMES]
