"""PaySim data access: real CSV loader and a schema-faithful synthetic generator.

PaySim (Lopez-Rojas et al., EMSS 2016, ref [21]) columns:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
In PaySim, fraud occurs only in TRANSFER and CASH_OUT, and a fraudster
typically drains the victim account (account-takeover pattern).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
TYPE_PROBS = [0.35, 0.34, 0.22, 0.08, 0.01]  # approximate PaySim mix
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]


def load_paysim_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a real PaySim CSV and validate the schema."""
    df = pd.read_csv(path, nrows=nrows)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing PaySim columns: {missing}")


def generate_synthetic(n: int = 20000, fraud_rate: float = 0.01, seed: int = 42,
                       hard_frac: float = 0.3) -> pd.DataFrame:
    """Generate PaySim-schema data with injected account-takeover-style fraud.

    Real PaySim fraud rate is ~0.13%; a higher default keeps small demos trainable.
    ``hard_frac`` makes that share of fraud look benign (partial drain, daytime) and
    makes ~3% of legitimate TRANSFER/CASH_OUT drain the account, so detection is not trivial.
    """
    if not 0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    def legit(m: int) -> pd.DataFrame:
        types = rng.choice(TYPES, size=m, p=TYPE_PROBS)
        amount = np.round(rng.lognormal(mean=9.0, sigma=1.3, size=m), 2)
        old_org = np.round(amount * rng.uniform(1.0, 30.0, m) * (types != "CASH_IN") +
                           rng.lognormal(8, 1.5, m) * (types == "CASH_IN"), 2)
        sign = np.where(np.isin(types, ["CASH_IN"]), 1.0, -1.0)
        new_org = np.round(np.maximum(old_org + sign * amount, 0.0), 2)
        merchant = types == "PAYMENT"
        old_dst = np.where(merchant, 0.0, np.round(rng.lognormal(10, 1.5, m), 2))
        new_dst = np.where(merchant, 0.0, np.round(old_dst + amount * (types != "CASH_IN"), 2))
        dest_prefix = np.where(merchant, "M", "C")
        return pd.DataFrame({
            "step": rng.integers(1, 744, m), "type": types, "amount": amount,
            "nameOrig": [f"C{x}" for x in rng.integers(10**8, 10**9, m)],
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "nameDest": [f"{p}{x}" for p, x in zip(dest_prefix, rng.integers(10**8, 10**9, m))],
            "oldbalanceDest": old_dst, "newbalanceDest": new_dst, "isFraud": 0,
        })

    def fraud(m: int) -> pd.DataFrame:
        types = rng.choice(FRAUD_TYPES, size=m)
        old_org = np.round(rng.lognormal(11.5, 1.2, m), 2)
        amount = old_org.copy()  # account drained
        # ATO fraud skews to off-hours
        step = rng.integers(1, 31, m) * 24 + rng.choice([0, 1, 2, 3, 4, 5, 22, 23], m)
        old_dst = np.where(rng.random(m) < 0.5, 0.0, np.round(rng.lognormal(9, 1.5, m), 2))
        return pd.DataFrame({
            "step": step, "type": types, "amount": amount,
            "nameOrig": [f"C{x}" for x in rng.integers(10**8, 10**9, m)],
            "oldbalanceOrg": old_org, "newbalanceOrig": np.zeros(m),
            "nameDest": [f"C{x}" for x in rng.integers(10**8, 10**9, m)],
            "oldbalanceDest": old_dst,
            "newbalanceDest": np.where(types == "TRANSFER", old_dst, np.round(old_dst + amount, 2)),
            "isFraud": 1,
        })

    lg, fr = legit(n_legit), fraud(n_fraud)
    # hard fraud: partial drain, daytime hours
    h = rng.random(len(fr)) < hard_frac
    frac = rng.uniform(0.5, 0.95, h.sum())
    fr.loc[h, "amount"] = np.round(fr.loc[h, "oldbalanceOrg"] * frac, 2)
    fr.loc[h, "newbalanceOrig"] = np.round(fr.loc[h, "oldbalanceOrg"] - fr.loc[h, "amount"], 2)
    fr.loc[h, "step"] = rng.integers(1, 31, h.sum()) * 24 + rng.integers(8, 20, h.sum())
    # hard legit: genuine full-balance withdrawals/transfers
    cand = lg.index[lg["type"].isin(FRAUD_TYPES)]
    pick = rng.choice(cand, size=max(1, int(0.03 * len(cand))), replace=False)
    lg.loc[pick, "amount"] = lg.loc[pick, "oldbalanceOrg"]
    lg.loc[pick, "newbalanceOrig"] = 0.0
    df = pd.concat([lg, fr], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["step"] = df["step"].clip(1, 744)
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > 200000)).astype(int)
    return df[COLUMNS]
