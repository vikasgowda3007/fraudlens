import pandas as pd
import pytest

from fraudlens.data.paysim import COLUMNS, generate_synthetic, validate_schema
from fraudlens.features.anonymize import (anonymize_frame, contains_raw_identifier,
                                          deidentified_summary, pseudonymize_id)
from fraudlens.features.engineering import FEATURE_NAMES, build_features

SECRET = b"test-secret"


def test_synthetic_schema_and_rate():
    df = generate_synthetic(5000, fraud_rate=0.02, seed=1)
    validate_schema(df)
    assert list(df.columns) == COLUMNS
    assert len(df) == 5000
    assert df["isFraud"].sum() == 100


def test_fraud_only_in_transfer_cashout_and_drains():
    df = generate_synthetic(5000, seed=2)
    fraud = df[df.isFraud == 1]
    assert set(fraud["type"]) <= {"TRANSFER", "CASH_OUT"}
    assert (fraud["newbalanceOrig"] == 0).mean() > 0.6  # hard_frac=0.3 are partial drains


def test_deterministic_seed():
    pd.testing.assert_frame_equal(generate_synthetic(500, seed=3), generate_synthetic(500, seed=3))


def test_bad_schema_and_rate():
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError):
        generate_synthetic(100, fraud_rate=0.9)


def test_features_shape_no_nan():
    df = generate_synthetic(2000, seed=4)
    X = build_features(df)
    assert list(X.columns) == FEATURE_NAMES
    assert not X.isna().any().any()
    assert X.loc[df.isFraud == 1, "orig_drained"].mean() > 0.5


def test_pseudonym_deterministic_keyed_and_opaque():
    a = pseudonymize_id("C123456789", SECRET)
    assert a == pseudonymize_id("C123456789", SECRET)
    assert a != pseudonymize_id("C123456789", b"other")
    assert "123456789" not in a
    assert pseudonymize_id("M123456789", SECRET).startswith("MERCH-")


def test_anonymize_frame_removes_ids():
    df = generate_synthetic(300, seed=5)
    an = anonymize_frame(df, SECRET)
    assert not an["nameOrig"].str.match(r"^C\d+$").any()
    assert not contains_raw_identifier(" ".join(an["nameOrig"]) + " " + " ".join(an["nameDest"]))
    assert contains_raw_identifier("sent to C123456789 today")


def test_deidentified_summary_has_no_ids():
    df = generate_synthetic(200, seed=6)
    s = deidentified_summary(df.iloc[0])
    blob = str(s)
    assert df.iloc[0]["nameOrig"] not in blob and df.iloc[0]["nameDest"] not in blob
    assert not contains_raw_identifier(blob)
