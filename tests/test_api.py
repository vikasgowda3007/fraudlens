import pytest
from fastapi.testclient import TestClient

from fraudlens.api.app import create_app
from fraudlens.api.service import FraudService


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(FraudService(n_train=8000))) as c:
        yield c


FRAUD = {"step": 3, "type": "TRANSFER", "amount": 250000, "nameOrig": "C100000001", "oldbalanceOrg": 250000,
         "newbalanceOrig": 0, "nameDest": "C200000002", "oldbalanceDest": 0, "newbalanceDest": 0}
LEGIT = {"step": 300, "type": "PAYMENT", "amount": 80, "nameOrig": "C100000003", "oldbalanceOrg": 5000,
         "newbalanceOrig": 4920, "nameDest": "M900000004", "oldbalanceDest": 0, "newbalanceDest": 0}


def test_health_and_dashboard(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "FraudLens analyst console" in client.get("/").text


def test_fraud_flagged_legit_not_and_no_raw_ids_leak(client):
    f = client.post("/score", json={**FRAUD, "message": "URGENT verify your password and OTP at http://x.example.com"}).json()
    assert f["flagged"] and f["risk_band"] in ("high", "medium") and f["phishing_context"]
    assert f["drivers"] and f["rationale"]
    assert "C100000001" not in str(f) and f["account_pseudonym"].startswith("ACCT-")
    l = client.post("/score", json=LEGIT).json()
    assert not l["flagged"] and l["phishing_score"] is None
    assert f["latency_ms"] < 1000


def test_alerts_and_feedback_flow(client):
    a = client.post("/score", json=FRAUD).json()
    ids = [x["alert_id"] for x in client.get("/alerts").json()]
    assert a["alert_id"] in ids
    assert client.post("/feedback", json={"alert_id": a["alert_id"], "verdict": "confirmed_fraud"}).json() == {"ok": True}
    got = [x for x in client.get("/alerts").json() if x["alert_id"] == a["alert_id"]][0]
    assert got["feedback"] == "confirmed_fraud"
    assert client.post("/feedback", json={"alert_id": "nope", "verdict": "false_positive"}).status_code == 404


def test_validation_rejects_bad_input(client):
    assert client.post("/score", json={**FRAUD, "type": "BOGUS"}).status_code == 422
    assert client.post("/score", json={**FRAUD, "amount": -5}).status_code == 422
