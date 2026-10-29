"""FastAPI app: POST /score, GET /alerts, POST /feedback, GET / (analyst dashboard)."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from fraudlens.api.service import FraudService

DASHBOARD = Path(__file__).with_name("dashboard.html")


class Transaction(BaseModel):
    step: int = Field(ge=1, le=744)
    type: Literal["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    amount: float = Field(ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    nameDest: str
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    message: str | None = Field(default=None, max_length=2000, description="optional SMS/email text received by the customer")


class Feedback(BaseModel):
    alert_id: str
    verdict: Literal["confirmed_fraud", "false_positive", "needs_review"]


def create_app(service: FraudService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = service or FraudService(db_path=os.environ.get("FRAUDLENS_DB", ":memory:"))
        yield

    app = FastAPI(title="FraudLens", version="0.1.0", lifespan=lifespan)

    @app.post("/score")
    def score(tx: Transaction):
        d = tx.model_dump()
        msg = d.pop("message")
        return app.state.svc.score_transaction(d, msg)

    @app.get("/alerts")
    def alerts(flagged_only: bool = True, limit: int = 50):
        return app.state.svc.list_alerts(flagged_only, min(limit, 200))

    @app.post("/feedback")
    def feedback(fb: Feedback):
        if not app.state.svc.add_feedback(fb.alert_id, fb.verdict):
            raise HTTPException(404, "unknown alert_id")
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD.read_text()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
