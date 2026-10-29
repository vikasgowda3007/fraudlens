"""Constrained LLM analyst rationale.

The detector decides; the LLM only narrates a whitelisted EvidencePacket (refs [12], [13], [20]).
Pipeline: build packet -> privacy scan -> LLM client -> output guard -> template fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Protocol

from fraudlens.features.anonymize import contains_raw_identifier
from fraudlens.xai.shap_explainer import Contribution

SYSTEM_PROMPT = (
    "You are a fraud-analyst assistant. Using ONLY the JSON evidence provided, write 2-3 sentences "
    "explaining why this transaction was flagged. Do not invent facts, names, or numbers that are not "
    "in the evidence. Do not recommend blocking the account; a human analyst decides."
)

FEATURE_PHRASES = {
    "orig_drained": "the origin account was emptied",
    "amount_to_balance": "the amount is a large share of the account balance",
    "orig_balance_error": "the origin balance does not reconcile with the amount",
    "dest_balance_error": "the destination balance does not reconcile with the amount",
    "is_night": "it occurred at night",
    "hour_of_day": "of the time of day",
    "dest_was_empty": "the destination account was previously empty",
    "dest_is_merchant": "the destination is a merchant",
    "log_amount": "of the transaction amount", "amount": "of the transaction amount",
    "newbalanceDest": "of the destination's resulting balance", "oldbalanceDest": "of the destination's prior balance",
    "newbalanceOrig": "of the origin's resulting balance", "oldbalanceOrg": "of the origin's prior balance",
    "type_TRANSFER": "it is a transfer", "type_CASH_OUT": "it is a cash-out",
}
ALLOWED_KEYS = {"risk_score", "risk_band", "transaction", "top_drivers"}


@dataclass
class EvidencePacket:
    risk_score: float
    risk_band: str
    transaction: dict       # output of deidentified_summary()
    top_drivers: list[dict]  # [{feature, direction}] -- no raw values beyond coarse flags

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def risk_band(score: float) -> str:
    return "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"


def build_packet(score: float, summary: dict, contribs: list[Contribution]) -> EvidencePacket:
    drivers = [{"feature": c.feature, "direction": c.direction} for c in contribs]
    pkt = EvidencePacket(round(float(score), 3), risk_band(score), summary, drivers)
    assert set(asdict(pkt)) == ALLOWED_KEYS
    return pkt


class PrivacyError(RuntimeError):
    pass


def privacy_scan(text: str) -> None:
    if contains_raw_identifier(text) or re.search(r"[\w.]+@[\w.]+\.\w+", text) or re.search(r"\b\d{9,}\b", text):
        raise PrivacyError("identifier-like content found in outbound prompt")


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class TemplateClient:
    """Deterministic offline fallback; also the default so the system works with no model."""

    def generate(self, system: str, user: str) -> str:
        ev = json.loads(user)
        tx, up = ev["transaction"], [d for d in ev["top_drivers"] if d["direction"] == "raises risk"]
        reasons = [FEATURE_PHRASES.get(d["feature"], d["feature"].replace("_", " ")) for d in up[:3]]
        why = "; ".join(reasons) if reasons else "the model's combined signals"
        return (f"This {tx['type'].replace('_', ' ').lower()} (amount {tx['amount_bucket']}) was rated "
                f"{ev['risk_band']} risk ({ev['risk_score']}). Main factors: {why}. "
                f"Analyst review is required before any action.")


class HFLocalClient:
    """Local Hugging Face text-generation model (raw data never leaves the machine)."""

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 120):
        from transformers import pipeline  # lazy: heavy optional dependency
        self.pipe = pipeline("text-generation", model=model)
        self.max_new_tokens = max_new_tokens

    def generate(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self.pipe(msgs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]


BANNED = re.compile(r"\b(block|freeze|close|terminate)\b.*\baccount\b", re.I)


def guard_output(text: str, packet: EvidencePacket) -> bool:
    """Reject empty/overlong text, identifiers, autonomous-action advice, or numbers absent from evidence."""
    if not text.strip() or len(text) > 800 or BANNED.search(text) or contains_raw_identifier(text):
        return False
    allowed = packet.to_json() + " 2 3"
    return all(n in allowed for n in re.findall(r"\d+(?:\.\d+)?", text))


@dataclass
class Rationale:
    text: str
    source: str  # "llm" or "template_fallback"


def generate_rationale(packet: EvidencePacket, client: LLMClient | None = None) -> Rationale:
    user = packet.to_json()
    privacy_scan(SYSTEM_PROMPT + user)
    client = client or TemplateClient()
    try:
        text = client.generate(SYSTEM_PROMPT, user).strip()
    except Exception:
        text = ""
    if guard_output(text, packet):
        return Rationale(text, "llm" if not isinstance(client, TemplateClient) else "template")
    return Rationale(TemplateClient().generate(SYSTEM_PROMPT, user), "template_fallback")
