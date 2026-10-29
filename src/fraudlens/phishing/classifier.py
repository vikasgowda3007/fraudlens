"""Phishing-message track (text-assist demo).

PaySim has no message data, so this track uses a small template-generated corpus and a
TF-IDF + logistic regression baseline. Its score is a *context signal* for the analyst
(account-takeover chain: phishing -> stolen credentials -> drain), not a PaySim label.
Ref [11] reports zero-shot LLM phishing detection with weak validation; a supervised baseline
with held-out evaluation is the more defensible choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PHISH = [
    "URGENT: your {bank} account is locked. Verify your password at {url} within 24 hours",
    "We detected suspicious login. Confirm your PIN and one-time code here {url}",
    "{bank} security alert: click {url} to avoid account suspension and update your credentials",
    "You have won a refund of ${amt}. Enter your card number and CVV at {url} to claim",
    "Action required: unusual activity. Reply with your OTP to restore access to your {bank} account",
    "Final notice! Your online banking will be closed. Sign in now at {url} to verify identity",
]
LEGIT = [
    "Your {bank} statement for last month is ready in the app",
    "Reminder: your appointment is tomorrow at {hr} pm. Reply C to cancel",
    "Thanks for your payment of ${amt}. Your receipt number is in the mobile app",
    "Hi, are we still meeting for lunch at {hr}? Let me know",
    "{bank} will never ask for your password. Visit a branch if you need help with your account",
    "Your order has shipped and should arrive by Friday. Track it in the {bank} rewards app",
]
BANKS = ["Chase", "Wells Fargo", "Citi", "BankOne", "Capital Trust"]
URLS = ["http://secure-login.example-verify.com", "http://bit.ly/3xYz", "http://acct-update.example.net/v"]


def generate_messages(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(i % 2 == 0)
        tpl = rng.choice(PHISH if label else LEGIT)
        text = tpl.format(bank=rng.choice(BANKS), url=rng.choice(URLS), amt=int(rng.integers(5, 900)), hr=int(rng.integers(1, 12)))
        if rng.random() < 0.15:  # label noise-free ambiguity: shared vocabulary
            text += " Please check your account."
        rows.append((text, label))
    return pd.DataFrame(rows, columns=["text", "is_phishing"]).sample(frac=1, random_state=seed).reset_index(drop=True)


class PhishingClassifier:
    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2)),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def fit(self, texts, labels) -> "PhishingClassifier":
        self.pipe.fit(texts, labels)
        return self

    def score(self, texts) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))[:, 1]
