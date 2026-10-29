from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fraudlens.phishing.classifier import PhishingClassifier, generate_messages


def test_phishing_baseline_generalizes_to_unseen_templates_text():
    df = generate_messages(600, seed=1)
    tr, te = train_test_split(df, test_size=0.3, stratify=df.is_phishing, random_state=0)
    clf = PhishingClassifier().fit(tr.text, tr.is_phishing)
    assert average_precision_score(te.is_phishing, clf.score(te.text)) > 0.95


def test_obvious_cases():
    df = generate_messages(600, seed=2)
    clf = PhishingClassifier().fit(df.text, df.is_phishing)
    s = clf.score(["URGENT verify your password and OTP at http://x.example.com now",
                   "Lunch tomorrow at noon?"])
    assert s[0] > 0.7 > 0.3 > s[1]
