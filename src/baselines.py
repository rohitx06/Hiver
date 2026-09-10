"""
Two intent baselines required by the take-home:

1. TrivialBaseline: always predicts the dominant operational intent
   (`order_status`). This is intentionally dumb and has no access to labels.
2. KeywordBaseline: ordered regex rules that a team could implement quickly.

Both are evaluated on the same final hand-labelled golden set as the Groq agent.
No synthetic labels or golden-set labels are used to train these baselines.
"""
import re
import pandas as pd
from intents import INTENT_LIST


class TrivialBaseline:
    """Always predicts a fixed high-volume intent; no fitting or test-label access."""
    def fit(self, texts=None, labels=None):
        return self

    def predict(self, texts):
        return ["order_status"] * len(texts)


KEYWORD_RULES = [
    ("account_access", [r"\blog ?in\b", r"\bpassword\b", r"\blocked\b", r"\baccount\b.*\baccess\b"]),
    ("billing_dispute", [r"\bcharged twice\b", r"\bdouble charge", r"\bdon'?t recognize", r"\bsubscription\b.*\bbill", r"\bwrong amount\b", r"\boverbilled\b"]),
    ("refund_request", [r"\brefund\b"]),
    ("cancel_order", [r"\bcancel\b"]),
    ("damaged_or_wrong_item", [r"\bbroken\b", r"\bdamaged\b", r"\bwrong item\b", r"\bdefective\b", r"\bsmashed\b"]),
    ("delivery_delay", [r"\blate\b", r"\bdelay", r"\bmissed.*delivery\b", r"\bstill (not|hasn'?t)\b"]),
    ("order_status", [r"\bwhere.*(order|package)\b", r"\btracking\b", r"\bship\b"]),
    ("positive_feedback", [r"\bthank\b", r"\bthanks\b", r"\bappreciate\b", r"\bgreat\b", r"\bawesome\b"]),
    ("general_complaint", [r"\bdisappointed\b", r"\bfrustrat", r"\bunacceptable\b", r"\bterrible\b", r"\bworst\b"]),
]


class KeywordBaseline:
    def fit(self, texts=None, labels=None):
        return self

    def predict(self, texts):
        preds = []
        for text in texts:
            t = str(text).lower()
            pred = "other"
            for intent, patterns in KEYWORD_RULES:
                if any(re.search(pattern, t) for pattern in patterns):
                    pred = intent
                    break
            preds.append(pred)
        return preds


if __name__ == "__main__":
    golden_path = "data/golden/golden_eval.csv"
    golden = pd.read_csv(golden_path)

    if golden["gold_intent"].isna().any() or (golden["gold_intent"].astype(str).str.strip() == "").any():
        raise ValueError("golden_eval.csv contains blank human labels. Complete the hand-labelled golden set first.")

    for name, model in [
        ("Trivial (fixed order_status prior)", TrivialBaseline()),
        ("Keyword rules", KeywordBaseline()),
    ]:
        model.fit()
        preds = model.predict(golden["text"].tolist())
        acc = sum(p == g for p, g in zip(preds, golden["gold_intent"])) / len(golden)
        print(f"{name:35s} accuracy = {acc:.3f}")
