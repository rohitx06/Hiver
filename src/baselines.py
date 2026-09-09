"""
Two baselines the LLM agent has to beat, per the assignment brief:

  1. TrivialBaseline   - always predicts the majority class from training data.
                          Tests: is the golden set even balanced enough that
                          "just guess the common answer" looks bad?
  2. KeywordBaseline    - simple if/elif keyword rules per intent. This is
                          roughly what a team would ship in an afternoon
                          before reaching for an LLM at all. If the LLM
                          barely beats this, that's a real finding, not a
                          footnote (see report.md).
  3. TfidfLogRegBaseline - "simple ML" baseline: TF-IDF + multinomial
                          logistic regression trained on the (non-golden)
                          synthetic training pool. Represents the
                          "why not just train a small classifier" option.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from intents import INTENT_LIST


class TrivialBaseline:
    def fit(self, texts, labels):
        self.majority = pd.Series(labels).value_counts().idxmax()
        return self

    def predict(self, texts):
        return [self.majority] * len(texts)


KEYWORD_RULES = [
    ("account_access", [r"\blog ?in\b", r"\bpassword\b", r"\blocked\b", r"\baccount\b.*\baccess\b"]),
    ("billing_dispute", [r"\bcharged twice\b", r"\bdouble charge", r"\bcharge[d]? .*don'?t recognize", r"\bsubscription\b.*\bbill", r"\bwrong amount\b"]),
    ("refund_request", [r"\brefund\b"]),
    ("cancel_order", [r"\bcancel\b"]),
    ("damaged_or_wrong_item", [r"\bbroken\b", r"\bdamaged\b", r"\bwrong item\b", r"\bdefective\b", r"\bsmashed\b"]),
    ("delivery_delay", [r"\blate\b", r"\bdelay", r"\bmissed.*delivery\b", r"\bstill not here\b"]),
    ("order_status", [r"\bwhere.*(order|package)\b", r"\btracking\b", r"\bship\b"]),
    ("positive_feedback", [r"\bthank\b", r"\bthanks\b", r"\bappreciate\b", r"\bgreat experience\b", r"\bshoutout\b"]),
    ("general_complaint", [r"\bdisappointed\b", r"\bfrustrat", r"\bgone downhill\b", r"\bunacceptable\b.*wait"]),
]


class KeywordBaseline:
    """Trivial-plus: ordered regex rules, first match wins, else 'other'."""
    def fit(self, texts, labels):
        return self

    def predict(self, texts):
        preds = []
        for t in texts:
            t_low = t.lower()
            matched = "other"
            for intent, patterns in KEYWORD_RULES:
                if any(re.search(p, t_low) for p in patterns):
                    matched = intent
                    break
            preds.append(matched)
        return preds


class TfidfLogRegBaseline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))


def load_training_pool(golden_ids):
    """All labeled synthetic data EXCLUDING whatever ended up in the golden
    set, so baselines are evaluated out-of-sample just like the LLM agent."""
    df = pd.read_csv("data/raw/tweets_with_gold_intent_INTERNAL.csv")
    inbound = df[df["inbound"] == True].copy()
    inbound = inbound[~inbound["tweet_id"].isin(golden_ids)]
    return inbound["text"].tolist(), inbound["_gold_intent"].tolist()


if __name__ == "__main__":
    golden = pd.read_csv("data/golden/golden_eval.csv")
    train_texts, train_labels = load_training_pool(golden["tweet_id"].tolist())

    for name, model in [("Trivial (majority class)", TrivialBaseline()),
                         ("Keyword rules", KeywordBaseline()),
                         ("TF-IDF + LogReg", TfidfLogRegBaseline())]:
        model.fit(train_texts, train_labels)
        preds = model.predict(golden["text"].tolist())
        acc = sum(p == g for p, g in zip(preds, golden["gold_intent"])) / len(golden)
        print(f"{name:30s} accuracy = {acc:.3f}")
