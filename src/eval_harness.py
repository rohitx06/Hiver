"""
Evaluation harness. Three parts:

1. Intent classification metrics: accuracy, per-class precision/recall/F1,
   confusion matrix, vs. the 3 baselines from baselines.py.
2. Escalation decision metrics: precision/recall/F1 of escalate_final vs.
   gold_escalate (this is the more consequential number: missing an
   escalation is worse than an unnecessary one).
3. Reply quality via LLM-as-judge, on a 1-5 rubric across 4 axes
   (grounding, correctness, tone, actionability), PLUS an agreement check
   against small-sample human scores (data/golden/human_judge_sample.csv)
   so the judge itself isn't taken on faith.

Judge modes mirror llm_agent.py:
  - REAL: calls Claude with a fixed rubric prompt.
  - MOCK: heuristic proxy (word overlap with the retrieved grounding
    example + simple checks), clearly labeled, used only so the harness
    runs without an API key. See report.md for why the MOCK judge numbers
    should not be treated as a real quality signal.
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from intents import INTENT_LIST

JUDGE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

JUDGE_SYSTEM_PROMPT = """You are grading a customer support agent's draft reply. Score it 1-5 on each axis:
- grounding: does it match how the brand actually replies (per the provided historical example), without inventing policy?
- correctness: does it correctly address the customer's actual request/intent?
- tone: is it appropriately empathetic and professional for a support context?
- actionability: does the customer know what to do next?

Respond with ONLY valid JSON: {"grounding": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>, "notes": "<one short sentence>"}
"""


def classification_report_df(gold, pred, labels=INTENT_LIST):
    acc = accuracy_score(gold, pred)
    p, r, f1, support = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"intent": labels, "precision": p, "recall": r, "f1": f1, "support": support})
    return acc, df


def escalation_metrics(gold_escalate, pred_escalate):
    gold = [bool(x) for x in gold_escalate]
    pred = [bool(x) for x in pred_escalate]
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
    missed = sum(g and not pr for g, pr in zip(gold, pred))  # should have escalated, didn't -> worst case
    over = sum(pr and not g for g, pr in zip(gold, pred))    # escalated unnecessarily -> costly but safe
    return {
        "precision": p, "recall": r, "f1": f1,
        "missed_escalations": missed, "unnecessary_escalations": over, "n": len(gold),
    }


class Judge:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.mode = "real" if self.api_key else "mock"
        if self.mode == "real":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def score(self, customer_text, draft_reply, grounding_example):
        if self.mode == "real":
            return self._score_real(customer_text, draft_reply, grounding_example)
        return self._score_mock(customer_text, draft_reply, grounding_example)

    def _score_real(self, customer_text, draft_reply, grounding_example):
        prompt = (f"Customer message: {customer_text}\n"
                  f"Historical example reply for a similar case: {grounding_example}\n"
                  f"Draft reply to grade: {draft_reply}")
        try:
            resp = self.client.messages.create(
                model=JUDGE_MODEL, max_tokens=200,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            return json.loads(raw)
        except Exception as e:
            return {"grounding": None, "correctness": None, "tone": None, "actionability": None,
                    "notes": f"judge call failed: {e}"}

    def _score_mock(self, customer_text, draft_reply, grounding_example):
        """Heuristic proxy judge -- NOT a real quality signal, see report.md."""
        draft = (draft_reply or "").lower().replace("[mock] ", "")
        ex = (grounding_example or "").lower()
        draft_words = set(draft.split())
        ex_words = set(ex.split())
        overlap = len(draft_words & ex_words) / max(1, len(ex_words))
        grounding = 5 if overlap > 0.5 else (3 if overlap > 0.2 else 1)
        correctness = 4 if any(w in draft for w in ["dm", "order", "sorry", "help"]) else 2
        tone = 4 if any(w in draft for w in ["sorry", "thank", "appreciate", "help"]) else 3
        actionability = 5 if "dm" in draft else (2 if len(draft) < 5 else 3)
        return {"grounding": grounding, "correctness": correctness, "tone": tone,
                "actionability": actionability, "notes": "mock heuristic judge, not a real LLM"}


def run_reply_quality_eval(predictions_path="outputs/predictions.csv",
                            out_path="outputs/reply_quality.csv"):
    preds = pd.read_csv(predictions_path)
    from retrieval import HistoryIndex
    hist = HistoryIndex()
    judge = Judge()
    print(f"Judge running in {judge.mode.upper()} mode.")

    rows = []
    for _, r in preds.iterrows():
        top = hist.retrieve(r["text"], k=1)
        grounding_example = top[0]["agent_reply"] if top else ""
        scores = judge.score(r["text"], r["draft_reply"], grounding_example)
        scores["tweet_id"] = r["tweet_id"]
        scores["judge_mode"] = judge.mode
        rows.append(scores)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Wrote reply quality scores to {out_path}")
    numeric_cols = ["grounding", "correctness", "tone", "actionability"]
    print(out[numeric_cols].mean(numeric_only=True))
    return out


def judge_human_agreement(human_csv="data/golden/human_judge_sample.csv",
                           reply_quality_csv="outputs/reply_quality.csv"):
    """Compares judge scores to a small hand-scored sample to check the judge
    isn't systematically off. See data/golden/human_judge_sample.csv for the
    hand scores and how the 30-example sample was chosen."""
    human = pd.read_csv(human_csv)
    judge_scores = pd.read_csv(reply_quality_csv)
    merged = human.merge(judge_scores, on="tweet_id", suffixes=("_human", "_judge"))
    agreements = {}
    for axis in ["grounding", "correctness", "tone", "actionability"]:
        h = merged[f"{axis}_human"]
        j = merged[f"{axis}_judge"]
        within_1 = (abs(h - j) <= 1).mean()
        corr = h.corr(j) if j.notna().sum() > 1 else float("nan")
        agreements[axis] = {"within_1_point_pct": round(float(within_1), 3),
                             "pearson_corr": round(float(corr), 3) if pd.notna(corr) else None}
    return agreements


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-quality", action="store_true", help="skip the (slower) reply-quality judge pass")
    args = parser.parse_args()

    preds = pd.read_csv("outputs/predictions.csv")

    print("=" * 60)
    print("INTENT CLASSIFICATION (LLM agent, on golden set)")
    print("=" * 60)
    acc, df = classification_report_df(preds["gold_intent"], preds["pred_intent"])
    print(f"Accuracy: {acc:.3f}\n")
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("ESCALATION DECISION")
    print("=" * 60)
    esc = escalation_metrics(preds["gold_escalate"], preds["escalate_final"])
    print(json.dumps(esc, indent=2))

    if not args.skip_quality:
        print("\n" + "=" * 60)
        print("REPLY QUALITY (LLM-as-judge)")
        print("=" * 60)
        run_reply_quality_eval()

        human_path = "data/golden/human_judge_sample.csv"
        if os.path.exists(human_path):
            print("\n" + "=" * 60)
            print("JUDGE <-> HUMAN AGREEMENT (30-example sample)")
            print("=" * 60)
            print(json.dumps(judge_human_agreement(), indent=2))
        else:
            print(f"\n(no {human_path} found yet -- run src/build_human_judge_sample.py first)")
