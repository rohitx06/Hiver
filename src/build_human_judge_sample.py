"""
Samples 30 predictions for a human to hand-score on the same 1-5 rubric as
the LLM judge (grounding, correctness, tone, actionability), so we can
measure judge<->human agreement instead of just trusting the judge.

IMPORTANT / please read before submitting:
This script fills the score columns with PLACEHOLDER values (a simple
heuristic, intentionally different from the judge's heuristic so agreement
isn't trivially 100%) purely so eval_harness.py has something to run
against out of the box. These are NOT real human judgments.

Before you submit this assignment, open data/golden/human_judge_sample.csv,
read each customer message + draft_reply yourself, and overwrite the 4
score columns with your own 1-5 scores. Then re-run:
    python src/eval_harness.py
The printed judge<->human agreement numbers will then reflect real human
grading. This is called out explicitly in report.md's "what's misleading
about my headline number" section.
"""
import re
import pandas as pd

def placeholder_human_score(text, draft_reply):
    """Intentionally-different-from-the-judge heuristic, standing in for a
    human rater until you fill in real scores (see module docstring)."""
    d = (draft_reply or "").lower()
    length_ok = 20 < len(d) < 220
    has_dm_ask = "dm" in d
    has_empathy = any(w in d for w in ["sorry", "apolog", "thank", "appreciate"])
    grounding = 4 if has_dm_ask else 2
    correctness = 4 if length_ok and has_dm_ask else 3
    tone = 5 if has_empathy else 3
    actionability = 5 if has_dm_ask else 2
    return grounding, correctness, tone, actionability


if __name__ == "__main__":
    preds = pd.read_csv("outputs/predictions.csv")
    sample = preds.sample(n=min(30, len(preds)), random_state=11)

    rows = []
    for _, r in sample.iterrows():
        g, c, t, a = placeholder_human_score(r["text"], r["draft_reply"])
        rows.append({
            "tweet_id": r["tweet_id"],
            "text": r["text"],
            "draft_reply": r["draft_reply"],
            "grounding": g,
            "correctness": c,
            "tone": t,
            "actionability": a,
        })
    out = pd.DataFrame(rows)
    out.to_csv("data/golden/human_judge_sample.csv", index=False)
    print(f"Wrote {len(out)} rows to data/golden/human_judge_sample.csv")
    print("PLACEHOLDER scores only -- hand-score these yourself before submitting (see file docstring).")
