"""
Samples 30 predictions for a human to hand-score on the same 1-5 rubric as
the LLM judge (grounding, correctness, tone, actionability), so we can
measure judge<->human agreement instead of trusting the judge alone.

The generated score columns are intentionally blank. Before submission,
read every customer message + draft reply and enter your own 1-5 scores,
then run `python src/eval_harness.py`.
"""
import re
import pandas as pd

def blank_human_scores():
    return None, None, None, None


if __name__ == "__main__":
    preds = pd.read_csv("outputs/predictions.csv")
    sample = preds.sample(n=min(30, len(preds)), random_state=11)

    rows = []
    for _, r in sample.iterrows():
        g, c, t, a = blank_human_scores()
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
    print("Human score columns are blank. Hand-score every row before running the agreement analysis.")
