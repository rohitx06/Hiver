"""
Runs the full agent (+ decision layer) over the golden evaluation set and
saves predictions to outputs/predictions.csv. This is the single script the
README points reviewers at.

Usage:
    python src/pipeline.py                 # uses existing data + golden set
    python src/pipeline.py --rebuild-all    # regenerate synthetic data + golden set first
"""
import argparse
import subprocess
import sys
import pandas as pd
from llm_agent import LLMAgent
from decision import final_decision


def run(golden_path="data/golden/golden_eval.csv", raw_csv="data/raw/tweets.csv",
        out_path="outputs/predictions.csv", limit=None):
    golden = pd.read_csv(golden_path)
    if limit:
        golden = golden.head(limit)

    agent = LLMAgent(raw_csv=raw_csv)
    print(f"Agent running in {agent.mode.upper()} mode.")
    if agent.mode == "mock":
        print("  -> No ANTHROPIC_API_KEY found. Set it to get real LLM classification/drafting.\n"
              "     Results below are from the deterministic mock agent, clearly labeled as such.")

    rows = []
    for i, r in golden.iterrows():
        result = agent.process(r["text"])
        decision = final_decision(result)
        rows.append({
            "tweet_id": r["tweet_id"],
            "text": r["text"],
            "gold_intent": r["gold_intent"],
            "gold_escalate": r["gold_escalate"],
            "gold_escalate_reason": r["gold_escalate_reason"],
            "pred_intent": result["intent"],
            "pred_confidence": result.get("intent_confidence"),
            "draft_reply": result.get("draft_reply", ""),
            "model_escalate": result.get("escalate"),
            "model_escalate_reason": result.get("escalate_reason", ""),
            "escalate_final": decision["escalate_final"],
            "escalate_final_reason": decision["escalate_final_reason"],
            "mode": result["mode"],
        })
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(golden)}")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} predictions to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-all", action="store_true",
                         help="Regenerate synthetic data + golden set before running.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run on the first N golden examples (useful for a quick smoke test).")
    args = parser.parse_args()

    if args.rebuild_all:
        subprocess.run([sys.executable, "src/generate_synthetic_data.py"], check=True)
        subprocess.run([sys.executable, "src/build_golden_set.py"], check=True)

    run(limit=args.limit)
