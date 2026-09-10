
"""
Evaluation harness.

Three parts:

1. Intent classification metrics:
   - Accuracy
   - Per-class precision / recall / F1
   - Confusion matrix
   - Comparison against baselines

2. Escalation decision metrics:
   - Precision
   - Recall
   - F1
   - Missed escalations
   - Unnecessary escalations

3. Reply quality:
   - LLM-as-judge on a 1-5 rubric:
       * grounding
       * correctness
       * tone
       * actionability
   - Judge <-> human agreement on a small sample

Judge modes:
    REAL  -> Groq LLM
    MOCK  -> heuristic proxy

The human agreement section intentionally does NOT fabricate
human scores. If the human score columns are blank, it reports
the agreement as pending instead of crashing.
"""

import os
import json
import argparse

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from dotenv import load_dotenv

from .intents import INTENT_LIST


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

JUDGE_MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


# ---------------------------------------------------------
# LLM Judge Prompt
# ---------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """
You are grading a customer support agent's draft reply.

Score the reply from 1-5 on each axis:

- grounding:
  Does it match how the brand actually replies, based on
  the provided historical example, without inventing policy?

- correctness:
  Does it correctly address the customer's actual request
  or intent?

- tone:
  Is it appropriately empathetic and professional for a
  customer support context?

- actionability:
  Does the customer know what to do next?

Respond with ONLY valid JSON:

{
  "grounding": <1-5>,
  "correctness": <1-5>,
  "tone": <1-5>,
  "actionability": <1-5>,
  "notes": "<one short sentence>"
}
"""


# ---------------------------------------------------------
# Intent classification metrics
# ---------------------------------------------------------

def classification_report_df(
    gold,
    pred,
    labels=INTENT_LIST,
):
    """
    Calculate overall accuracy and per-class metrics.
    """

    acc = accuracy_score(
        gold,
        pred,
    )

    p, r, f1, support = (
        precision_recall_fscore_support(
            gold,
            pred,
            labels=labels,
            zero_division=0,
        )
    )

    df = pd.DataFrame(
        {
            "intent": labels,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    )

    return acc, df


# ---------------------------------------------------------
# Escalation metrics
# ---------------------------------------------------------

def escalation_metrics(
    gold_escalate,
    pred_escalate,
):
    """
    Calculate escalation precision, recall and F1.

    Missed escalation:
        Should have escalated but didn't.

    Unnecessary escalation:
        Escalated when it wasn't required.
    """

    gold = [
        bool(x)
        for x in gold_escalate
    ]

    pred = [
        bool(x)
        for x in pred_escalate
    ]

    p, r, f1, _ = (
        precision_recall_fscore_support(
            gold,
            pred,
            average="binary",
            zero_division=0,
        )
    )

    missed = sum(
        g and not pr
        for g, pr in zip(gold, pred)
    )

    unnecessary = sum(
        pr and not g
        for g, pr in zip(gold, pred)
    )

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "missed_escalations": missed,
        "unnecessary_escalations": unnecessary,
        "n": len(gold),
    }


# ---------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------

class Judge:

    def __init__(self):

        self.api_key = os.environ.get(
            "GROQ_API_KEY"
        )

        self.mode = (
            "real"
            if self.api_key
            else "mock"
        )

        if self.mode == "real":

            from groq import Groq

            self.client = Groq(
                api_key=self.api_key
            )

    # -----------------------------------------------------
    # Public score method
    # -----------------------------------------------------

    def score(
        self,
        customer_text,
        draft_reply,
        grounding_example,
    ):

        if self.mode == "real":

            return self._score_real(
                customer_text,
                draft_reply,
                grounding_example,
            )

        return self._score_mock(
            customer_text,
            draft_reply,
            grounding_example,
        )

    # -----------------------------------------------------
    # Real Groq judge
    # -----------------------------------------------------

    def _score_real(
        self,
        customer_text,
        draft_reply,
        grounding_example,
    ):

        prompt = (
            f"Customer message:\n"
            f"{customer_text}\n\n"

            f"Historical example reply for a "
            f"similar case:\n"
            f"{grounding_example}\n\n"

            f"Draft reply to grade:\n"
            f"{draft_reply}"
        )

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=JUDGE_MODEL,

                    messages=[
                        {
                            "role": "system",
                            "content":
                                JUDGE_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],

                    temperature=0.1,

                    # GPT-OSS can use reasoning tokens.
                    # Give it enough room to produce valid JSON.
                    max_tokens=500,

                    response_format={
                        "type": "json_object"
                    },
                )
            )

            raw = (
                response
                .choices[0]
                .message
                .content
                or ""
            ).strip()

            parsed = json.loads(
                raw
            )

            return parsed

        except Exception as e:

            print(
                f"Judge call failed: "
                f"{type(e).__name__}: {e}"
            )

            return {
                "grounding": None,
                "correctness": None,
                "tone": None,
                "actionability": None,
                "notes":
                    f"judge call failed: {e}",
            }

    # -----------------------------------------------------
    # Mock judge
    # -----------------------------------------------------

    def _score_mock(
        self,
        customer_text,
        draft_reply,
        grounding_example,
    ):
        """
        Heuristic proxy judge.

        This is NOT a real quality signal.
        """

        draft = (
            draft_reply
            or ""
        ).lower().replace(
            "[mock] ",
            "",
        )

        example = (
            grounding_example
            or ""
        ).lower()

        draft_words = set(
            draft.split()
        )

        example_words = set(
            example.split()
        )

        overlap = (
            len(
                draft_words
                & example_words
            )
            /
            max(
                1,
                len(example_words),
            )
        )

        grounding = (
            5
            if overlap > 0.5
            else (
                3
                if overlap > 0.2
                else 1
            )
        )

        correctness = (
            4
            if any(
                word in draft
                for word in [
                    "dm",
                    "order",
                    "sorry",
                    "help",
                ]
            )
            else 2
        )

        tone = (
            4
            if any(
                word in draft
                for word in [
                    "sorry",
                    "thank",
                    "appreciate",
                    "help",
                ]
            )
            else 3
        )

        actionability = (
            5
            if "dm" in draft
            else (
                2
                if len(draft) < 5
                else 3
            )
        )

        return {
            "grounding": grounding,
            "correctness": correctness,
            "tone": tone,
            "actionability": actionability,
            "notes":
                "mock heuristic judge, "
                "not a real LLM",
        }


# ---------------------------------------------------------
# Reply quality evaluation
# ---------------------------------------------------------

def run_reply_quality_eval(
    predictions_path="outputs/predictions.csv",
    human_sample_path="data/golden/human_judge_sample.csv",
    out_path="outputs/reply_quality.csv",
):
    """
    Run the LLM judge only on the 30-example human evaluation sample.

    This keeps LLM judging cheap and allows direct comparison
    between human and LLM scores.
    """

    preds = pd.read_csv(predictions_path)
    human_sample = pd.read_csv(human_sample_path)

    # Only evaluate tweets selected for the human sample.
    sample_ids = set(
        human_sample["tweet_id"].astype(str)
    )

    preds["tweet_id"] = (
        preds["tweet_id"].astype(str)
    )

    preds = preds[
        preds["tweet_id"].isin(sample_ids)
    ].copy()

    if len(preds) == 0:
        raise ValueError(
            "No matching tweet IDs found between "
            "predictions.csv and human_judge_sample.csv."
        )

    print(
        f"Running LLM judge on "
        f"{len(preds)} human-evaluation examples."
    )

    from .retrieval import HistoryIndex

    hist = HistoryIndex()

    judge = Judge()

    print(
        f"Judge running in "
        f"{judge.mode.upper()} mode "
        f"using model {JUDGE_MODEL}"
    )

    rows = []

    for i, row in preds.reset_index(drop=True).iterrows():

        top = hist.retrieve(
            row["text"],
            k=1,
            exclude_texts={
                str(row["text"])
            },
        )

        grounding_example = (
            top[0]["agent_reply"]
            if top
            else ""
        )

        scores = judge.score(
            row["text"],
            row["draft_reply"],
            grounding_example,
        )

        scores["tweet_id"] = row["tweet_id"]
        scores["judge_mode"] = judge.mode

        rows.append(scores)

        print(
            f"  judged {i + 1}/{len(preds)}"
        )

    out = pd.DataFrame(rows)

    out.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\nWrote reply quality scores "
        f"to {out_path}"
    )

    numeric_cols = [
        "grounding",
        "correctness",
        "tone",
        "actionability",
    ]

    # Only calculate averages from successful judge calls.
    valid = out.dropna(
        subset=numeric_cols
    )

    print(
        f"\nSuccessful judge calls: "
        f"{len(valid)}/{len(out)}"
    )

    print(
        f"Failed judge calls: "
        f"{len(out) - len(valid)}"
    )

    if len(valid) > 0:

        print(
            "\nAverage judge scores:"
        )

        print(
            valid[numeric_cols].mean()
        )

    else:

        print(
            "\nNo valid judge scores were produced."
        )

    return out


# ---------------------------------------------------------
# Judge <-> Human Agreement
# ---------------------------------------------------------

def judge_human_agreement(
    human_csv="data/golden/human_judge_sample.csv",
    reply_quality_csv="outputs/reply_quality.csv",
):
    """
    Compare LLM judge scores with human scores.

    Human scores must be present in:

        grounding
        correctness
        tone
        actionability

    If those fields are blank, the function returns a
    pending status instead of crashing.
    """

    # -----------------------------------------------------
    # Load files
    # -----------------------------------------------------

    human = pd.read_csv(
        human_csv
    )

    judge_scores = pd.read_csv(
        reply_quality_csv
    )

    # -----------------------------------------------------
    # Check required human columns BEFORE merging
    # -----------------------------------------------------

    required_human_columns = [
        "grounding",
        "correctness",
        "tone",
        "actionability",
    ]

    missing_columns = [
        column
        for column in required_human_columns
        if column not in human.columns
    ]

    if missing_columns:

        return {
            "status":
                "invalid_human_sample",

            "missing_columns":
                missing_columns,

            "message":
                "Human judge sample is missing "
                "required score columns.",
        }

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    merged = human.merge(
        judge_scores,
        on="tweet_id",
        suffixes=(
            "_human",
            "_judge",
        ),
    )

    # -----------------------------------------------------
    # Check merged columns
    # -----------------------------------------------------

    required_merged = [
        "grounding_human",
        "correctness_human",
        "tone_human",
        "actionability_human",
    ]

    missing_merged = [
        column
        for column in required_merged
        if column not in merged.columns
    ]

    if missing_merged:

        return {
            "status":
                "invalid_merge",

            "missing_columns":
                missing_merged,

            "message":
                "Human/judge merge did not produce "
                "the expected score columns.",
        }

    # -----------------------------------------------------
    # Convert scores to numeric
    # -----------------------------------------------------

    for column in required_merged:

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Check for blank human scores
    # -----------------------------------------------------

    if merged[
        required_merged
    ].isna().any().any():

        return {
            "status":
                "pending_human_scores",

            "n_examples":
                len(merged),

            "message":
                "Human scores are incomplete. "
                "Fill grounding, correctness, tone, "
                "and actionability before calculating "
                "judge-human agreement.",
        }

    # -----------------------------------------------------
    # Calculate agreement
    # -----------------------------------------------------

    agreements = {
        "status": "complete",
        "n_examples": len(merged),
    }

    for axis in [
        "grounding",
        "correctness",
        "tone",
        "actionability",
    ]:

        human_scores = merged[
            f"{axis}_human"
        ]

        judge_scores_axis = merged[
            f"{axis}_judge"
        ]

        # Percentage within one point.
        within_1 = (
            abs(
                human_scores
                -
                judge_scores_axis
            )
            <= 1
        ).mean()

        # Pearson correlation.
        if (
            human_scores.notna().sum()
            > 1
            and
            judge_scores_axis.notna().sum()
            > 1
        ):

            corr = (
                human_scores.corr(
                    judge_scores_axis
                )
            )

        else:

            corr = float("nan")

        agreements[axis] = {
            "within_1_point_pct":
                round(
                    float(
                        within_1
                    ),
                    3,
                ),

            "pearson_corr":
                (
                    round(
                        float(corr),
                        3,
                    )
                    if pd.notna(corr)
                    else None
                ),
        }

    return agreements


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help=(
            "Skip the slower "
            "reply-quality judge pass."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load predictions
    # -----------------------------------------------------

    preds = pd.read_csv(
        "outputs/predictions.csv"
    )

    # -----------------------------------------------------
    # Intent classification
    # -----------------------------------------------------

    print(
        "=" * 60
    )

    print(
        "INTENT CLASSIFICATION "
        "(LLM agent, on golden set)"
    )

    print(
        "=" * 60
    )

    acc, report = (
        classification_report_df(
            preds["gold_intent"],
            preds["pred_intent"],
        )
    )

    print(
        f"Accuracy: {acc:.3f}\n"
    )

    print(
        report.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ESCALATION DECISION"
    )

    print(
        "=" * 60
    )

    escalation = escalation_metrics(
        preds["gold_escalate"],
        preds["escalate_final"],
    )

    print(
        json.dumps(
            escalation,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Reply quality
    # -----------------------------------------------------

    if not args.skip_quality:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "REPLY QUALITY "
            "(LLM-as-judge)"
        )

        print(
            "=" * 60
        )

        run_reply_quality_eval()

        # -------------------------------------------------
        # Human agreement
        # -------------------------------------------------

        human_path = (
            "data/golden/"
            "human_judge_sample.csv"
        )

        if os.path.exists(
            human_path
        ):

            print(
                "\n"
                + "=" * 60
            )

            print(
                "JUDGE <-> HUMAN AGREEMENT "
                "(30-example sample)"
            )

            print(
                "=" * 60
            )

            agreement = (
                judge_human_agreement()
            )

            print(
                json.dumps(
                    agreement,
                    indent=2,
                )
            )

        else:

            print(
                f"\n(no {human_path} found yet "
                "-- run "
                "src/build_human_judge_sample.py "
                "first)"
            )
