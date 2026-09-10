
"""
Generates SUGGESTED labels for an English-only 220-row golden evaluation set.

The script:
1. Loads the existing golden evaluation template.
2. Removes non-English tweets.
3. Randomly samples exactly 220 English tweets.
4. Uses Groq to generate first-pass suggested labels.
5. Leaves gold_* fields blank for HUMAN verification.
6. Sorts low-confidence rows to the top.

IMPORTANT:
The Groq labels are only suggestions. They do NOT replace hand-labeling.

Usage:
    pip install langdetect
    python src/pre_label_golden_set.py

Output:
    data/golden/golden_eval_SUGGESTED.csv
"""

import os
import re
import json
import pandas as pd

from dotenv import load_dotenv
from langdetect import detect, LangDetectException

from intents import INTENTS, INTENT_LIST


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ---------------------------------------------------------
# Groq labeling prompt
# ---------------------------------------------------------

LABEL_SYSTEM_PROMPT = f"""
You are helping a human build a hand-labeled evaluation set for a
customer support classifier for the brand AmazonHelp on Twitter.

For the given customer tweet, suggest:

- intent: exactly one of:
{chr(10).join(
    f'  - {name}: {info["description"]}'
    for name, info in INTENTS.items()
)}

- escalate: true if this realistically needs a human support agent
  rather than an automated reply.

Escalate when appropriate for:
  - money leaving the company
  - account or PII access
  - repeat or clearly escalated complaints
  - serious unresolved problems
  - ambiguous messages that cannot safely be handled automatically

- escalate_reason: one short sentence explaining the escalation decision.

- confidence: your confidence in the intent label, from 0.0 to 1.0.

Use LOW confidence (<0.6) honestly when:
  - the tweet is ambiguous
  - multiple issues are mentioned
  - the tweet does not clearly fit an intent
  - the message is too short
  - the intent is uncertain

Do not inflate confidence.

Respond with ONLY valid JSON:

{{
    "intent": "...",
    "escalate": true,
    "escalate_reason": "...",
    "confidence": 0.0
}}
"""


# ---------------------------------------------------------
# Keyword fallback rules
# ---------------------------------------------------------

KEYWORD_RULES = [
    (
        "account_access",
        [
            r"\blog ?in\b",
            r"\bpassword\b",
            r"\blocked\b",
            r"\baccount\b.*\baccess\b",
        ],
    ),

    (
        "billing_dispute",
        [
            r"\bcharged twice\b",
            r"\bdouble charge",
            r"\bdon'?t recognize",
            r"\bsubscription\b.*\bbill",
            r"\bwrong amount\b",
            r"\boverbilled\b",
        ],
    ),

    (
        "refund_request",
        [
            r"\brefund\b",
        ],
    ),

    (
        "cancel_order",
        [
            r"\bcancel\b",
        ],
    ),

    (
        "damaged_or_wrong_item",
        [
            r"\bbroken\b",
            r"\bdamaged\b",
            r"\bwrong item\b",
            r"\bdefective\b",
            r"\bsmashed\b",
        ],
    ),

    (
        "delivery_delay",
        [
            r"\blate\b",
            r"\bdelay",
            r"\bmissed.*delivery\b",
            r"\bstill (not|hasn'?t)\b",
        ],
    ),

    (
        "order_status",
        [
            r"\bwhere.*(order|package)\b",
            r"\btracking\b",
            r"\bship\b",
        ],
    ),

    (
        "positive_feedback",
        [
            r"\bthank\b",
            r"\bthanks\b",
            r"\bappreciate\b",
            r"\bgreat\b",
            r"\bawesome\b",
        ],
    ),

    (
        "general_complaint",
        [
            r"\bdisappointed\b",
            r"\bfrustrat",
            r"\bunacceptable\b",
            r"\bterrible\b",
            r"\bworst\b",
        ],
    ),
]


# ---------------------------------------------------------
# English language detection
# ---------------------------------------------------------

def is_english(text):
    """
    Detect whether a tweet is English.

    Short or invalid messages are rejected because language
    detection is unreliable on very short text.
    """

    try:
        if not isinstance(text, str):
            return False

        text = text.strip()

        if len(text) < 10:
            return False

        language = detect(text)

        return language == "en"

    except LangDetectException:
        return False


# ---------------------------------------------------------
# Pre-labeler
# ---------------------------------------------------------

class PreLabeler:

    def __init__(self):

        self.api_key = os.environ.get("GROQ_API_KEY")

        self.mode = "real" if self.api_key else "mock"

        if self.mode == "real":
            from groq import Groq

            self.client = Groq(
                api_key=self.api_key
            )

    # -----------------------------------------------------
    # Public label method
    # -----------------------------------------------------

    def label(self, text):

        if self.mode == "real":
            return self._label_real(text)

        return self._label_mock(text)

    # -----------------------------------------------------
    # Groq labeling
    # -----------------------------------------------------

    def _label_real(self, text):

        try:

            response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": LABEL_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Tweet: {text}",
                },
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={
                "type": "json_object"
            },
            )

            raw = (
                response
                .choices[0]
                .message
                .content
                or ""
            ).strip()

            parsed = json.loads(raw)

            # Validate intent
            if parsed.get("intent") not in INTENT_LIST:

                parsed["intent"] = "other"

                parsed["confidence"] = 0.3

            return parsed

        except Exception as e:

            print("\n--- GROQ ERROR ---")
            print(type(e).__name__)
            print(str(e))
            print("------------------\n")

            return {
                "intent": "other",
                "escalate": True,
                "escalate_reason": (
                    f"labeling call failed "
                    f"({type(e).__name__}), needs manual review"
                ),
                "confidence": 0.0,
            }

    # -----------------------------------------------------
    # Keyword fallback
    # -----------------------------------------------------

    def _label_mock(self, text):

        """
        Keyword-based fallback used only when
        GROQ_API_KEY is not configured.

        This is intentionally lower confidence than
        the real LLM because keyword matching is weaker.
        """

        t = text.lower()

        intent = "other"

        matched = False

        for name, patterns in KEYWORD_RULES:

            if any(
                re.search(pattern, t)
                for pattern in patterns
            ):

                intent = name

                matched = True

                break

        confidence = (
            0.55
            if matched
            else 0.2
        )

        escalate = INTENTS[
            intent
        ]["default_escalate"]

        if matched:

            reason = (
                f"[mock] keyword match -> "
                f"policy default for '{intent}'"
            )

        else:

            reason = (
                "[mock] no keyword match, "
                "low-confidence guess, needs review"
            )

        return {
            "intent": intent,
            "escalate": escalate,
            "escalate_reason": reason,
            "confidence": confidence,
        }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------
    # Input / output paths
    # -----------------------------------------------------

    # Use the larger AmazonHelp dataset instead of the
    # old 220-row golden template.
    in_path = "data/raw/tweets.csv"

    out_path = (
        "data/golden/"
        "golden_eval_SUGGESTED.csv"
    )

    # -----------------------------------------------------
    # Load AmazonHelp source data
    # -----------------------------------------------------

    print(
        f"\nLoading source data from {in_path}..."
    )

    template = pd.read_csv(
        in_path,
        encoding="utf-8"
    )

    original_count = len(template)

    print(
        f"Loaded {original_count:,} rows "
        "from data/raw/tweets.csv"
    )

    # -----------------------------------------------------
    # Keep inbound customer tweets only
    # -----------------------------------------------------

    if "inbound" in template.columns:

        template = template[
            template["inbound"] == True
        ].copy()

        print(
            f"Inbound customer tweets: "
            f"{len(template):,}"
        )

    else:

        print(
            "WARNING: 'inbound' column not found. "
            "Continuing without inbound filtering."
        )

    # -----------------------------------------------------
    # English filtering
    # -----------------------------------------------------

    print(
        "\nFiltering for English-language tweets..."
    )

    template = template[
        template["text"].apply(is_english)
    ].copy()

    english_count = len(template)

    print(
        f"English customer tweets retained: "
        f"{english_count:,}"
    )

    # -----------------------------------------------------
    # Make sure enough rows exist
    # -----------------------------------------------------

    REQUIRED_ROWS = 220
    

    if english_count < REQUIRED_ROWS:

        raise ValueError(
            f"\nOnly {english_count} English customer "
            f"tweets are available, but {REQUIRED_ROWS} "
            "are required."
        )

    # -----------------------------------------------------
    # Randomly select exactly 220 examples
    # -----------------------------------------------------

    template = (
        template
        .sample(
            n=REQUIRED_ROWS,
            random_state=42
        )
        .reset_index(drop=True)
    )

    print(
        f"Selected {len(template)} English customer "
        "tweets for the golden set."
    )

    # -----------------------------------------------------
    # Initialize labeler
    # -----------------------------------------------------

    labeler = PreLabeler()

    if labeler.mode == "real":

        print(
            f"\nPre-labeling {len(template)} rows "
            f"using Groq model: {MODEL}"
        )

    else:

        print(
            "\nWARNING: GROQ_API_KEY was not found."
        )

        print(
            "Using keyword-based MOCK mode."
        )

        print(
            "Set GROQ_API_KEY and rerun before "
            "starting manual labeling."
        )

    # -----------------------------------------------------
    # Generate suggested labels
    # -----------------------------------------------------

    rows = []

    for i, row in template.iterrows():

        text = row["text"]

        result = labeler.label(text)

        rows.append(
            {
                "tweet_id": row["tweet_id"],

                "text": text,

                "suggested_intent":
                    result["intent"],

                "suggested_escalate":
                    int(
                        bool(
                            result["escalate"]
                        )
                    ),

                "suggested_escalate_reason":
                    result["escalate_reason"],

                "confidence":
                    round(
                        float(
                            result.get(
                                "confidence",
                                0.5
                            )
                        ),
                        2
                    ),

                # Human review fields.
                # These MUST remain blank until
                # the human reviewer checks the row.
                "reviewed": 0,

                "gold_intent": "",

                "gold_escalate": "",

                "gold_escalate_reason": "",
            }
        )

        if (i + 1) % 25 == 0:

            print(
                f"  labeled "
                f"{i + 1}/{len(template)}"
            )

    # -----------------------------------------------------
    # Create output dataframe
    # -----------------------------------------------------

    out = pd.DataFrame(rows)

    # -----------------------------------------------------
    # Sort low-confidence examples first
    # -----------------------------------------------------

    out = (
        out
        .sort_values(
            "confidence"
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Save suggested golden set
    # -----------------------------------------------------

    out.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    n_low_conf = (
        out["confidence"] < 0.6
    ).sum()

    print(
        f"\nWrote {len(out)} suggested labels "
        f"to {out_path}"
    )

    print(
        f"{n_low_conf}/{len(out)} rows "
        "flagged as low-confidence (<0.6)."
    )

    print(
        "Low-confidence rows are sorted "
        "to the TOP for manual review."
    )

    print(
        f"\nMode used: {labeler.mode.upper()}"
    )

    if labeler.mode == "mock":

        print(
            "Set GROQ_API_KEY and rerun before "
            "starting manual labeling."
        )

    # -----------------------------------------------------
    # Next step
    # -----------------------------------------------------

    print("\nNext:")

    print(
        "1. Open data/golden/"
        "golden_eval_SUGGESTED.csv"
    )

    print(
        "2. Review every suggested label."
    )

    print(
        "3. Enter the final human-confirmed "
        "values in gold_intent, gold_escalate, "
        "and gold_escalate_reason."
    )

    print(
        "4. Set reviewed=1 for every row "
        "you have actually checked."
    )

    print(
        "5. Save the completed file as:"
    )

    print(
        "   data/golden/golden_eval.csv"
    )