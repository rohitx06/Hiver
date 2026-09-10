"""
LLM-based customer support agent for AmazonHelp.

Produces:
    intent
    intent_confidence
    draft_reply
    escalate
    escalate_reason

Uses:
    - Groq LLM for REAL mode
    - Deterministic keyword baseline for MOCK mode
    - TF-IDF retrieval of historical customer/agent interactions
"""

import os
import json

from dotenv import load_dotenv

from .retrieval import HistoryIndex
from .intents import INTENTS, INTENT_LIST


load_dotenv()

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


SYSTEM_PROMPT = f"""You are a customer support triage assistant for the brand AmazonHelp on Twitter.

Classify the customer's message into exactly one of these intents:

{chr(10).join(
    f'- {name}: {info["description"]}'
    for name, info in INTENTS.items()
)}

You will also be given 1-3 examples of how this brand has actually replied to similar past messages.

Ground your drafted reply in those examples.

Match the historical AmazonHelp tone and keep the reply short:
1-2 sentences, Twitter support style.

DO NOT invent specific policy details such as:
- refund amounts
- delivery timelines
- guarantees
- compensation
- policies

unless they are clearly supported by the retrieved examples.

Then decide whether this should be auto-handled or escalated to a human agent.

Escalate if:
- it involves money leaving the company, such as refunds or billing disputes
- it involves account or PII access
- it is a repeat or escalated complaint
- the intent is unclear
- confidence is low

Return ONLY valid JSON.
"""


class LLMAgent:

    def __init__(
        self,
        raw_csv="data/raw/tweets.csv",
        exclude_texts=None
    ):
        self.history = HistoryIndex(raw_csv)

        self.exclude_texts = set(
            exclude_texts or []
        )

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

    def process(self, text):

        examples = self.history.retrieve(
    text,
    k=2,
    exclude_texts=self.exclude_texts
)

# Prevent huge prompts
        for e in examples:
            e["customer_text"] = str(e["customer_text"])[:500]
            e["agent_reply"] = str(e["agent_reply"])[:500]

        if self.mode == "real":
            result = self._call_llm(
                text,
                examples
            )
        else:
            result = self._mock(
                text,
                examples
            )

        result["mode"] = self.mode

        result["retrieved_examples"] = examples

        return result

    def _call_llm(
        self,
        text,
        examples
    ):

        examples_block = "\n\n".join(
            f"Past customer message: {e['customer_text']}\n"
            f"Past agent reply: {e['agent_reply']}"
            for e in examples
        )

        user_prompt = f"""
Customer:
{text}

Historical examples:
{examples_block}

Return the required JSON. Keep draft_reply under 200 characters.
"""

        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "customer_support_response",
                "strict": True,
                "schema": {
                    "type": "object",

                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": INTENT_LIST
                        },

                        "intent_confidence": {
                            "type": "number"
                        },

                        "draft_reply": {
                            "type": "string"
                        },

                        "escalate": {
                            "type": "boolean"
                        },

                        "escalate_reason": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "intent",
                        "intent_confidence",
                        "draft_reply",
                        "escalate",
                        "escalate_reason"
                    ],

                    "additionalProperties": False
                }
            }
        }

        try:

            resp = self.client.chat.completions.create(

                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.2,

                max_tokens=800,

                response_format=response_schema
            )

            raw = (
                resp.choices[0]
                .message
                .content
                or ""
            ).strip()

            if not raw:
                raise ValueError(
                    "Groq returned an empty response"
                )

            parsed = json.loads(raw)

            # Validate intent
            if parsed["intent"] not in INTENT_LIST:

                parsed["intent"] = "other"

                parsed["intent_confidence"] = min(
                    float(
                        parsed.get(
                            "intent_confidence",
                            0.5
                        )
                    ),
                    0.5
                )

            # Validate confidence
            parsed["intent_confidence"] = max(
                0.0,
                min(
                    1.0,
                    float(
                        parsed["intent_confidence"]
                    )
                )
            )

            return parsed

        except Exception as e:

            print(
                "\n================ LLM ERROR ================\n"
            )

            print(
                f"Error type: {type(e).__name__}"
            )

            print(
                f"Error message: {e}"
            )

            print(
                "\n============================================\n"
            )

            # Fail safe:
            # if the LLM fails, don't pretend it succeeded.
            return {
                "intent": "other",

                "intent_confidence": 0.0,

                "draft_reply": "",

                "escalate": True,

                "escalate_reason": (
                    f"LLM call/parse failed "
                    f"({type(e).__name__}), "
                    "defaulting to human escalation."
                )
            }

    def _mock(
        self,
        text,
        examples
    ):

        from .baselines import KeywordBaseline

        intent = KeywordBaseline().predict(
            [text]
        )[0]

        if examples:

            best_reply = examples[0][
                "agent_reply"
            ]

        else:

            best_reply = (
                "Thanks for reaching out, "
                "please DM us more details. ^AB"
            )

        default_escalate = INTENTS[
            intent
        ]["default_escalate"]

        return {

            "intent": intent,

            "intent_confidence": 0.6,

            "draft_reply": (
                f"[MOCK] {best_reply}"
            ),

            "escalate": default_escalate,

            "escalate_reason": (
                f"[MOCK] policy default for "
                f"intent '{intent}'"
                if default_escalate
                else
                f"[MOCK] resolvable via standard "
                f"template for '{intent}'"
            )
        }


if __name__ == "__main__":

    agent = LLMAgent()

    print(
        f"Running in {agent.mode.upper()} mode "
        f"({'GROQ_API_KEY found' if agent.mode == 'real' else 'no GROQ_API_KEY set'})"
    )

    demo = (
        "This is the second time my headphones "
        "arrived broken, order #123456789. "
        "I want a refund."
    )

    result = agent.process(demo)

    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k != "retrieved_examples"
            },
            indent=2
        )
    )