"""
The actual agent: given one inbound customer tweet, produces
  { intent, intent_confidence, draft_reply, escalate, escalate_reason }

Two modes:
  - REAL mode: calls Claude (model set by ANTHROPIC_MODEL, default
    claude-sonnet-4-6) via the Anthropic API if ANTHROPIC_API_KEY is set.
  - MOCK mode: if no API key is present, falls back to a deterministic
    stand-in (keyword intent + templated draft) so `python src/pipeline.py`
    still runs end-to-end in <15 minutes with zero setup, per the README's
    reproducibility requirement. Every mock output is tagged so it's never
    confused with a real LLM result in the eval report.

Design choices worth noting (see decision_log.md for the full list):
  - One LLM call does classification + drafting + decision together, not
    three separate calls. Cheaper, and forces the "should I escalate" reason
    to be grounded in the same read of the message as the intent, rather
    than a second model re-deriving it from scratch.
  - The model is given: the customer message, the intent taxonomy with
    descriptions, and 3 retrieved historical (customer, agent) pairs. It is
    explicitly told to ground the draft reply in those examples and not to
    invent policy (refund amounts, timelines) that isn't in the retrieved
    examples.
  - Escalation is NOT purely the model's call. escalate_final in decision.py
    ORs the model's own escalate flag with a hard rule-based policy layer,
    because for money/PII-touching intents we don't want reliability to
    depend entirely on the LLM remembering the policy every time.
"""
import os
import json
from retrieval import HistoryIndex
from intents import INTENTS, INTENT_LIST

MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

SYSTEM_PROMPT = f"""You are a customer support triage assistant for the brand AmazonHelp on Twitter.

Classify the customer's message into exactly one of these intents:
{chr(10).join(f'- {name}: {info["description"]}' for name, info in INTENTS.items())}

You will also be given 1-3 examples of how this brand has actually replied to similar past messages.
Ground your drafted reply in those examples: match their tone, length (1-2 sentences, Twitter support style),
and DO NOT invent specific policy details (refund amounts, timelines, promises) that aren't implied by the examples.

Then decide if this should be auto-handled or escalated to a human agent. Escalate if: it involves money leaving
the company (refunds, billing disputes), account/PII access, a repeat or escalated complaint pattern, or you are
not confident about the intent.

Respond with ONLY valid JSON, no other text, in this exact shape:
{{"intent": "<one of the intent names>", "intent_confidence": <0.0-1.0>, "draft_reply": "<reply text>",
  "escalate": <true|false>, "escalate_reason": "<one short sentence>"}}
"""


class LLMAgent:
    def __init__(self, raw_csv="data/raw/tweets.csv"):
        self.history = HistoryIndex(raw_csv)

        # Get Groq API key from environment
        self.api_key = os.environ.get("GROQ_API_KEY")

        self.mode = "real" if self.api_key else "mock"

        if self.mode == "real":
            from groq import Groq

            self.client = Groq(
                api_key=self.api_key
            )

    def process(self, text):
        examples = self.history.retrieve(text, k=3)

        if self.mode == "real":
            result = self._call_llm(text, examples)
        else:
            result = self._mock(text, examples)

        result["mode"] = self.mode
        result["retrieved_examples"] = examples

        return result

    def _call_llm(self, text, examples):
        examples_block = "\n\n".join(
            f"Past customer message: {e['customer_text']}\n"
            f"Past agent reply: {e['agent_reply']}"
            for e in examples
        )

        user_prompt = (
            f"Customer message to handle now:\n{text}\n\n"
            f"Relevant past examples:\n{examples_block}"
        )

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
                max_tokens=400,
                response_format={
                    "type": "json_object"
                }
            )

            raw = resp.choices[0].message.content

            parsed = json.loads(raw)

            if parsed.get("intent") not in INTENT_LIST:
                parsed["intent"] = "other"
                parsed["intent_confidence"] = min(
                    parsed.get("intent_confidence", 0.5),
                    0.5
                )

            return parsed

        except Exception as e:
            # Fail safe:
            # If the LLM fails, escalate to a human.
            return {
                "intent": "other",
                "intent_confidence": 0.0,
                "draft_reply": "",
                "escalate": True,
                "escalate_reason": (
                    f"LLM call/parse failed ({type(e).__name__}), "
                    "defaulting to human escalation."
                ),
            }

    def _mock(self, text, examples):
        """Deterministic stand-in used only when no API key is configured."""

        from baselines import KeywordBaseline

        intent = KeywordBaseline().predict([text])[0]

        best_reply = (
            examples[0]["agent_reply"]
            if examples
            else "Thanks for reaching out, please DM us more details. ^AB"
        )

        default_escalate = INTENTS[intent]["default_escalate"]

        return {
            "intent": intent,
            "intent_confidence": 0.6,
            "draft_reply": f"[MOCK] {best_reply}",
            "escalate": default_escalate,
            "escalate_reason": (
                f"[MOCK] policy default for intent '{intent}'"
                if default_escalate
                else f"[MOCK] resolvable via standard template for '{intent}'"
            ),
        }


if __name__ == "__main__":
    agent = LLMAgent()

    print(
        f"Running in {agent.mode.upper()} mode "
        f"({'GROQ_API_KEY found' if agent.mode == 'real' else 'no GROQ_API_KEY set'})"
    )

    demo = (
        "This is the second time my headphones arrived broken, "
        "order #123456789. I want a refund."
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
