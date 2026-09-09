"""
Generates SUGGESTED labels for data/golden/golden_eval_TEMPLATE.csv so you're
correcting/confirming instead of typing 660 cells from scratch. This does NOT
replace hand-labeling -- it's a first pass. You are still the one who decides
each final label; that's what makes the eval set meaningful (see the
"why hand-label" conversation this came out of -- an eval set labeled by the
same kind of model it's evaluating is circular evidence, not real evidence).

Usage:
    python src/pre_label_golden_set.py
    # writes data/golden/golden_eval_SUGGESTED.csv

Workflow:
    1. Run this (uses ANTHROPIC_API_KEY if set, otherwise a keyword-based
       mock -- see llm_agent.py's mock mode for what that means).
    2. Open golden_eval_SUGGESTED.csv. Rows are sorted so low-confidence /
       flagged-for-review rows are at the TOP -- review those most
       carefully, since that's where the model is most likely wrong.
    3. For each row: read the tweet, check the suggested_* columns, edit
       them directly if wrong (or leave as-is if right), and set
       `reviewed` to 1 once you've actually looked at that row.
    4. When every row has reviewed=1, copy the suggested_* columns' final
       values into gold_intent / gold_escalate / gold_escalate_reason,
       drop the suggested_*/confidence/reviewed columns, and save as
       data/golden/golden_eval.csv.

Cost/time note: this makes one LLM call per row (~220 calls). At Sonnet
pricing that's well under $1 and a couple of minutes; each call is a small
classification prompt, not a long generation.
"""
import os
import re
import json
import pandas as pd

from intents import INTENTS, INTENT_LIST

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

LABEL_SYSTEM_PROMPT = f"""You are helping a human build a hand-labeled evaluation set for a customer
support classifier for the brand AmazonHelp on Twitter. For the given customer tweet, suggest:

- intent: exactly one of:
{chr(10).join(f'  - {name}: {info["description"]}' for name, info in INTENTS.items())}
- escalate: true if this realistically needs a human support agent rather than an automated reply
  (money leaving the company, account/PII access, a repeat or clearly escalated complaint, or the
  message is too ambiguous/unclear to safely auto-handle), false otherwise.
- escalate_reason: one short sentence justifying the escalate call.
- confidence: your confidence in the intent label, 0.0-1.0. Use LOW confidence (<0.6) honestly
  when the tweet is ambiguous, off-topic, mixes multiple issues, or doesn't clearly fit one bucket
  -- a human reviewer is going to prioritize checking your low-confidence rows first, so do not
  inflate this.

Respond with ONLY valid JSON: {{"intent": "...", "escalate": true|false, "escalate_reason": "...", "confidence": 0.0}}
"""

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


class PreLabeler:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.mode = "real" if self.api_key else "mock"
        if self.mode == "real":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def label(self, text):
        if self.mode == "real":
            return self._label_real(text)
        return self._label_mock(text)

    def _label_real(self, text):
        try:
            resp = self.client.messages.create(
                model=MODEL, max_tokens=200,
                system=LABEL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Tweet: {text}"}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
            if parsed.get("intent") not in INTENT_LIST:
                parsed["intent"], parsed["confidence"] = "other", 0.3
            return parsed
        except Exception as e:
            return {"intent": "other", "escalate": True,
                    "escalate_reason": f"labeling call failed ({type(e).__name__}), needs manual review",
                    "confidence": 0.0}

    def _label_mock(self, text):
        """Keyword-based stand-in, used only without ANTHROPIC_API_KEY. Confidence
        is deliberately capped lower than the real model's would be, since a
        keyword match is a much weaker signal -- more rows should get flagged
        for review in mock mode, which is the honest behavior here."""
        t = text.lower()
        intent, matched = "other", False
        for name, patterns in KEYWORD_RULES:
            if any(re.search(p, t) for p in patterns):
                intent, matched = name, True
                break
        confidence = 0.55 if matched else 0.2
        escalate = INTENTS[intent]["default_escalate"]
        reason = (f"[mock] keyword match -> policy default for '{intent}'" if matched
                   else "[mock] no keyword match, low-confidence guess, needs review")
        return {"intent": intent, "escalate": escalate, "escalate_reason": reason, "confidence": confidence}


if __name__ == "__main__":
    in_path = "data/golden/golden_eval_TEMPLATE.csv"
    out_path = "data/golden/golden_eval_SUGGESTED.csv"

    template = pd.read_csv(in_path)
    labeler = PreLabeler()
    print(f"Pre-labeling {len(template)} rows in {labeler.mode.upper()} mode "
          f"({'ANTHROPIC_API_KEY found' if labeler.mode == 'real' else 'no ANTHROPIC_API_KEY set -- using keyword fallback, expect more rows flagged for review'})...")

    rows = []
    for i, r in template.iterrows():
        result = labeler.label(r["text"])
        rows.append({
            "tweet_id": r["tweet_id"],
            "text": r["text"],
            "suggested_intent": result["intent"],
            "suggested_escalate": int(bool(result["escalate"])),
            "suggested_escalate_reason": result["escalate_reason"],
            "confidence": round(float(result.get("confidence", 0.5)), 2),
            "reviewed": 0,
            # final columns you'll actually edit -- pre-filled with the suggestion as a starting point
            "gold_intent": result["intent"],
            "gold_escalate": int(bool(result["escalate"])),
            "gold_escalate_reason": result["escalate_reason"],
        })
        if (i + 1) % 25 == 0:
            print(f"  labeled {i+1}/{len(template)}")

    out = pd.DataFrame(rows).sort_values("confidence").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    n_low_conf = (out["confidence"] < 0.6).sum()
    print(f"\nWrote {len(out)} suggested labels to {out_path}")
    print(f"{n_low_conf}/{len(out)} rows flagged as low-confidence (<0.6) -- these are sorted to the TOP, review those first.")
    print(f"\nMode used: {labeler.mode.upper()}."
          + (" Set ANTHROPIC_API_KEY and re-run for better first-pass suggestions before you review."
             if labeler.mode == "mock" else ""))
    print("\nNext: open the file, correct the gold_intent/gold_escalate/gold_escalate_reason columns "
          "as needed, mark reviewed=1 per row, then save the final version (with those 3 columns plus "
          "tweet_id and text) as data/golden/golden_eval.csv")
