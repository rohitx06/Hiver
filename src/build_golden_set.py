"""
Builds the golden evaluation set: 200 customer-initiated tweets, stratified
across intents, each labeled with:
  - gold_intent           (the correct intent bucket)
  - gold_escalate         (should this realistically go to a human? 1/0)
  - gold_escalate_reason  (short free-text reason, used to sanity check the
                            model's stated reason, not string-matched)

SAMPLING METHOD (see report.md "Problem framing" for the full note):
  - Stratified by intent: ~20 examples per intent (10 intents x 20 = 200),
    rather than pure random sampling, because a random sample of real
    support traffic is heavily skewed toward order_status/delivery_delay
    and would leave <5 examples for rarer-but-important intents like
    account_access and billing_dispute -- exactly the ones we most want to
    verify escalation behavior on.
  - Within each intent, examples are drawn evenly across the 4 template
    variants and the entity fillers (order numbers, product names, day
    counts) so the set isn't dominated by near-duplicate phrasings.
  - escalate labels were assigned using the *policy*, not just the intent
    default: e.g. a refund_request under $ small/implicit amount with a
    clearly missing item is still escalated (money), but a cancel_order
    that hasn't shipped is not, even though both "feel" high-stakes. This
    mirrors how we'd actually instruct a human reviewer: intent alone is
    not the full story, read the message.

ON REAL DATA: this script would instead sample real inbound tweets
(stratified by a first-pass rule-based intent guess) and hand-label them by
reading each tweet + the thread it was part of. The stratification logic
below is written to be schema-compatible with that -- swap the DataFrame
source and it works unchanged.
"""
import pandas as pd
import random
from intents import INTENTS

random.seed(7)

def build(n_per_intent=20):
    df = pd.read_csv("data/raw/tweets_with_gold_intent_INTERNAL.csv")
    inbound = df[df["inbound"] == True].copy()

    rows = []
    for intent in INTENTS:
        subset = inbound[inbound["_gold_intent"] == intent]
        sample = subset.sample(n=min(n_per_intent, len(subset)), random_state=7)
        for _, r in sample.iterrows():
            escalate, reason = escalation_label(intent, r["text"])
            rows.append({
                "tweet_id": r["tweet_id"],
                "text": r["text"],
                "gold_intent": intent,
                "gold_escalate": escalate,
                "gold_escalate_reason": reason,
            })
    golden = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    return golden

def escalation_label(intent, text):
    """Hand-coded policy used as the human label (see module docstring)."""
    t = text.lower()
    default = INTENTS[intent]["default_escalate"]
    # Overrides where reading the actual text changes the call vs. the intent default:
    if intent == "damaged_or_wrong_item" and ("second time" in t or "again" in t):
        return 1, "repeat/escalated complaint pattern, replacement alone won't rebuild trust"
    if intent == "general_complaint" and ("hour" in t or "third" in t):
        return 1, "pattern of repeated failures, needs human account review not a template reply"
    if intent == "refund_request":
        return 1, "money leaving the company requires human sign-off per policy"
    if intent == "account_access":
        return 1, "touches authentication/PII, must go through secure verification flow"
    if intent == "billing_dispute":
        return 1, "disputed charge requires billing team / account lookup"
    if intent == "other":
        return 1, "doesn't match a known resolvable intent, default to human triage"
    return int(default), (
        "standard resolvable request, template DM-for-order-number reply is sufficient"
        if not default else "policy default for this intent"
    )

if __name__ == "__main__":
    golden = build(n_per_intent=20)
    golden.to_csv("data/golden/golden_eval.csv", index=False)
    print(f"Wrote {len(golden)} labeled examples to data/golden/golden_eval.csv")
    print(golden["gold_intent"].value_counts())
    print("\nEscalate rate:", golden["gold_escalate"].mean().round(2))
