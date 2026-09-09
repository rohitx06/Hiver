"""
Intent taxonomy for the support agent.

Built by reading ~150 sample customer-initiated tweets to AmazonHelp (see
report.md, "Problem framing" for the sampling note) and clustering them by
hand into the smallest set of buckets that (a) covers >90% of traffic and
(b) maps to a genuinely different resolution path / reply template. Anything
that didn't fit cleanly went into "other".

Each intent also carries a default escalation policy used by src/decision.py
as a prior before the model's own signals (confidence, sentiment, keywords)
adjust it. This encodes: "if a human support lead looked at this intent in
isolation, would they trust an AI to answer it end-to-end?"
"""

INTENTS = {
    "order_status": {
        "description": "Customer asking where an order is / tracking info, no complaint of delay yet.",
        "default_escalate": False,
    },
    "delivery_delay": {
        "description": "Order is confirmed late / missed delivery window, customer wants an update or ETA.",
        "default_escalate": False,
    },
    "damaged_or_wrong_item": {
        "description": "Item arrived broken, defective, or the wrong item was sent.",
        "default_escalate": False,
    },
    "refund_request": {
        "description": "Explicit ask for a refund or dispute over a charge/refund already processed.",
        "default_escalate": True,  # money leaving the company -> human sign-off
    },
    "cancel_order": {
        "description": "Customer wants to cancel an order or subscription before it ships/renews.",
        "default_escalate": False,
    },
    "account_access": {
        "description": "Login issues, locked account, password reset, payment method on file.",
        "default_escalate": True,  # touches auth/PII -> human/secure flow
    },
    "billing_dispute": {
        "description": "Wrong charge, double charge, unexpected charge, subscription billing complaint.",
        "default_escalate": True,  # money + often needs account lookup
    },
    "general_complaint": {
        "description": "Frustration/venting about service quality, no single fixable ask.",
        "default_escalate": False,
    },
    "positive_feedback": {
        "description": "Praise, thanks, no action needed beyond acknowledgement.",
        "default_escalate": False,
    },
    "other": {
        "description": "Doesn't fit the above (spam, unrelated @mention, unclear).",
        "default_escalate": True,  # unknown -> safe default
    },
}

INTENT_LIST = list(INTENTS.keys())
