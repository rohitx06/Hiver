# Golden Set Hand-Labeling Guide

The assignment requires 150–250 **hand-labelled** real examples. This repo contains a 220-row
real AmazonHelp template in `data/golden/golden_eval_TEMPLATE.csv`.

## Required workflow

1. Read every tweet. Use the tweet itself and, when needed, the surrounding thread in
   `data/raw/tweets.csv` to understand context.
2. Choose exactly one `gold_intent` from the taxonomy below.
3. Set `gold_escalate` to `1` when a human should take over; otherwise `0`.
4. Write a short, specific `gold_escalate_reason`.
5. Do not copy model suggestions blindly. If `pre_label_golden_set.py` is used, its output is
   only a suggested first pass.
6. After reviewing all 220 rows, save the final five-column file as
   `data/golden/golden_eval.csv`.

## Intent taxonomy

- `order_status`: asking where an order/package is or for tracking information, without a confirmed delay.
- `delivery_delay`: delivery is late/missed and the customer wants an update or ETA.
- `damaged_or_wrong_item`: item is broken, defective, damaged, or the wrong item was received.
- `refund_request`: explicit refund request or a refund already being requested/disputed.
- `cancel_order`: wants to cancel an order/subscription before shipment/renewal.
- `account_access`: login, password reset, locked account, or account/payment-method access.
- `billing_dispute`: unexpected, duplicate, incorrect, or disputed charge/subscription billing.
- `general_complaint`: frustration or dissatisfaction without one clear operational request.
- `positive_feedback`: praise/thanks with no support action required.
- `other`: spam, unrelated message, unclear request, or anything outside the taxonomy.

## Escalation policy

Escalate (`1`) when:
- money/refunds/billing disputes require account or financial review;
- authentication, account access, or PII/security is involved;
- the customer explicitly reports a repeated failure or clearly escalates a complaint;
- the message is too ambiguous to safely auto-handle;
- the intent is `other` and there is no safe standard response.

Do not escalate merely because a customer is mildly unhappy if the issue has a clear, low-risk
resolution path such as ordinary order tracking or cancellation.

The goal is not to make the labels agree with the model. The goal is to create an independent
test set that could expose model mistakes.
