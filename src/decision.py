"""
Final auto-handle vs. escalate decision. The LLM/mock agent proposes an
escalate flag + reason; this module ORs that with a deterministic policy
layer so escalation for money/PII intents doesn't depend solely on the LLM
remembering the rule on every call.

This two-layer design is deliberate: an LLM-only decision is a single point
of failure for the cases where being wrong is expensive (auto-issuing a
refund reply, auto-answering an account-access request). A rule the team
can read, audit, and change without touching a prompt is worth the
redundancy.
"""
from .intents import INTENTS

LOW_CONFIDENCE_THRESHOLD = 0.55


def final_decision(agent_result):
    intent = agent_result["intent"]
    confidence = agent_result.get("intent_confidence", 0.0)
    model_escalate = bool(agent_result.get("escalate", False))
    model_reason = agent_result.get("escalate_reason", "")

    reasons = []
    escalate = False

    if model_escalate:
        escalate = True
        reasons.append(f"model flagged: {model_reason}")

    if INTENTS.get(intent, {}).get("default_escalate", True) and not model_escalate:
        escalate = True
        reasons.append(f"policy override: '{intent}' is always escalated regardless of model output")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        escalate = True
        reasons.append(f"low intent confidence ({confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD})")

    if not escalate:
        reasons.append("resolvable intent, high confidence, no policy override triggered")

    return {
        "escalate_final": escalate,
        "escalate_final_reason": "; ".join(reasons),
    }
