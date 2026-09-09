# Decision Log

Plain list of non-obvious decisions and why, roughly in the order they came up.

1. **Picked AmazonHelp over other brands** — high volume in the real dataset and a very
   recognizable, consistent reply pattern (DM redirect + agent initials sign-off), which makes
   "grounded vs. not grounded" easy to judge, including for a human reviewer skimming the report.

2. **Generated synthetic data instead of skipping the exercise** — the sandbox can't reach
   kaggle.com/huggingface.co. Rather than hand-waving "assume I have the data," I built a generator
   that matches the real schema exactly and documented the 3-line swap to real data, so the
   pipeline is provably correct even if the numbers aren't production-representative.

3. **Kept the mock/real LLM split explicit everywhere (not silently degraded)** — every output row,
   log line, and report table calls out `mode: mock` vs `real`. A system that silently falls back to
   a worse mode and doesn't tell you is more dangerous than one that's honest about it.

4. **One combined LLM call for intent + draft + escalate, not three** — cheaper, and keeps the
   escalation reasoning grounded in the same read of the message as the classification, instead of
   a second call re-deriving "why" from a bare intent label.

5. **Escalation is policy-OR-model, not model-only** — `decision.py` applies a hard rule (e.g.
   refund/billing/account intents always escalate) regardless of what the model says, specifically
   so the safety-critical cases don't depend on the LLM remembering the rule every single call.

6. **Chose recall over precision for escalation** — an unnecessary escalation costs a few minutes
   of a human's time; a missed one risks an AI auto-answering a refund or account-security issue.
   The eval harness reports both, but the report explicitly says which failure mode is worse.

7. **Stratified the golden set by intent (20 per intent) instead of random sampling** — raw customer
   traffic is dominated by order_status/delivery_delay; random sampling would leave too few examples
   of account_access/billing_dispute to say anything meaningful about escalation behavior on exactly
   the intents where it matters most.

8. **Golden escalation labels come from a written policy function, not vibes** — `build_golden_set.py`
   encodes the escalation policy explicitly (e.g. "second time" → escalate even if the base intent
   normally wouldn't), so the label is reproducible and auditable, not just "I felt like it should
   escalate." Flagged in the report as a limitation since I wrote both the policy and the eval.

9. **TF-IDF over embeddings for retrieval** — deliberately the cheaper choice for a v1 on a
   single-brand corpus; documented as the first thing to upgrade, not presented as a final answer.

10. **LLM-as-judge scores 4 separate axes (grounding/correctness/tone/actionability) instead of one
    overall score** — a single 1-5 "quality" number hides *why* something is bad; a reply that's
    perfectly toned but fabricates a policy detail is a very different failure than one that's rude
    but factually safe, and the fix is different in each case.

11. **Judge-human agreement uses "within 1 point" as well as raw correlation** — Pearson correlation
    is undefined/unstable on a 30-sample, low-variance rubric score; %-within-1-point is a more
    robust and interpretable number for this sample size.

12. **Included a placeholder human-scoring file with big warnings rather than fabricating "real"
    human labels** — faking human agreement numbers would defeat the entire point of that deliverable.
    The repo is built so a real reviewer fills in real scores and the harness "just works," rather
    than requiring code changes.

13. **Auto-handle vs. escalate produces a draft either way** — even escalated messages get a drafted
    reply, because a human reviewing an escalation still benefits from a starting point; the
    distinction is who has to approve it before it's sent, not whether a draft exists.

14. **No auto-send in this system** — output is draft + decision, not action. Whether "auto-handle"
    means "send immediately" is a product/trust decision that shouldn't be baked into the eval
    harness's definition of success.

15. **`--limit` flag on the pipeline runner** — added purely so a reviewer (or CI) can smoke-test the
    whole thing on 10 examples in a few seconds before committing to the full 200-example run, given
    the review said reproducibility in under 15 minutes matters.
