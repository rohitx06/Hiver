# AI Support Agent for AmazonHelp — Report

## 1. Problem framing

**Brand chosen:** AmazonHelp (one of the highest-volume brands in the Kaggle Customer Support on
Twitter dataset, with a clear DM-for-order-number resolution pattern that's easy to verify grounding against).

**What "good" means here, concretely:**
- **Classification** is good if it's right often enough that a human doesn't have to re-read every
  message before trusting the routing — the cost of a wrong intent is a wrong draft reply, not
  catastrophic, but it burns reviewer trust fast if frequent.
- **Drafting** is good if the reply matches how this brand *actually* talks (short, DM-redirect,
  light apology) and never invents specifics (refund amounts, dates, promises) that aren't backed
  by a real historical example. A grounded-but-boring reply beats a fluent-but-fabricated one.
- **Escalation** is good if it almost never lets a money- or PII-touching case get auto-answered
  (high recall on "should escalate"), even at the cost of some unnecessary escalations (lower
  precision is acceptable; missed escalation is not).

**What I chose not to build:**
- No multi-turn conversation state / dialogue manager — each inbound tweet is scored independently.
  Real threads have back-and-forth; handling that well is a bigger project than a take-home allows.
- No fine-tuning. Everything is prompting + retrieval over a small brand corpus, which is the right
  scope for a single-brand inbox and avoids the cost/complexity of training for a take-home.
- No actual auto-send capability — the system produces a draft + a routing decision. Whether
  "auto-handle" means "send automatically" or "pre-fill for one-click approval" is a product
  decision above this system, and I'm deliberately not assuming it.
- No PII redaction/security review of the account-access flow itself — I escalate those to a human
  rather than trying to design a secure verification flow, which is out of scope here.

## 2. Data note (read this first)

This sandboxed environment cannot reach kaggle.com or huggingface.co (network is restricted to
pypi/github/anthropic domains). `src/generate_synthetic_data.py` generates a **synthetic** dataset
that matches the real Kaggle schema exactly (`tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id`) and mimics AmazonHelp's real reply style (DM redirects,
`^XX` agent sign-offs) from template + entity substitution across 10 intents. Every downstream
script is schema-compatible with the real file — see the docstring in that file for the 3-line swap
to real data. **All numbers below are therefore on synthetic data and should be read as "does the
pipeline work correctly," not "this is production accuracy."** Section 5 makes this explicit.

## 3. Results vs. baselines

Intent classification accuracy on the 200-example golden set:

| Model | Accuracy | Notes |
|---|---|---|
| Trivial (majority class) | **0.100** | 10 balanced intents, so majority-class guessing is exactly floor |
| Keyword rules (if/elif regex) | **0.975** | what a team would ship in an afternoon, no ML |
| TF-IDF + Logistic Regression | **1.000** | trained on the non-golden synthetic pool |
| **LLM agent (mock mode)*** | **0.975** | see note below |

\* This run used the deterministic **mock** agent (no `ANTHROPIC_API_KEY` set in this environment),
which internally *is* the keyword baseline plus a retrieval-grounded draft — so its accuracy number
being identical to the keyword baseline's is not a coincidence, it's the same classifier. **This is
exactly the kind of number a reviewer should distrust at face value** — see Section 5.

Escalation decision (`escalate_final` vs. hand-labeled `gold_escalate`), same 200 examples:

| Metric | Value |
|---|---|
| Precision | 1.00 |
| Recall | 0.91 |
| F1 | 0.95 |
| Missed escalations (worst case) | 8 / 200 |
| Unnecessary escalations | 0 / 200 |

Reply quality (LLM-as-judge, 1-5 scale, mock heuristic judge in this run):

| Axis | Mean score |
|---|---|
| Grounding | 5.00 |
| Correctness | 3.85 |
| Tone | 3.52 |
| Actionability | 4.60 |

Judge <-> human agreement (30-example hand-scored sample, `data/golden/human_judge_sample.csv`):
correctness and actionability had >0.9 agreement-within-1-point in this run; **but the "human"
scores in this repo are a placeholder heuristic, not real human grading** — see the file's docstring
and Section 5. This harness is built so you plug in real scores and it just works.

## 4. Failure analysis (top 5, with real examples)

1. **Keyword/mock classifier misses phrasing outside the regex vocabulary.**
   *Example:* "Why was I billed for a subscription I cancelled months ago?" → predicted `other`,
   gold `billing_dispute`. The rule set matches "charged twice" / "double charge" but not "billed
   for a subscription [I cancelled]". *Hypothesis:* rule-based/mock intent detection generalizes
   poorly to paraphrase; a real LLM call (not the mock) should catch this via the intent
   *descriptions* in the prompt rather than exact keywords — this is the single biggest reason to
   re-run with a real API key before trusting these numbers.

2. **Escalation policy defaults miss "repeat complaint" signals.**
   *Example:* "This is the second time the backpack came defective, order #528133274." →
   `damaged_or_wrong_item` classified correctly, but the mock agent's escalate flag falls back to
   the intent's *default* (not escalated) instead of catching "second time" as an aggravating
   factor. 8/200 golden examples were missed this way. *Hypothesis:* escalation needs a signal
   that looks at language *beyond* the intent label (repetition, frustration markers) — the real
   LLM prompt already asks for this, but the mock fallback (used because no API key was set here)
   doesn't implement that nuance, which is exactly why it's flagged as mock-only.

3. **Retrieval grounding is topically right but not always the best-matching precedent.**
   Cosine similarity on TF-IDF pulls examples with the same *intent* reliably, but similarity
   scores are often low (~0.2-0.3) because order numbers and product nouns dominate token overlap
   more than the actual complaint semantics. *Hypothesis:* an embeddings-based retriever would
   separate "my order is late" from "my order arrived damaged" more cleanly than n-gram TF-IDF,
   which is one of the concrete "next week" items in Section 6.

4. **The judge (mock mode) rewards surface patterns, not real quality.**
   The mock judge scores "actionability" near-perfect (4.6/5) largely because it checks for the
   literal substring "dm" — a reply could say "please DM us" and still be actionable-sounding while
   being wrong for the situation. *Hypothesis:* this is a known weakness of cheap heuristic judges
   and is exactly why Section 3 reports judge<->human agreement rather than trusting the judge in
   isolation — and why the real (non-mock) judge should be re-run before this number is trusted.

5. **Synthetic data is too clean to stress-test the system.**
   Every message maps unambiguously to one intent because it was generated from a template for
   that intent. Real Twitter support threads have sarcasm, multiple issues in one tweet, quoted
   screenshots, and replies-to-replies with missing context. *Hypothesis (untested here):*
   real-data accuracy will be meaningfully lower than 97.5%, and the gap between the keyword
   baseline and a real LLM call will widen, because the LLM's advantage is exactly in handling
   the paraphrase/ambiguity the synthetic set doesn't contain.

## 5. What is misleading about my headline number

**The 97.5% "LLM agent" accuracy is misleading in three ways:**

1. **It's not actually the LLM.** No `ANTHROPIC_API_KEY` was set when these headline numbers were
   produced (sandbox constraint), so the reported 97.5% is the deterministic *mock* agent — which
   is literally the keyword-rule baseline underneath. The real Claude-backed agent has not been
   benchmarked in this report. Anyone reading only the table in Section 3 would reasonably assume
   an LLM achieved 97.5%; it did not. **Run `ANTHROPIC_API_KEY=... python src/pipeline.py` to get
   the real number** — I'd expect it to be different (plausibly lower on trickier phrasing per
   Failure #1, but possibly higher on generalizing beyond the fixed keyword list).

2. **The data is synthetic and template-generated**, so intents are far more separable than real
   Twitter text. A keyword baseline hitting 97.5% and TF-IDF+LogReg hitting 100% is itself evidence
   the eval set is too easy — a well-built classical baseline should not beat an LLM this cleanly
   on a task LLMs are supposed to be good at (handling paraphrase/ambiguity). On real data I'd
   expect the ordering of baselines vs. LLM to look very different.

3. **The golden set's escalation labels were written by the same person who wrote the escalation
   policy** (me, in `build_golden_set.py`), so precision/recall against those labels partly measures
   "did the code correctly implement the policy I wrote" rather than "does the policy match what a
   real support lead would want." A real project needs those written by someone else, ideally an
   actual support team member, reviewing real edge cases.

Net: treat every number in Section 3 as **"the pipeline is wired correctly and the eval harness
works,"** not as **"this is what to expect in production."** That gap is the main thing I'd close
first with more time.

## 6. What I'd do next with one more week

1. **Get real data and a real API key running end-to-end.** Download the actual Kaggle dataset,
   filter to AmazonHelp, re-run `src/build_golden_set.py` against real threads (hand-labeling for
   real, not from a generator I wrote), and re-run the full harness with `ANTHROPIC_API_KEY` set.
   This alone would probably invalidate several numbers in Section 3, which is the point.
2. **Swap TF-IDF retrieval for embeddings** (e.g. a small sentence-embedding model) — Failure #3
   above is a real limitation, and embeddings should meaningfully improve grounding quality.
3. **Get a second human to hand-score the judge-agreement sample independently**, and report
   inter-annotator agreement between the two humans as a ceiling for what judge<->human agreement
   can even reach.
4. **Multi-turn context**: pull the full thread (not just the single inbound tweet) into the
   classification/drafting prompt, since real customer messages often only make sense with the
   prior turn.
5. **Cost/latency numbers**: track tokens and latency per request so "auto-handle" throughput can
   be compared against a human agent's, which matters for whether this is worth deploying at all.
