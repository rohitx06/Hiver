# AI Support Agent for AmazonHelp — SDE Intern Take-Home Report

## 1. Problem framing

**Brand:** AmazonHelp.

The goal is not to build a general chatbot. The system is a support-triage agent that must make
three linked decisions:

1. classify the customer's intent;
2. draft a short response grounded in historical AmazonHelp replies;
3. decide whether the case can be auto-handled or should go to a human.

For this project, a good system prioritizes **safe routing over aggressive automation**. Refunds,
billing disputes, account-access/security issues, ambiguous cases, and repeated escalations should
not be silently auto-resolved.

### Deliberately out of scope

- multi-turn dialogue management;
- fine-tuning;
- automatic sending of replies;
- secure identity-verification workflows;
- production observability/authentication infrastructure.

## 2. Data and evaluation design

The source is the Customer Support on Twitter dataset. The project filters it to AmazonHelp and
keeps AmazonHelp-related customer/agent threads. In the supplied source snapshot, the filter produced
357,883 AmazonHelp-related rows, including 188,042 inbound customer rows.

A 220-example golden set is sampled from real inbound AmazonHelp messages. **The final labels must
be independently hand-confirmed before submission.** `data/golden/HAND_LABELING_GUIDE.md` defines
the intent and escalation rubric.

The retrieval corpus and golden set should be treated as separate evaluation/training resources;
the golden examples are excluded from any supervised baseline training.

## 3. System architecture

```text
Customer tweet
     |
     v
TF-IDF retrieval over historical AmazonHelp exchanges
     |
     +---- similar customer/agent examples
     |
     v
Groq LLM
     |
     +---- intent
     +---- confidence
     +---- grounded draft
     +---- escalation recommendation
     |
     v
Deterministic policy layer
     |
     v
Final auto-handle / escalate decision
```

The model receives the customer message, the intent taxonomy, and retrieved historical examples.
The reply is instructed not to invent refund amounts, timelines, or other unsupported policy
details.

## 4. Results

**Run these commands after the final human labels are completed:**

```bash
python src/pipeline.py
python src/baselines.py
python src/eval_harness.py
```

Insert the resulting numbers here. Do not report synthetic/mock numbers as LLM performance.

### Intent classification

| Model | Accuracy |
|---|---:|
| Majority baseline | **[FILL]** |
| Keyword rules | **[FILL]** |
| TF-IDF + Logistic Regression | **[FILL]** |
| Groq LLM + retrieval | **[FILL]** |

### Escalation

| Metric | Value |
|---|---:|
| Precision | **[FILL]** |
| Recall | **[FILL]** |
| F1 | **[FILL]** |
| Missed escalations | **[FILL]** |
| Unnecessary escalations | **[FILL]** |

For escalation, recall is the more important metric because a missed high-risk escalation is
more costly than an unnecessary human review.

### Reply quality

Report the four judge axes separately:

| Axis | Mean |
|---|---:|
| Grounding | **[FILL]** |
| Correctness | **[FILL]** |
| Tone | **[FILL]** |
| Actionability | **[FILL]** |

### Judge-human agreement

Use the 30 manually scored examples and report both within-one-point agreement and correlation:

| Axis | Within 1 point | Pearson r |
|---|---:|---:|
| Grounding | **[FILL]** | **[FILL]** |
| Correctness | **[FILL]** | **[FILL]** |
| Tone | **[FILL]** | **[FILL]** |
| Actionability | **[FILL]** | **[FILL]** |

## 5. Failure analysis

After the final evaluation, replace the placeholders below with five actual examples from the
prediction output.

### Failure 1 — [TITLE]

**Customer:** [REAL EXAMPLE]

**Prediction:** [WHAT THE SYSTEM DID]

**Why it failed:** [ANALYSIS]

**Hypothesis:** [WHY THIS HAPPENED]

### Failure 2 — [TITLE]

**Customer:** [REAL EXAMPLE]

**Prediction:** [WHAT THE SYSTEM DID]

**Why it failed:** [ANALYSIS]

**Hypothesis:** [WHY THIS HAPPENED]

### Failure 3 — [TITLE]

**Customer:** [REAL EXAMPLE]

**Prediction:** [WHAT THE SYSTEM DID]

**Why it failed:** [ANALYSIS]

**Hypothesis:** [WHY THIS HAPPENED]

### Failure 4 — [TITLE]

**Customer:** [REAL EXAMPLE]

**Prediction:** [WHAT THE SYSTEM DID]

**Why it failed:** [ANALYSIS]

**Hypothesis:** [WHY THIS HAPPENED]

### Failure 5 — [TITLE]

**Customer:** [REAL EXAMPLE]

**Prediction:** [WHAT THE SYSTEM DID]

**Why it failed:** [ANALYSIS]

**Hypothesis:** [WHY THIS HAPPENED]

## 6. What is misleading about my headline number?

A single intent-accuracy number is insufficient evidence that this system is safe to deploy.

First, intent accuracy says nothing about whether high-risk cases are escalated. A system could
have excellent classification accuracy while incorrectly auto-handling a small number of refunds
or account-security requests.

Second, the golden set is only 220 examples and is intentionally sampled to cover rare intents.
Its distribution therefore does not represent production traffic. A headline score on this set
should not be interpreted as expected production accuracy.

Third, reply quality is a different task from classification. A correct intent can still produce
a generic, unhelpful, or insufficiently grounded reply.

Fourth, the LLM-as-judge is itself a model and can have systematic preferences. The human-rating
sample exists specifically to test whether the judge's scores track human judgments.

Finally, the system is evaluated on historical text. Future support traffic may contain new
products, policies, slang, and failure modes not represented in the corpus.

## 7. What I would do with one more week

1. Add full thread context rather than only the inbound tweet.
2. Replace TF-IDF retrieval with sentence embeddings and compare retrieval recall.
3. Have a second human independently label a subset and report inter-annotator agreement.
4. Add latency/token/cost tracking for Groq requests.
5. Add adversarial evaluation for ambiguous, multi-intent, sarcastic, and repeated complaints.
6. Add automated regression tests for the high-risk escalation policy.
