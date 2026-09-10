# Hiver SDE Intern Take-Home — AmazonHelp Support Agent

An AI customer-support triage agent for inbound `@AmazonHelp` Twitter/X messages.

The system:

1. Classifies customer messages into 10 support intents derived from the AmazonHelp data.
2. Retrieves similar historical customer/agent interactions using TF-IDF cosine similarity.
3. Drafts a short support reply grounded in historical responses.
4. Uses a Groq-hosted LLM to determine whether the case should be escalated.
5. Applies a deterministic safety policy on top of the model decision.
6. Evaluates intent classification, escalation decisions, and reply quality.

## Architecture

```text
Inbound customer tweet
        |
        v
TF-IDF Retrieval
        |
        +----> Similar historical customer/agent examples
        |
        v
Groq LLM
        |
        +----> Intent
        +----> Confidence
        +----> Draft reply
        +----> Model escalation decision
        |
        v
Deterministic Safety Policy
        |
        v
AUTO-HANDLE / ESCALATE
```

The LLM proposes the classification, response and escalation decision. The deterministic policy layer provides an additional safety boundary for higher-risk cases such as refunds, billing disputes and account access.

The system **never automatically sends a customer message**. It only produces a draft and routing recommendation.

---

## Tech Stack

* **Language:** Python
* **LLM:** Groq Chat Completions
* **Model:** `openai/gpt-oss-20b`
* **Retrieval:** TF-IDF + cosine similarity
* **Classification baselines:** Majority class, keyword rules, TF-IDF + Logistic Regression
* **Evaluation:** Accuracy, precision, recall, F1, escalation metrics, LLM-as-judge
* **Dataset:** Customer Support on Twitter — AmazonHelp subset

---

## Quick Start

### 1. Create the environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Groq

Create `.env` in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit `.env` or an API key.

### 3. Run the LLM smoke test

From the repository root:

```bash
python -m src.llm_agent
```

Expected:

```text
Running in REAL mode (GROQ_API_KEY found)
```

The smoke test also retrieves historical AmazonHelp interactions and demonstrates the complete agent output.

### 4. Run the pipeline

Quick 10-example test:

```bash
python -m src.pipeline --limit 10
```

Full golden evaluation:

```bash
python -m src.pipeline
```

Predictions are written to:

```text
outputs/predictions.csv
```

Each prediction records whether the result came from `real` or `mock` mode.

---

## Evaluation

Run:

```bash
python -m src.eval_harness
```

The evaluation covers:

### Intent classification

* Accuracy
* Per-intent precision
* Per-intent recall
* Per-intent F1

### Escalation

* Precision
* Recall
* F1
* Missed escalations
* Unnecessary escalations

### Reply quality

A subset of replies is evaluated using an LLM judge on:

* Grounding
* Correctness
* Tone
* Actionability

Human ratings can additionally be entered in:

```text
data/golden/human_judge_sample.csv
```

The harness does not calculate judge-human agreement until the human rating fields are completed.

---

## Headline Evaluation Result

On the 220-example golden evaluation set, the current system achieved:

| Metric               |    Result |
| -------------------- | --------: |
| Intent accuracy      | **50.9%** |
| Escalation precision | **56.8%** |
| Escalation recall    | **38.3%** |
| Escalation F1        | **45.8%** |

Detailed per-intent results and failure analysis are included in `report.md`.

The evaluation set contains multiple closely related support categories, making intent boundaries ambiguous for some messages.

---

## LLM-as-Judge Note

The reply-quality judge successfully evaluated 8 of the 30 sampled examples before the Groq token-per-day limit was reached.

Therefore, the resulting reply-quality averages are treated as **exploratory rather than statistically definitive**.

The failed judge calls were not converted into scores.

This limitation is documented in the evaluation output and report.

---

## Golden Evaluation Set

The repository contains a 220-example AmazonHelp evaluation set.

The required intent fields are:

* `gold_intent`
* `gold_escalate`
* `gold_escalate_reason`

The labeling guide is available at:

```text
data/golden/HAND_LABELING_GUIDE.md
```

The evaluation data should be interpreted together with the labeling methodology described in the report.

---

## Real Dataset

The repository contains a filtered AmazonHelp subset at:

```text
data/raw/tweets.csv
```

The original Customer Support on Twitter dataset is not required to run the standard evaluation.

To rebuild the AmazonHelp subset from the original dataset:

```bash
python -m src.prepare_real_data \
    --raw_csv /path/to/twcs.csv \
    --brand AmazonHelp \
    --n_sample 220
```

The full source dataset should not be committed to the repository.

---

## Repository Structure

```text
src/
├── intents.py
├── retrieval.py
├── llm_agent.py
├── decision.py
├── pipeline.py
├── baselines.py
├── eval_harness.py
├── prepare_real_data.py
├── pre_label_golden_set.py
└── build_human_judge_sample.py

data/
├── raw/
│   └── tweets.csv
└── golden/
    ├── golden_eval.csv
    ├── golden_eval_TEMPLATE.csv
    ├── HAND_LABELING_GUIDE.md
    └── human_judge_sample.csv

outputs/
├── predictions.csv
├── reply_quality.csv
└── baseline_comparison.txt

report.md
decision_log.md
requirements.txt
README.md
```

---

## Key Design Decisions

### Retrieval-grounded generation

The model receives similar historical AmazonHelp customer/agent interactions rather than generating support responses entirely from general knowledge.

This reduces unsupported policy claims and helps match the historical support tone.

### Single LLM call

Classification, drafting and model-level escalation are performed in a single call.

This reduces latency and token usage while keeping the decisions based on the same interpretation of the customer message.

### Deterministic safety layer

The final escalation decision is not based entirely on the LLM.

Higher-risk categories such as refunds, billing disputes and account-access cases are protected by deterministic rules.

### No automatic sending

The system produces:

```text
classification
draft response
escalation recommendation
escalation reason
```

It does not send messages to customers.

---

## Limitations

The current evaluation highlights several limitations:

1. Intent boundaries can be ambiguous for short or context-dependent tweets.
2. Retrieval can surface historical URLs, handles or other artifacts that require filtering before production use.
3. The current LLM-as-judge evaluation is limited by API token availability.
4. Escalation recall remains an important area for improvement.
5. The golden-set labeling methodology and sample size affect the reliability of the headline metrics.

These limitations and proposed improvements are discussed in `report.md`.

---

## Reproducibility

From a clean environment:

```bash
pip install -r requirements.txt
```

Configure:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Then:

```bash
python -m src.pipeline --limit 10
```

For the complete evaluation:

```bash
python -m src.pipeline
python -m src.eval_harness
```

The repository contains the filtered AmazonHelp data required for the standard run.
