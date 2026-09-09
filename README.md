# AI Support Agent — AmazonHelp (Twitter Customer Support)

Classifies inbound customer tweets to `@AmazonHelp` into one of 10 intents, drafts a reply grounded
in how the brand has historically resolved similar issues, and decides whether to auto-handle or
escalate to a human — with a stated reason.

**Read `report.md` first**, especially Section 2 (data note) and Section 5 ("what's misleading
about my headline number") before looking at any metric below.

## TL;DR reproduction (under 15 minutes, no API key required)

```bash
git clone <this repo>
cd hiver-support-agent
pip install -r requirements.txt

# 1. Generate the synthetic dataset + golden eval set (schema-matched to the real Kaggle dataset)
python src/generate_synthetic_data.py
python src/build_golden_set.py

# 2. Baselines (trivial + keyword-rule + TF-IDF/LogReg)
python src/baselines.py

# 3. Run the agent (mock mode without an API key, real Claude with one) over the golden set
python src/pipeline.py            # full 200 examples, ~seconds in mock mode
# python src/pipeline.py --limit 10   # quick smoke test

# 4. Evaluate: classification metrics, escalation metrics, reply-quality judge, judge<->human agreement
python src/build_human_judge_sample.py   # only needed once
python src/eval_harness.py
```

All of the above runs with **zero external API calls** by default (mock mode), so it's reproducible
without any credentials. To use the real LLM:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# optional: export ANTHROPIC_MODEL=claude-sonnet-4-6
python src/pipeline.py
python src/eval_harness.py
```

Every script prints and every output row records whether it ran in `mock` or `real` mode — never
silently degraded.

## Using the real Kaggle dataset instead of the synthetic one

This sandbox couldn't reach kaggle.com to download data directly, so `src/generate_synthetic_data.py`
builds a synthetic dataset with the **exact same schema** as the real one. To use real data:

1. Download `twcs.csv` from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
   (free Kaggle account required; or use the `kaggle` CLI with an API token from
   kaggle.com/settings → API).
2. Filter to your brand and build a hand-labeling template in one step:
   ```bash
   python src/prepare_real_data.py --raw_csv /path/to/twcs.csv --brand AmazonHelp
   ```
   This streams the ~3M-row file in chunks (won't blow up memory), writes the filtered thread to
   `data/raw/tweets.csv`, and writes `data/golden/golden_eval_TEMPLATE.csv` — a random sample of
   real inbound tweets with empty `gold_intent` / `gold_escalate` / `gold_escalate_reason` columns.
3. (Optional but recommended) Get an LLM-suggested first pass instead of labeling from a blank sheet:
   ```bash
   python src/pre_label_golden_set.py
   ```
   Writes `data/golden/golden_eval_SUGGESTED.csv` with suggested labels + a `confidence` score,
   sorted so the lowest-confidence (most likely wrong) rows are at the top. **This does not replace
   hand-labeling** — you still review and correct every row, mark `reviewed=1` as you go, and only
   the final human-confirmed labels count. What it changes is that you're correcting a first draft
   instead of typing 660 cells from nothing. See the script's docstring for the full workflow and
   why an eval set needs a human in the loop, not just an LLM, at the end of it.
4. Save your reviewed/corrected file as `data/golden/golden_eval.csv` (columns: `tweet_id, text,
   gold_intent, gold_escalate, gold_escalate_reason`). 150-250 rows recommended.
5. Nothing else changes — `python src/pipeline.py` and `python src/eval_harness.py` run exactly as
   before, now against real data and real hand labels.

Other brands worth trying (check exact `author_id` capitalization in the data first):
`AppleSupport`, `Uber_Support`, `SpotifyCares`, `Delta`, `AmazonHelp`, `AmericanAir`.

## Repo structure

```
src/
  intents.py                    # intent taxonomy + escalation policy defaults
  generate_synthetic_data.py    # synthetic dataset generator (Kaggle-schema-compatible)
  build_golden_set.py           # stratified sample + hand-coded gold labels -> golden_eval.csv
  retrieval.py                  # TF-IDF retrieval over historical (customer, agent-reply) pairs
  baselines.py                  # trivial / keyword-rule / TF-IDF+LogReg baselines
  llm_agent.py                  # the actual agent: classify + draft (grounded) + escalate
  decision.py                   # rule-based policy layer on top of the model's escalate flag
  pipeline.py                   # runs the agent over the golden set -> outputs/predictions.csv
  build_human_judge_sample.py   # builds the human hand-scoring template for judge agreement
  eval_harness.py               # all metrics: classification, escalation, LLM-as-judge, agreement
data/
  raw/tweets.csv                       # synthetic, Kaggle-schema-compatible
  golden/golden_eval.csv               # 200 hand-labeled (well, policy-labeled, see report.md §5) examples
  golden/human_judge_sample.csv        # 30-example judge-agreement sample (placeholder scores, see docstring)
outputs/
  predictions.csv, reply_quality.csv, baseline_comparison.txt
report.md          # problem framing, results, failure analysis, limitations, next steps
decision_log.md    # 15 non-obvious decisions and why
```

## What's actually being tested here

Given this brand's traffic pattern, the highest-leverage things to get right are: (1) never let a
refund/billing/account-access request get auto-answered without a human, (2) never draft a reply
that invents a policy detail (refund amount, timeline) the brand hasn't actually offered before, and
(3) keep the false-escalation rate low enough that "always escalate" isn't secretly the strategy.
Section 1 of `report.md` covers this in full, including what was deliberately left out of scope.
