# Submission Checklist

Complete these items before submitting the Notion form.

## Required

- [ ] Public GitHub repository (or private repo with reviewer access)
- [ ] `README.md` reproduces the final headline results in <15 minutes
- [ ] Real AmazonHelp data is used by the pipeline
- [ ] 150–250 real, independently hand-labelled golden examples
- [ ] `data/golden/golden_eval.csv` contains the final human labels
- [ ] Two baselines: trivial + keyword
- [ ] Groq LLM results on the same golden set
- [ ] Escalation precision/recall/F1 and missed-escalation count
- [ ] LLM-as-judge reply-quality results
- [ ] 30-example human rating sample with every score manually entered
- [ ] Judge-human agreement reported
- [ ] Five failure modes with real examples
- [ ] "What is misleading about my headline number?" section
- [ ] 10–15 decision-log entries
- [ ] Report <= 6 pages
- [ ] No API keys, `.env`, or credentials committed
- [ ] No unnecessary full 3M-row source dataset committed

## Final commands

After completing the golden labels:

```bash
python src/pipeline.py
python src/baselines.py
python src/build_human_judge_sample.py
# Manually score data/golden/human_judge_sample.csv
python src/eval_harness.py
```

Then update `report.md` with the actual printed results and five actual failure examples.

## What NOT to claim

Do not claim that a model achieved a metric unless the run was made with:
- the real Groq API (`mode=real`);
- the final hand-labelled real golden set;
- the final code in the submitted repository.

Do not describe LLM-generated suggested labels as "hand-labelled."
