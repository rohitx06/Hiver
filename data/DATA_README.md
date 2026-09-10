# Data

`data/raw/tweets.csv` is a filtered subset of the Customer Support on Twitter dataset containing
AmazonHelp-related threads. The full public dataset is not committed to this repository.

For a fresh reproduction from the source dataset, download `twcs.csv` from the Kaggle dataset
`thoughtvector/customer-support-on-twitter`, then run:

```bash
python src/prepare_real_data.py --raw_csv /path/to/twcs.csv --brand AmazonHelp --n_sample 220
```

The resulting file is a real-data subset suitable for retrieval. Do not commit API keys or a
`.env` file.
