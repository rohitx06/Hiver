"""
Prepares REAL Kaggle data for this pipeline. Run this instead of
generate_synthetic_data.py once you've downloaded twcs.csv from
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Usage:
    python src/prepare_real_data.py --raw_csv /path/to/twcs.csv --brand AmazonHelp

What it does:
  1. Filters the ~3M row file down to threads involving the chosen brand
     (both the customer tweets directed at them AND the brand's replies),
     keeping the exact same columns the rest of the pipeline expects.
  2. Writes data/raw/tweets.csv (overwrites the synthetic one).
  3. Writes data/golden/golden_eval_TEMPLATE.csv: a stratified-ish random
     sample of N inbound tweets to that brand, with EMPTY label columns
     (gold_intent, gold_escalate, gold_escalate_reason) for you to hand-fill.
     This replaces build_golden_set.py, which only works against the
     synthetic generator's internal ground truth and has no way to know the
     real intent of a real tweet.

After this:
  1. Open data/golden/golden_eval_TEMPLATE.csv, read each tweet, fill in
     gold_intent (must be one of src/intents.py's INTENT_LIST), gold_escalate
     (0/1), and gold_escalate_reason (a short sentence). Save it as
     data/golden/golden_eval.csv when done (150-250 rows recommended).
  2. Everything else (pipeline.py, eval_harness.py, etc.) runs unchanged.
"""
import argparse
import pandas as pd
from intents import INTENT_LIST


def filter_to_brand(raw_csv, brand, chunksize=200_000):
    """Streams the large CSV in chunks to avoid loading all ~3M rows into
    memory at once, and keeps only rows involving the target brand."""
    keep_chunks = []
    for chunk_num, chunk in enumerate(pd.read_csv(raw_csv, chunksize=chunksize, dtype=str)):
        is_brand_reply = chunk["author_id"] == brand
        # customer tweets that were replied-to by the brand, OR that reply to the brand
        brand_tweet_ids = set(chunk.loc[is_brand_reply, "tweet_id"])
        is_reply_to_brand = chunk["response_tweet_id"].fillna("").apply(
            lambda x: bool(brand_tweet_ids & set(x.split(","))) if x else False
        )
        is_customer_to_brand = chunk["in_response_to_tweet_id"].isin(brand_tweet_ids)
        keep = chunk[is_brand_reply | is_reply_to_brand | is_customer_to_brand]
        if len(keep):
            keep_chunks.append(keep)
        print(f"  chunk {chunk_num + 1} ({(chunk_num + 1) * chunksize:,} rows read so far), "
              f"{len(keep)} matched in this chunk", flush=True)
    if not keep_chunks:
        raise ValueError(f"No rows found for brand '{brand}'. Check the exact author_id "
                          f"spelling (e.g. 'AmazonHelp', 'AppleSupport', 'Uber_Support').")
    return pd.concat(keep_chunks, ignore_index=True)


def make_labeling_template(brand_df, n_sample=220, seed=7):
    inbound = brand_df[brand_df["author_id"] != brand_df["author_id"].mode()[0]]
    # crude inbound filter: rows NOT authored by the brand account itself
    inbound = brand_df[brand_df["inbound"].astype(str).str.lower() == "true"]
    n = min(n_sample, len(inbound))
    sample = inbound.sample(n=n, random_state=seed)[["tweet_id", "text"]].copy()
    sample["gold_intent"] = ""  # fill with one of: " + ", ".join(INTENT_LIST)
    sample["gold_escalate"] = ""       # 0 or 1
    sample["gold_escalate_reason"] = ""  # short sentence
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True, help="path to the downloaded twcs.csv")
    parser.add_argument("--brand", default="AmazonHelp", help="exact author_id of the brand account")
    parser.add_argument("--n_sample", type=int, default=220,
                         help="rows to sample into the labeling template (aim for 150-250 after filtering)")
    args = parser.parse_args()

    print(f"Filtering {args.raw_csv} to brand '{args.brand}' (this streams the file, may take ~1-2 min)...")
    brand_df = filter_to_brand(args.raw_csv, args.brand)
    print(f"Kept {len(brand_df)} rows ({(brand_df['inbound'].astype(str).str.lower()=='true').sum()} inbound).")

    cols = ["tweet_id", "author_id", "inbound", "created_at", "text",
            "response_tweet_id", "in_response_to_tweet_id"]
    brand_df[cols].to_csv("data/raw/tweets.csv", index=False)
    print("Wrote data/raw/tweets.csv (real data, overwrote the synthetic file).")

    template = make_labeling_template(brand_df, n_sample=args.n_sample)
    template.to_csv("data/golden/golden_eval_TEMPLATE.csv", index=False)
    print(f"Wrote data/golden/golden_eval_TEMPLATE.csv with {len(template)} rows to hand-label.")
    print(f"\nValid gold_intent values: {', '.join(INTENT_LIST)}")
    print("Fill in the 3 label columns by hand, save as data/golden/golden_eval.csv, "
          "then run: python src/pipeline.py")
