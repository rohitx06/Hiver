
"""
Builds an index of historical (customer_text -> agent_reply) pairs and
retrieves the top-k most similar past cases for a new incoming message.

This grounds drafted replies in how AmazonHelp historically responded to
similar customer issues.

Implementation:
    - Pandas merge builds customer/reply pairs efficiently.
    - TF-IDF + cosine similarity retrieves similar historical cases.
    - No vector database or embeddings API is required.
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoryIndex:

    def __init__(self, raw_csv="data/raw/tweets.csv"):

        print(f"Loading historical support data from {raw_csv}...")

        # -------------------------------------------------
        # Load only the columns we actually need.
        # This reduces memory usage significantly.
        # -------------------------------------------------

        df = pd.read_csv(
            raw_csv,
            dtype=str,
            usecols=[
                "tweet_id",
                "text",
                "inbound",
                "in_response_to_tweet_id",
            ],
        )

        # -------------------------------------------------
        # Normalize inbound field
        # -------------------------------------------------

        df["inbound"] = (
            df["inbound"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

        # -------------------------------------------------
        # Separate customer and agent tweets
        # -------------------------------------------------

        inbound = df[
            df["inbound"]
        ][
            [
                "tweet_id",
                "text",
            ]
        ].copy()

        outbound = df[
            ~df["inbound"]
        ][
            [
                "in_response_to_tweet_id",
                "text",
            ]
        ].copy()

        # -------------------------------------------------
        # Rename columns before merge
        # -------------------------------------------------

        inbound = inbound.rename(
            columns={
                "tweet_id": "customer_tweet_id",
                "text": "customer_text",
            }
        )

        outbound = outbound.rename(
            columns={
                "in_response_to_tweet_id":
                    "customer_tweet_id",
                "text":
                    "agent_reply",
            }
        )

        # -------------------------------------------------
        # Remove rows without a reply reference
        # -------------------------------------------------

        outbound = outbound.dropna(
            subset=["customer_tweet_id"]
        )

        # -------------------------------------------------
        # Build customer -> agent reply pairs.
        #
        # This replaces the slow Python iterrows() loop.
        # -------------------------------------------------

        pairs = inbound.merge(
            outbound,
            on="customer_tweet_id",
            how="inner",
        )

        # -------------------------------------------------
        # Remove missing/empty text
        # -------------------------------------------------

        pairs = pairs.dropna(
            subset=[
                "customer_text",
                "agent_reply",
            ]
        )

        pairs["customer_text"] = (
            pairs["customer_text"]
            .astype(str)
            .str.strip()
        )

        pairs["agent_reply"] = (
            pairs["agent_reply"]
            .astype(str)
            .str.strip()
        )

        pairs = pairs[
            (pairs["customer_text"] != "")
            & (pairs["agent_reply"] != "")
        ].copy()

        # -------------------------------------------------
        # Keep only required fields
        # -------------------------------------------------

        self.pairs = pairs[
            [
                "customer_text",
                "agent_reply",
            ]
        ].reset_index(drop=True)

        print(
            f"Indexed {len(self.pairs):,} "
            "historical customer/reply pairs."
        )

        # -------------------------------------------------
        # TF-IDF index
        # -------------------------------------------------

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )

        print("Building TF-IDF index...")

        self.matrix = self.vectorizer.fit_transform(
            self.pairs["customer_text"]
        )

        print(
            f"TF-IDF matrix built: "
            f"{self.matrix.shape[0]:,} documents × "
            f"{self.matrix.shape[1]:,} features."
        )

    # -----------------------------------------------------
    # Retrieve similar historical cases
    # -----------------------------------------------------

    def retrieve(
        self,
        text,
        k=3,
        exclude_texts=None,
    ):

        # Convert query into TF-IDF representation.
        q = self.vectorizer.transform(
            [text]
        )

        # Calculate similarity against historical cases.
        sims = cosine_similarity(
            q,
            self.matrix,
        )[0]

        # Exact-text exclusions prevent golden-set leakage.
        excluded = set(
            exclude_texts or []
        )

        # Highest similarity first.
        ranked_idx = sims.argsort()[::-1]

        results = []

        for i in ranked_idx:

            customer_text = (
                self.pairs.iloc[i][
                    "customer_text"
                ]
            )

            if customer_text in excluded:
                continue

            results.append(
                {
                    "customer_text":
                        customer_text,

                    "agent_reply":
                        self.pairs.iloc[i][
                            "agent_reply"
                        ],

                    "similarity":
                        float(sims[i]),
                }
            )

            if len(results) >= k:
                break

        return results


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    idx = HistoryIndex()

    print(
        f"\nIndexed {len(idx.pairs):,} "
        "historical (customer, agent) pairs."
    )

    demo = idx.retrieve(
        "my order is a week late and never showed up",
        k=3,
    )

    for r in demo:

        print(
            f"\n[{r['similarity']:.2f}] "
            f"CUST: {r['customer_text']}"
        )

        print(
            f"       AGENT: {r['agent_reply']}"
        )
