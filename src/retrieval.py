"""
Builds an index of historical (customer_text -> agent_reply) pairs and
retrieves the top-k most similar past cases for a new incoming message.
This is what "grounds" the drafted reply in how the brand has actually
resolved similar issues before, instead of letting the LLM freelance.

Kept deliberately simple: TF-IDF + cosine similarity over customer message
text. No vector DB / embeddings API needed, runs in-memory in <1s for
thousands of rows, which is enough for a single-brand support inbox. See
report.md "What we'd do next" for when this stops being enough.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoryIndex:
    def __init__(self, raw_csv="data/raw/tweets.csv"):
        df = pd.read_csv(raw_csv, dtype=str)
        df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
        inbound = df[df["inbound"] == True].copy()
        outbound = df[df["inbound"] == False].copy()

        # Join each customer tweet to the agent reply that responded to it
        outbound_by_reply_to = outbound.set_index("in_response_to_tweet_id")
        pairs = []
        for _, row in inbound.iterrows():
            tid = str(row["tweet_id"])
            if tid in outbound_by_reply_to.index:
                reply_row = outbound_by_reply_to.loc[tid]
                if isinstance(reply_row, pd.DataFrame):
                    reply_row = reply_row.iloc[0]
                pairs.append({
                    "customer_text": row["text"],
                    "agent_reply": reply_row["text"],
                })
        self.pairs = pd.DataFrame(pairs)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.pairs["customer_text"])

    def retrieve(self, text, k=3):
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            results.append({
                "customer_text": self.pairs.iloc[i]["customer_text"],
                "agent_reply": self.pairs.iloc[i]["agent_reply"],
                "similarity": float(sims[i]),
            })
        return results


if __name__ == "__main__":
    idx = HistoryIndex()
    print(f"Indexed {len(idx.pairs)} historical (customer, agent) pairs.")
    demo = idx.retrieve("my order is a week late and never showed up", k=3)
    for r in demo:
        print(f"\n[{r['similarity']:.2f}] CUST: {r['customer_text']}\n       AGENT: {r['agent_reply']}")
