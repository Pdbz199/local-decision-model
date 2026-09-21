"""Experiment 3: map-reduce over a large pile of documents.

Map: run a small schema over thousands of news articles and movie reviews the
model never trained on. Reduce: aggregate the typed outputs with ordinary
Python. Because outputs are probabilities, the reduce step can use expected
counts instead of hard votes. We measure throughput, since this workload is
about cost per item rather than latency.
"""

import time
from collections import Counter

import numpy as np
from common import check, finish, header
from datasets import load_dataset

from decisionmodel import Bool, Enum, DecisionModel

TOPICS = ["world", "sports", "business", "science and technology"]


def main():
    dm = DecisionModel.load()
    header("Experiment 3: map-reduce over 7,600 news articles and 2,000 long reviews")

    news = load_dataset("fancyzhx/ag_news", split="test")
    schema = {"topic": Enum("What is this news article about?", TOPICS),
              "markets": Bool("Does this article discuss stock prices, earnings or financial markets?")}
    n_tok = sum(len(t) for t in dm.encoder.tok(list(news["text"]), add_special_tokens=False)["input_ids"])
    t0 = time.perf_counter()
    out = dm.decide_many(news["text"], schema)
    dt = time.perf_counter() - t0
    gold = [TOPICS[y] for y in news["label"]]
    acc = np.mean([d.topic.value == g for d, g in zip(out, gold)])

    hard = Counter(d.topic.value for d in out)
    soft = {t: sum(d.topic.probs[t] for d in out) for t in TOPICS}
    true = Counter(gold)
    print(f"\n  MAP    {len(news)} articles x {len(schema)} fields = {len(news) * len(schema)} decisions in {dt:.1f} s")
    print(f"         {len(news) / dt:,.0f} articles/s, {len(news) * len(schema) / dt:,.0f} decisions/s, "
          f"{n_tok * len(schema) / dt:,.0f} state tokens/s")
    print(f"  REDUCE topic counts (true / hard votes / expected counts from probabilities):")
    for t in TOPICS:
        print(f"         {t:24s} {true[t]:5d} / {hard[t]:5d} / {soft[t]:7.1f}")
    share = np.mean([d.markets.p() for d, g in zip(out, gold) if g == "business"])
    share_sports = np.mean([d.markets.p() for d, g in zip(out, gold) if g == "sports"])
    print(f"         mean P(discusses markets): business articles {share:.2f}, sports articles {share_sports:.2f}")
    print(f"  zero-shot topic accuracy: {acc:.3f} (chance is 0.25). Most misses are tech-company business stories, which this")
    print("  dataset files under Sci/Tech and the model files under business: a labeling convention it was never shown.")

    reviews = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=0).select(range(2000))
    r_tok = sum(min(len(t), 8192) for t in dm.encoder.tok(list(reviews["text"]), add_special_tokens=False)["input_ids"])
    t0 = time.perf_counter()
    rout = dm.decide_many(reviews["text"], {"positive": Bool("Is this movie review positive?")})
    rdt = time.perf_counter() - t0
    racc = np.mean([d.positive.value == bool(y) for d, y in zip(rout, reviews["label"])])
    true_pos, soft_pos = int(sum(reviews["label"])), sum(d.positive.p() for d in rout)
    print(f"\n  Long inputs: 2,000 IMDB reviews (mean {r_tok / 2000:.0f} tokens) in {rdt:.1f} s = "
          f"{2000 / rdt:,.0f} reviews/s, {r_tok / rdt:,.0f} tokens/s, accuracy {racc:.3f}")
    print(f"  REDUCE positive reviews: true {true_pos}, expected count from probabilities {soft_pos:.1f}")
    print(f"  A billion input tokens would take about {1e9 / (r_tok / rdt) / 3600:.1f} GPU hours on this one consumer card.")

    print("\nChecks:")
    check("zero-shot news topic accuracy above 0.70 (chance 0.25)", acc > 0.70, f"{acc:.3f}")
    check("zero-shot review sentiment accuracy above 0.90", racc > 0.90, f"{racc:.3f}")
    check("sports count from probabilities within 10% of truth", abs(soft["sports"] - true["sports"]) / true["sports"] < 0.10)
    check("expected number of positive reviews within 5% of truth", abs(soft_pos - true_pos) / true_pos < 0.05, f"{soft_pos:.0f} vs {true_pos}")
    check("market predicate separates business from sports", share > share_sports + 0.3)
    check("throughput above 500 decisions per second", len(news) * len(schema) / dt > 500)
    check("throughput above 50,000 tokens per second on long inputs", r_tok / rdt > 50_000, f"{r_tok / rdt:,.0f}")
    finish()


if __name__ == "__main__":
    main()
