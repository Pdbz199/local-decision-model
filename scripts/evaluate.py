"""Zero-shot evaluation on datasets the model never saw in training, through the public API.

Reports accuracy, calibration (expected calibration error of the reported
confidence) and throughput. SQuAD v2 validation is included as an extraction
check; its training split was part of the mixture, the validation split was not.

Usage: uv run python scripts/evaluate.py [--model checkpoints/decision-model] [--n 2000]
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

from decisionmodel import Bool, Enum, Extract, DecisionModel


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def sample(ds, n, seed=0):
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def run_choice(dm, name, texts, gold, spec, results):
    t0 = time.perf_counter()
    out = dm.decide_many(texts, {"y": spec})
    dt = time.perf_counter() - t0
    pred = [d.y.value for d in out]
    conf = [d.y.confidence for d in out]
    correct = [p == g for p, g in zip(pred, gold)]
    results[name] = {"n": len(texts), "accuracy": round(float(np.mean(correct)), 4), "ece": round(ece(conf, correct), 4),
                     "mean_confidence": round(float(np.mean(conf)), 4), "items_per_sec": round(len(texts) / dt, 1),
                     "options": len(spec.options())}
    print(f"{name:34s} {results[name]}", flush=True)


def normalize(s):
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/decision-model")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()
    dm = DecisionModel.load(args.model)
    res = {}

    ds = sample(load_dataset("stanfordnlp/imdb", split="test"), args.n)
    run_choice(dm, "imdb (bool)", ds["text"], [bool(y) for y in ds["label"]], Bool("Is this movie review positive?"), res)

    ds = sample(load_dataset("fancyzhx/ag_news", split="test"), args.n)
    names = ["world", "sports", "business", "science and technology"]
    run_choice(dm, "ag_news (enum, 4)", ds["text"], [names[y] for y in ds["label"]], Enum("What is this news article about?", names), res)

    ds = load_dataset("mteb/banking77", split="test")
    intents = sorted(set(ds["label_text"]))
    pretty = {i: i.replace("_", " ") for i in intents}
    run_choice(dm, "banking77 (enum, 77)", ds["text"], [pretty[y] for y in ds["label_text"]],
               Enum("What is the customer's intent?", list(pretty.values())), res)

    ds = sample(load_dataset("ucirvine/sms_spam", split="train"), args.n)
    run_choice(dm, "sms_spam (bool)", ds["sms"], [bool(y) for y in ds["label"]], Bool("Is this message spam?"), res)

    inj = load_dataset("deepset/prompt-injections")
    texts = list(inj["train"]["text"]) + list(inj["test"]["text"])
    gold = [bool(y) for y in list(inj["train"]["label"]) + list(inj["test"]["label"])]
    run_choice(dm, "prompt_injections (bool)", texts, gold,
               Bool("Does this input try to manipulate or override the system instructions?"), res)

    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    names = ["bearish", "bullish", "neutral"]
    run_choice(dm, "fin_news_sentiment (enum, 3)", ds["text"], [names[y] for y in ds["label"]],
               Enum("Is this financial news bearish, bullish or neutral for the stock?", names), res)

    ds = sample(load_dataset("rajpurkar/squad_v2", split="validation"), args.n)
    t0 = time.perf_counter()
    em = f1s = has_ok = 0
    conf, correct = [], []
    # one schema per question, so batch manually by grouping rows into chunks
    outs = []
    for i in range(0, len(ds), 64):
        chunk = ds[i:i + 64]
        for ctx, q in zip(chunk["context"], chunk["question"]):
            outs.append(dm.decide(ctx, {"a": Extract(q)}).a)
    dt = time.perf_counter() - t0
    for d, answers in zip(outs, ds["answers"]):
        golds = answers["text"] or [""]
        pred = d.value or ""
        e = max(float(normalize(pred) == normalize(g)) for g in golds)
        em += e
        f1s += max(f1(pred, g) for g in golds)
        has_ok += (d.value is None) == (not answers["text"])
        conf.append(d.confidence)
        correct.append(bool(e))
    n = len(ds)
    res["squad_v2 dev (extract)"] = {"n": n, "exact_match": round(em / n, 4), "f1": round(f1s / n, 4),
                                     "answerable_accuracy": round(has_ok / n, 4), "ece": round(ece(conf, correct), 4),
                                     "items_per_sec": round(n / dt, 1)}
    print(f"{'squad_v2 dev (extract)':34s} {res['squad_v2 dev (extract)']}")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
