"""Held-out multi-label check: join two never-seen examples so that two labels are true at once.

Compares the native multi-label head ("any" mode) with a decomposition into one yes/no question per option.

Usage: uv run python scripts/eval_multilabel.py [--model checkpoints/decision-model]
"""

import argparse
import random

import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from decisionmodel import Bool, DecisionModel, MultiLabel


def build(n=600, seed=0):
    rng = random.Random(seed)
    sets = []
    news = load_dataset("fancyzhx/ag_news", split="test")
    topics = ["world", "sports", "business", "science and technology"]
    items = [(t, topics[y]) for t, y in zip(news["text"], news["label"])]
    sets.append(("ag_news pairs", "Which of these topics are covered in the text?", items, topics, 4))
    bank = load_dataset("mteb/banking77", split="test")
    intents = sorted({l.replace("_", " ") for l in bank["label_text"]})
    items = [(t, l.replace("_", " ")) for t, l in zip(bank["text"], bank["label_text"])]
    sets.append(("banking77 pairs", "Which of these does the customer ask about?", items, intents, 8))
    out = []
    for name, q, items, labels, k in sets:
        rows = []
        for _ in range(n):
            parts = rng.sample(items, rng.choice([1, 2, 2]))
            true = {l for _, l in parts}
            cand = list(true | set(rng.sample(labels, min(k, len(labels)))))
            rng.shuffle(cand)
            rows.append((" Also: ".join(t for t, _ in parts), cand, [c in true for c in cand]))
        out.append((name, q, rows))
    return out


def report(tag, y, p):
    y, p = np.array(y), np.array(p)
    print(f"    {tag:34s} AUROC {roc_auc_score(y, p):.3f}  AP {average_precision_score(y, p):.3f}  "
          f"F1@0.5 {f1_score(y, p >= 0.5):.3f}  predicted positive rate {np.mean(p >= 0.5):.2f} (true {y.mean():.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    dm = DecisionModel.load(args.model)
    for name, q, rows in build():
        print(f"  {name}")
        y, p_native, p_bool = [], [], []
        for text, cand, truth in rows:
            d = dm.decide(text, {"m": MultiLabel(q, cand), **{f"b{i}": Bool(f'{q} Does the label "{c}" apply?') for i, c in enumerate(cand)}})
            y += truth
            p_native += [d.m.probs[c] for c in cand]
            p_bool += [d[f"b{i}"].p() for i in range(len(cand))]
        report("native multi-label head", y, p_native)
        report("one yes/no question per option", y, p_bool)


if __name__ == "__main__":
    main()
