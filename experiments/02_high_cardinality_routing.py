"""Experiment 2: high-cardinality routing on a task the model never trained on.

Banking77 has 77 customer intents. The model saw none of this dataset during
training; the intent names are passed as plain strings at call time. All 77
options are scored in a single forward pass. We also show why calibrated
confidence matters: automate the confident calls, escalate the rest.
"""

import numpy as np
from common import JEV_FAST_MS, check, finish, fmt, header, latency
from datasets import load_dataset

from decisionmodel import Enum, DecisionModel


def main():
    dm = DecisionModel.load()
    header("Experiment 2: 77-way intent routing, zero-shot")
    ds = load_dataset("mteb/banking77", split="test")
    intents = sorted(set(ds["label_text"]))
    names = [i.replace("_", " ") for i in intents]
    spec = Enum("What is the customer's intent?", names)
    gold = [g.replace("_", " ") for g in ds["label_text"]]

    out = dm.decide_many(ds["text"], {"intent": spec})
    pred = np.array([d.intent.value for d in out])
    conf = np.array([d.intent.confidence for d in out])
    correct = pred == np.array(gold)
    top3 = np.mean([g in sorted(d.intent.probs, key=d.intent.probs.get, reverse=True)[:3] for d, g in zip(out, gold)])

    for i in (0, 1, 2):
        best = sorted(out[i].intent.probs.items(), key=lambda kv: -kv[1])[:3]
        print(f"\n  '{ds['text'][i]}'\n    gold: {gold[i]}\n    top3: " + ", ".join(f"{k} ({v:.2f})" for k, v in best))

    print(f"\n  {len(ds)} messages, {len(names)} options, chance accuracy {1 / len(names):.3f}")
    print(f"  top-1 accuracy {correct.mean():.3f}   top-3 accuracy {top3:.3f}")
    print("\n  Selective automation (act only when confidence >= threshold, escalate otherwise):")
    print("    threshold  automated  accuracy on automated")
    rows = []
    for th in (0.0, 0.5, 0.7, 0.9):
        m = conf >= th
        rows.append((th, m.mean(), correct[m].mean() if m.any() else float("nan")))
        print(f"    {th:9.1f}  {m.mean():8.1%}  {rows[-1][2]:.3f}")

    print("\nChecks:")
    check("zero-shot top-1 accuracy at least 30x chance", correct.mean() > 30 / len(names), f"{correct.mean():.3f}")
    check("accuracy rises with the confidence threshold", rows[0][2] < rows[1][2] < rows[3][2])
    check("every prediction is one of the 77 declared options", set(pred) <= set(names))
    lat = latency(lambda: dm.decide(ds["text"][0], {"intent": spec}), runs=200)
    print(f"\n  latency, one message against 77 options: {fmt(lat)}")
    check(f"77-option decision p95 under {JEV_FAST_MS:.0f} ms", lat["p95"] < JEV_FAST_MS)
    big = Enum("Which code applies?", [f"{n} variant {k}" for k in range(1, 4) for n in names][:255])
    lat = latency(lambda: dm.decide(ds["text"][0], {"intent": big}), runs=100)
    print(f"  latency, one message against 255 options (Jev's stated maximum): {fmt(lat)}")
    check("255-option decision p95 under 150 ms, in one pass with no second stage", lat["p95"] < 150)
    finish()


if __name__ == "__main__":
    main()
