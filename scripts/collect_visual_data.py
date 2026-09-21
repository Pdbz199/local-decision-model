"""Runs the model on the demo inputs and saves real outputs and timings to assets/data.json.

scripts/make_visuals.py renders every image and GIF from that file, so nothing in the visuals is mocked.

Usage: uv run python scripts/collect_visual_data.py
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from decisionmodel import Bool, DecisionModel, Enum, Extract, Float, Int, MultiLabel

TICKETS = [
    "Hi, my order #48213 arrived with a cracked screen. I want a refund of $259.99 today or I will dispute the "
    "charge with my bank. You can reach me at dana.kim@example.org. Dana",
    "I've been locked out of my account since this morning and the password reset email never arrives. "
    "I have a client demo in an hour, please help.",
    "Hey team, just wanted to say the new dashboard is fantastic. Setup took two minutes. Thanks! Marcus",
]
SCHEMA = {
    "urgent": Bool("Does this need a response within the hour?"),
    "team": Enum("Which team should handle this ticket?", ["billing and refunds", "technical support", "sales", "customer success"]),
    "tags": MultiLabel("Which of these apply to the message?",
                       ["refund request", "damaged product", "login problem", "positive feedback", "pricing question"]),
    "anger": Int("How angry is the customer, from 1 (calm) to 5 (furious)?", 1, 5),
    "negativity": Float("How negative is this text, from 0 to 1?", 0.0, 1.0),
    "damaged": Bool("Did the product arrive damaged?"),
    "order_id": Extract("What is the order number?"),
    "email": Extract("What is the customer's email address?"),
}
SOURCE = ("Refund policy, updated March 2026. Customers may return unopened items within 30 days of delivery for a full "
          "refund. Opened items can be returned within 14 days for store credit only. Digital downloads and gift cards "
          "are not refundable. Refunds are issued to the original payment method within 5 to 7 business days. "
          "Return shipping is free for defective items; otherwise the customer pays return shipping.")
ANSWERS = [("You can return unopened items within 30 days for a full refund.", True),
           ("Opened items can be returned within 14 days, but only for store credit.", True),
           ("Gift cards can be refunded within 30 days.", False),
           ("Refunds usually arrive in 5 to 7 business days.", True),
           ("We cover return shipping on all returns.", False),
           ("Opened items get a full cash refund within 60 days.", False),
           ("If your item is defective, you do not pay for return shipping.", True),
           ("Refunds are paid out in cryptocurrency within one hour.", False)]


def p50(fn, runs=40, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    dm = DecisionModel.load()
    out = {}

    out["tickets"] = []
    for t in TICKETS:
        d = dm.decide(t, SCHEMA)
        fields = {}
        for k, spec in SCHEMA.items():
            fields[k] = {"type": type(spec).__name__, "value": d[k].value, "confidence": d[k].confidence, "expected": d[k].expected,
                         "span": d[k].span, "p_yes": d[k].p() if isinstance(spec, Bool) else None,
                         "probs": {str(a): b for a, b in d[k].probs.items()} if isinstance(spec, (Enum, MultiLabel)) else None}
        out["tickets"].append({"text": t, "fields": fields, "ms": p50(lambda: dm.decide(t, SCHEMA))})

    bank = load_dataset("mteb/banking77", split="test")
    names = sorted({l.replace("_", " ") for l in bank["label_text"]})
    spec = Enum("What is the customer's intent?", names)
    res = dm.decide_many(bank["text"], {"y": spec})
    gold = [l.replace("_", " ") for l in bank["label_text"]]
    conf = np.array([r.y.confidence for r in res])
    correct = np.array([r.y.value == g for r, g in zip(res, gold)])
    out["selective"] = [{"threshold": float(th), "automated": float((conf >= th).mean()), "accuracy": float(correct[conf >= th].mean())}
                        for th in np.arange(0.0, 0.96, 0.05)]
    picks = [i for i in (3, 40, 85, 600, 1210, 2000, 2500, 3000)]
    out["routing"] = {"options": names, "examples": [
        {"text": bank["text"][i], "gold": gold[i], "probs": [res[i].y.probs[n] for n in names],
         "ms": p50(lambda i=i: dm.decide(bank["text"][i], {"y": spec}), runs=25)} for i in picks]}

    big = Enum("Which code applies?", [f"{n} variant {k}" for k in range(1, 4) for n in names][:255])
    long_text = " ".join(load_dataset("stanfordnlp/imdb", split="test").select(range(40))["text"])
    ids = dm.encoder.tok(long_text, add_special_tokens=False)["input_ids"]
    two = {"y": Bool("Is the first review positive?"), "t": Enum("What kind of text is this?", ["movie reviews", "legal contract", "source code"])}
    doc2k, doc8k = dm.encoder.tok.decode(ids[:2048]), dm.encoder.tok.decode(ids[:8000])
    out["latency"] = [
        {"name": "One smart if (single Bool)", "ms": p50(lambda: dm.check(TICKETS[1], "Is the customer locked out of their account?"))},
        {"name": "Full ticket schema (8 typed fields)", "ms": p50(lambda: dm.decide(TICKETS[0], SCHEMA))},
        {"name": "Routing, 77 options", "ms": p50(lambda: dm.decide(bank["text"][0], {"y": spec}))},
        {"name": "Routing, 255 options", "ms": p50(lambda: dm.decide(bank["text"][0], {"y": big}))},
        {"name": "Guardrail: is the answer grounded?", "ms": p50(lambda: dm.check(SOURCE, f"Is this answer grounded in the source? Answer: {ANSWERS[0][0]}"))},
        {"name": "2,048 token document, 2 fields", "ms": p50(lambda: dm.decide(doc2k, two), runs=15)},
        {"name": "8,000 token document, 2 fields", "ms": p50(lambda: dm.decide(doc8k, two), runs=10, warmup=2)},
    ]

    out["grounding"] = [{"answer": a, "grounded": ok, "p": dm.check(SOURCE, f"Is this answer grounded in the source? Answer: {a}")} for a, ok in ANSWERS]
    out["source"] = SOURCE

    conf, correct = [], []
    imdb = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=1).select(range(1500))
    for r, y in zip(dm.decide_many(imdb["text"], {"y": Bool("Is this movie review positive?")}), imdb["label"]):
        conf.append(r.y.confidence); correct.append(r.y.value == bool(y))
    sms = load_dataset("ucirvine/sms_spam", split="train").shuffle(seed=1).select(range(1500))
    for r, y in zip(dm.decide_many(sms["sms"], {"y": Bool("Is this message spam?")}), sms["label"]):
        conf.append(r.y.confidence); correct.append(r.y.value == bool(y))
    fin = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    keep = [(t, y) for t, y in zip(fin["text"], fin["label"]) if y != 2][:1500]
    for r, (_, y) in zip(dm.decide_many([t for t, _ in keep], {"y": Enum("Is this news bearish or bullish for the stock?", ["bearish", "bullish"])}), keep):
        conf.append(r.y.confidence); correct.append(r.y.value == ["bearish", "bullish"][y])
    conf, correct = np.array(conf), np.array(correct, dtype=float)
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    out["calibration"] = [{"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())}
                          for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf >= lo) & (conf < hi)).sum()]

    Path("assets/data.json").write_text(json.dumps(out, indent=1))
    for t in out["tickets"]:
        print(round(t["ms"], 1), {k: (v["value"], round(v["confidence"], 2)) for k, v in t["fields"].items()})
    for e in out["routing"]["examples"]:
        top = sorted(zip(e["probs"], names), reverse=True)[:2]
        print(round(e["ms"], 1), e["text"][:60], "| gold:", e["gold"], "| top:", [(n, round(p, 2)) for p, n in top])
    print([(l["name"], round(l["ms"], 1)) for l in out["latency"]])


if __name__ == "__main__":
    main()
