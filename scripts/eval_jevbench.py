"""Runs the model on the public items of JevBench, a third-party benchmark for Jev-class decision models.

JevBench (https://github.com/fstandhartinger/jevbench, MIT) publishes 231 of its decisions: 48 easy, 72
standard and 111 hard. Its judge tier and its 308 sealed items are private, so this script measures accuracy and
calibration on the public items only. It cannot produce an official JevBench Score or rank; those come from the
benchmark's own runs (https://benchmarkheaven.com/jev-models).

The files are downloaded from a pinned commit and checked against the SHA-256 values in JevBench's manifest, so the
numbers are reproducible. Two mechanical mappings of each JevBench question onto our types are run:

  plain      the question text as written; choice options are the raw labels, score options are "level N"
  criteria   the per-option criteria JevBench provides are appended to the option texts (and, for yes/no
             questions, to the question), because that is the information a Jev user would pass along too

Nothing is tuned on these items, and they must never be used for training: JevBench penalises models whose
public accuracy runs far ahead of their sealed accuracy.

Usage: uv run python scripts/eval_jevbench.py [--model checkpoints/decision-model] [--out results/eval_jevbench.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from decisionmodel import Bool, DecisionModel, Enum
from decisionmodel.types import Spec

JEVBENCH_COMMIT = "2fa63fa3226cb369795525ed011800f57dcbd894"  # main on 2026-09-23
RAW = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEVBENCH_COMMIT}/datasets/public/"
TIERS = {  # tier name -> (file, sha256 from datasets/manifest.json)
    "easy": ("easy.jsonl", "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "standard": ("original.jsonl", "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "hard": ("hard.jsonl", "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
VARIANTS = ("plain", "criteria")


def to_spec(question: dict, labels: list[str], variant: str) -> tuple[Spec, list[str]]:
    """Maps one JevBench question onto a type spec.

    Returns the spec and the list of option texts in label order, so a decoded value can be mapped back to a
    JevBench label by position. JevBench question types: "noul" is yes/no, "choice" picks one label (criteria is a
    label -> description dict), "score" picks an ordinal level (labels are level indices, criteria is a list of
    level descriptions).
    """
    kind, text, crit = question["type"], question["instructions"], question.get("criteria")
    if kind == "noul":
        if variant == "criteria" and isinstance(crit, dict):
            text = f"{text} Yes means: {crit.get('true', '')} No means: {crit.get('false', '')}"
        return Bool(text), ["no", "yes"]
    if kind == "choice":
        if variant == "criteria" and isinstance(crit, dict):
            opts = [f"{label}: {crit[label]}" if crit.get(label) else label for label in labels]
        else:
            opts = list(labels)
        return Enum(text, opts), opts
    if kind == "score":
        if variant == "criteria" and isinstance(crit, list) and len(crit) == len(labels):
            opts = [f"level {label}: {desc}" for label, desc in zip(labels, crit)]
        else:
            opts = [f"level {label}" for label in labels]
        return Enum(text, opts), opts
    raise ValueError(f"unknown JevBench question type {kind!r}")


def prediction(decision, question: dict, labels: list[str], opts: list[str]) -> tuple[str, float]:
    """The predicted JevBench label and the confidence in it."""
    if question["type"] == "noul":
        p = decision.p()
        return ("yes" if decision.value else "no"), max(p, 1 - p)
    return labels[opts.index(decision.value)], float(decision.confidence)


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fetch(tier: str, cache: Path) -> list[dict]:
    name, digest = TIERS[tier]
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {RAW}{name}")
        urllib.request.urlretrieve(RAW + name, path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise RuntimeError(f"{path} has sha256 {got}, expected {digest}; delete it and retry")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_variant(dm, items: dict[str, list[dict]], variant: str) -> dict:
    out, all_conf, all_ok, latency = {}, [], [], []
    for tier, rows in items.items():
        conf, ok, fam, tier_latency = [], [], defaultdict(lambda: [0, 0]), []
        for r in rows:
            spec, opts = to_spec(r["question"], r["labels"], variant)
            t0 = time.perf_counter()
            d = dm.decide(r["state"], {"decision": spec})["decision"]
            tier_latency.append(time.perf_counter() - t0)
            pred, c = prediction(d, r["question"], r["labels"], opts)
            expected = str(r["expected"])  # score answers are integer level indices
            hit = pred == expected
            conf.append(c), ok.append(hit)
            fam[r["family"]][0] += hit
            fam[r["family"]][1] += 1
        out[tier] = {
            "n": len(rows),
            "accuracy": float(np.mean(ok)),
            "chance": float(np.mean([1 / len(r["labels"]) for r in rows])),
            "ece": ece(conf, ok),
            "mean_confidence": float(np.mean(conf)),
            "by_family": {k: {"correct": v[0], "n": v[1]} for k, v in sorted(fam.items())},
            "serial_latency_ms": {"p50": float(np.percentile(tier_latency, 50) * 1000),
                                  "p95": float(np.percentile(tier_latency, 95) * 1000)},
        }
        all_conf += conf
        all_ok += ok
        latency += tier_latency
    lat = np.array(latency) * 1000
    out["all_public"] = {"n": len(all_ok), "accuracy": float(np.mean(all_ok)), "ece": ece(all_conf, all_ok)}
    out["serial_latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                "max": float(lat.max())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="checkpoint directory or Hub id (default: DecisionModel.load's default)")
    ap.add_argument("--cache", default="data/jevbench", help="where the downloaded public items are kept")
    ap.add_argument("--out", default="results/eval_jevbench.json")
    args = ap.parse_args()

    items = {tier: fetch(tier, Path(args.cache)) for tier in TIERS}
    dm = DecisionModel.load(args.model)
    results = {
        "benchmark": "JevBench public items (easy, standard, hard); the judge tier and sealed items are private",
        "source": f"https://github.com/fstandhartinger/jevbench at {JEVBENCH_COMMIT}",
        "files": {tier: {"file": f, "sha256": h} for tier, (f, h) in TIERS.items()},
        "note": "Self-measured on the public items only. Not an official JevBench Score or rank.",
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": dm.device,
        "platform": platform.platform(),
        "variants": {v: run_variant(dm, items, v) for v in VARIANTS},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n| Tier | n | chance |" + "".join(f" {v}: accuracy | {v}: ECE |" for v in VARIANTS))
    print("| --- | --- | --- |" + " --- | --- |" * len(VARIANTS))
    for tier in TIERS:
        base = results["variants"][VARIANTS[0]][tier]
        row = f"| {tier} | {base['n']} | {base['chance']:.1%} |"
        for v in VARIANTS:
            t = results["variants"][v][tier]
            row += f" {t['accuracy']:.1%} | {t['ece']:.3f} |"
        print(row)
    for v in VARIANTS:
        a, lat = results["variants"][v]["all_public"], results["variants"][v]["serial_latency_ms"]
        print(f"{v}: all public accuracy {a['accuracy']:.1%}, ECE {a['ece']:.3f}, "
              f"serial latency p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
