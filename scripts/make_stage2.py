"""Builds the stage 2 "refresh" set: every multi-positive multi-label example, plus a replay sample of the rest.

Stage 1 trains on data/train.jsonl from scratch. Stage 2 continues from those weights for one short pass over this
smaller set, which upweights the multi-label tasks (about 4% of the full mixture) without forgetting the others.

Usage: uv run python scripts/make_stage2.py [--replay 0.15]
"""

import argparse
import json
import os
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--replay", type=float, default=0.15)
args = ap.parse_args()

rng = random.Random(0)
out_dir = Path(args.data) / "stage2"
out_dir.mkdir(exist_ok=True)
kept = boosted = 0
with open(Path(args.data) / "train.jsonl") as f, open(out_dir / "train.jsonl", "w") as out:
    for line in f:
        multi = json.loads(line)["src"].startswith("multi_")
        if multi or rng.random() < args.replay:
            out.write(line)
            kept += 1
            boosted += multi
for name in ("val.jsonl", "calib.jsonl"):
    link = out_dir / name
    if not link.exists():
        os.symlink(Path("..") / name, link)
print(f"stage 2 set: {kept} examples, {boosted} of them multi-label boosts")
