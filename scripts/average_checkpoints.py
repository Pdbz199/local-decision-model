"""Averages the weights of checkpoints that share a training lineage (a "model soup").

Usage: uv run python scripts/average_checkpoints.py --out checkpoints/soup checkpoints/a checkpoints/b
Afterwards refit calibration: uv run python scripts/train.py --calibrate-only --out checkpoints/soup
"""

import argparse

import torch

from decisionmodel.model import DecisionNet

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="+")
ap.add_argument("--out", required=True)
args = ap.parse_args()

models = [DecisionNet.load(p, device="cpu", dtype=torch.float32) for p in args.checkpoints]
base, tok, meta = models[0]
states = [m.state_dict() for m, _, _ in models]
avg = {k: (sum(s[k].float() for s in states) / len(states)).to(v.dtype) if v.is_floating_point() else v for k, v in states[0].items()}
base.load_state_dict(avg)
base.save(args.out, tok, {"backbone": meta.get("backbone"), "averaged_from": args.checkpoints, "temperature": meta.get("temperature")})
print(f"averaged {len(models)} checkpoints into {args.out}")
