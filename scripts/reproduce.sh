#!/usr/bin/env bash
# Rebuilds everything in this repo from scratch: data, model, evaluation, experiments, visuals.
# Needs Linux or WSL2, an NVIDIA GPU with 16 GB of memory, and uv (https://docs.astral.sh/uv/).
# Total time on an RTX 5070 Ti is about two hours. See REPRODUCING.md for what each step does.
#
# Usage: bash scripts/reproduce.sh            (run from the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/experiments

echo "== 1/7 environment";          uv sync
echo "== 2/7 training data";        uv run python scripts/build_data.py            2>&1 | tee logs/build_data.log
echo "== 3/7 stage 1 training";     uv run python scripts/train.py --out checkpoints/decision-model-stage1 2>&1 | tee logs/train_stage1.log
echo "== 4/7 stage 2 and averaging"
uv run python scripts/make_stage2.py
uv run python scripts/train.py --data data/stage2 --init-from checkpoints/decision-model-stage1 \
    --lr 2e-5 --head-lr 1e-4 --out checkpoints/decision-model-stage2                2>&1 | tee logs/train_stage2.log
uv run python scripts/average_checkpoints.py --out checkpoints/decision-model \
    checkpoints/decision-model-stage1 checkpoints/decision-model-stage2
uv run python scripts/train.py --calibrate-only --out checkpoints/decision-model   2>&1 | tee logs/calibrate.log
echo "== 5/7 evaluation"
uv run pytest -q
uv run python scripts/evaluate.py
uv run python scripts/eval_multilabel.py | tee results/eval_multilabel.txt
echo "== 6/7 experiments"
for f in experiments/0*.py; do
    (cd experiments && uv run python "$(basename "$f")") 2>&1 | tee "results/experiments/$(basename "${f%.py}").txt"
done
echo "== 7/7 visuals"
uv run python scripts/collect_visual_data.py
uv run python scripts/make_visuals.py
echo "done: model in checkpoints/decision-model, numbers in results/, images in assets/"
