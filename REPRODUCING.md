# Reproducing this repo from scratch

Everything here (training data, model, evaluation numbers, experiment outputs, images) can be rebuilt with one command:

```bash
bash scripts/reproduce.sh
```

It takes about two hours on an RTX 5070 Ti. If you only want to use the model, skip all of this: the trained weights are
published at https://huggingface.co/Pdbz199/local-decision-model and `DecisionModel.load("Pdbz199/local-decision-model")` fetches them. The sections below explain each step so you can run them one at a time.

## Requirements

* Linux or WSL2, with an NVIDIA GPU that has 16 GB of memory (training peaks near 10 GB).
* [uv](https://docs.astral.sh/uv/). Nothing else needs to be installed by hand.
* About 5 GB of free disk: 1 GB of training data, under 2 GB of checkpoints, plus the HuggingFace dataset cache.
* Internet access to download the public datasets and the ModernBERT-base weights. No HuggingFace account is needed,
  though setting `HF_TOKEN` avoids anonymous rate limits.

`pyproject.toml` tells uv to use its own managed Python builds. Those ship C headers, which PyTorch needs to compile
its Triton GPU kernels. Many system Pythons do not, and fail with `Python.h: No such file or directory`.

## Step by step

| # | Command | Time | Output |
| --- | --- | --- | --- |
| 1 | `uv sync` | 2 min | `.venv/` with PyTorch (CUDA), transformers, datasets |
| 2 | `uv run python scripts/build_data.py` | 15 min | `data/train.jsonl` (about 1.07M examples), `data/val.jsonl`, `data/calib.jsonl` |
| 3 | `uv run python scripts/train.py --out checkpoints/decision-model-stage1` | 70 min | stage 1 checkpoint |
| 4a | `uv run python scripts/make_stage2.py` | 1 min | `data/stage2/` (all multi-label examples plus a 15% replay sample) |
| 4b | `uv run python scripts/train.py --data data/stage2 --init-from checkpoints/decision-model-stage1 --lr 2e-5 --head-lr 1e-4 --out checkpoints/decision-model-stage2` | 10 min | stage 2 checkpoint |
| 4c | `uv run python scripts/average_checkpoints.py --out checkpoints/decision-model checkpoints/decision-model-stage1 checkpoints/decision-model-stage2` | 1 min | the averaged model |
| 4d | `uv run python scripts/train.py --calibrate-only --out checkpoints/decision-model` | 2 min | calibration temperatures, written into `checkpoints/decision-model/meta.json` |
| 5 | `uv run pytest`, `uv run python scripts/evaluate.py`, `uv run python scripts/eval_multilabel.py` | 5 min | `results/eval.json` and the multi-label numbers |
| 6 | `cd experiments && for f in 0*.py; do uv run python $f; done` | 5 min | five experiments, each ending in pass/fail checks |
| 7 | `uv run python scripts/collect_visual_data.py && uv run python scripts/make_visuals.py` | 3 min | `assets/data.json`, then every image and GIF in `assets/` |

Stage 1 alone already gives a usable model: pass `--out checkpoints/decision-model` in step 3 and skip step 4.

## If something goes wrong

* **Training is interrupted.** Checkpoints are saved every 3,000 steps. Continue with
  `uv run python scripts/train.py --resume-from <checkpoint dir> --out <same dir>`. The batch order is deterministic,
  so the run continues with exactly the data it had not seen yet.
* **GPU runs out of memory, or training suddenly slows to a crawl.** On WSL2 a full GPU spills into system RAM instead
  of failing. Lower `--token-budget` (default 8000) or `--micro-tokens` (default 11000, the size above which a batch is
  split in two).
* **A dataset fails to download.** `build_data.py` logs `FAILED` for that source and continues without it.
* **You moved or renamed the repo directory.** Run `uv sync` again. The virtual environment stores absolute paths.
* **The model loads from the wrong place.** `DecisionModel.load()` uses its argument, then the `DECISION_MODEL_PATH`
  environment variable, then `checkpoints/decision-model` relative to the repo root.

## What will differ from the published numbers

* The shipped stage 1 checkpoint was trained before the three multi-positive multi-label tasks were added to the
  mixture (1.02M examples instead of 1.07M). A fresh run includes them in stage 1 as well.
* GPU kernels are not bit-for-bit deterministic, so expect accuracy to move by a few tenths of a point, and sensitive
  prompts (see "SMS spam" in the README) by more.
* Latency depends on your GPU, CPU and whatever else is using them.

## Larger model

To trade speed for quality, train a larger encoder. This is untested here and needs a smaller batch budget:

```bash
uv run python scripts/train.py --backbone answerdotai/ModernBERT-large --token-budget 3500 --out checkpoints/decision-model-large
```
