# Changelog

## Unreleased

* Experiment 6: the model plays Snake zero-shot by reading one English sentence per candidate move
  (`experiments/06_snake.py`, with `--watch` for a live terminal game and `--record` for the README GIF).
  Demo idea inspired by [laya-mlx](https://github.com/mizorewww/laya-mlx).

## v0.1.0 (2026-09-21)

First public release.

* `decisionmodel` package: `DecisionModel.decide`, `decide_many` and `check`, with the typed specs `Bool`, `Enum`
  (2 to 255 options), `MultiLabel`, `Int`, `Float` and `Extract`. Every result carries a typed value, a confidence,
  and the full probability distribution.
* Trained weights on the HuggingFace Hub: https://huggingface.co/Pdbz199/local-decision-model
  (ModernBERT-base plus two heads, 150M parameters, safetensors only). `DecisionModel.load("Pdbz199/local-decision-model")`
  downloads and caches them.
* Full training recipe from public datasets, with held-out decontamination, a resumable trainer, calibration on tasks
  that were never trained on, and a one-command rebuild (`scripts/reproduce.sh`, see `REPRODUCING.md`).
* Zero-shot results on tasks never seen in training: IMDB 94.4%, SQuAD v2 F1 85.8, AG News 74.0%, Banking77 63.0%
  top 1 across 77 options, financial news sentiment 74.1%. Latency on an RTX 5070 Ti: 10 ms for one yes/no question,
  16 ms for 8 typed fields, 21 ms for 255 options. On CPU: about 48 ms for one yes/no question.
* Five runnable experiments with pass/fail checks, unit tests, and README visuals rendered from real model outputs.

Known limitations are listed in the README: weak on implied or conditional meaning, sensitive to question wording,
`MultiLabel` under-predicts when several options are true, English only.

Code is MIT licensed. The published weights are CC BY-NC 4.0 because of the licenses of some training datasets.
