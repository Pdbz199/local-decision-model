"""Uploads the trained checkpoint and a model card to the HuggingFace Hub.

Log in first with a token that has write access:  uv run hf auth login

Usage: uv run python scripts/publish_to_hub.py <user>/local-decision-model [--private] [--dry-run]
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

GITHUB = "https://github.com/Pdbz199/local-decision-model"

CARD = """---
license: {license}
language: en
library_name: pytorch
base_model: answerdotai/ModernBERT-base
pipeline_tag: zero-shot-classification
tags: [decision-model, zero-shot-classification, structured-output, guardrails, extraction, modernbert]
---

# local-decision-model

A small, fast decision model: **unstructured state in, typed probabilistic decisions out.**
One forward pass of a 150M parameter encoder answers a whole schema of typed questions about a piece of text,
typically in 10 to 25 ms on a consumer GPU. It never generates text, so every output is guaranteed to match its declared type.

Code, training recipe, experiments and full results: {github}

![demo]({github}/raw/main/assets/hero.gif)

## Usage

These weights need the small `decisionmodel` package from the GitHub repo (they are not a standard `transformers` head).

```bash
pip install git+{github}
```

```python
from decisionmodel import DecisionModel, Bool, Enum, MultiLabel, Int, Extract

dm = DecisionModel.load("{repo_id}")

d = dm.decide("My order #48213 arrived with a cracked screen. Refund me today or I dispute the charge.", {{
    "urgent":   Bool("Does this need a response within the hour?"),
    "team":     Enum("Which team should handle this ticket?", ["billing and refunds", "technical support", "sales"]),
    "anger":    Int("How angry is the customer, from 1 (calm) to 5 (furious)?", 1, 5),
    "order_id": Extract("What is the order number?"),
}})
print(d.team.value, d.team.confidence, d.urgent.p(), d.order_id.value)
```

Types: `Bool`, `Enum` (2 to 255 options), `MultiLabel`, `Int`, `Float`, `Extract` (a verbatim substring of the input, or `None`).

## How it works

ModernBERT-base with two small heads. The schema is written into the input as
`[CLS] [MODE] question [OPT] option 1 [OPT] option 2 ... [SEP] state [SEP]`. The hidden state at each `[OPT]` token is scored,
so all options are judged in the same pass; a span head handles extraction. Trained with log loss on about one million
examples recast from roughly 50 public datasets, then temperature-calibrated on tasks that were never trained on.

## Results on tasks never seen in training

| Held-out task | Type | Result |
| --- | --- | --- |
| IMDB sentiment | Bool | 94.4% accuracy, calibration error 0.016 |
| SQuAD v2 dev | Extract | 82.8% exact match, F1 85.8 |
| AG News topics | Enum (4) | 74.0% |
| Twitter financial news sentiment | Enum (3) | 74.1% |
| Banking77 intents | Enum (77) | 63.0% top 1, 80.2% top 3; 83.6% on the 48% of calls with confidence of at least 0.9 |
| deepset prompt injections | Bool | 69.6% accuracy, AUROC 0.87 |
| SMS spam | Bool | 64.9% with a vague question, 89.4% with a precise one (AUROC 0.94 to 0.96) |

Latency (RTX 5070 Ti, bfloat16, median): single yes/no 10 ms, 8 typed fields 16 ms, 77 options 11 ms, 255 options 21 ms, 8,000 token input 351 ms.

## Limitations

* A small model: good at judgments a person could make at a glance, weak at implied or conditional meaning.
* Sensitive to question wording. Test a schema on a few real examples before trusting it.
* `MultiLabel` under-predicts when several options are true at once. Ask important tags as their own `Bool`.
* Confidence is well calibrated on familiar kinds of task and can be overconfident on unfamiliar ones.
* English only. Guardrail decisions can be wrong: do not use it as a sole security boundary.

## Licensing

The code is MIT licensed. The base model is Apache 2.0. These weights were trained on public datasets with mixed
licenses, some of which restrict commercial use (for example lmsys/toxic-chat, CC BY-NC 4.0, and the Yelp reviews
dataset). The weights are therefore released under {license}. If you need different terms, retrain with
`scripts/build_data.py` after removing the sources that do not fit your use; every source is listed in that file.

## Relation to Jev

This project was inspired by TypeSafe AI's blog post introducing "System One Models" and their model Jev.
It is an independent, generic implementation built only from that public post, and is not affiliated with TypeSafe AI.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="for example yourname/local-decision-model")
    ap.add_argument("--checkpoint", default="checkpoints/decision-model")
    ap.add_argument("--weights-license", default="cc-by-nc-4.0")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the upload folder and print its contents, upload nothing")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    assert (src / "heads.safetensors").exists(), "expected a safetensors checkpoint (no pickle files are uploaded)"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "upload"
        shutil.copytree(src, out, ignore=shutil.ignore_patterns("*.pt", "*.bin"))
        meta = json.loads((out / "meta.json").read_text())
        meta.pop("averaged_from", None)  # local paths mean nothing to a downloader
        meta["source"] = GITHUB
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        (out / "README.md").write_text(CARD.format(license=args.weights_license, github=GITHUB, repo_id=args.repo_id))
        files = sorted(p for p in out.rglob("*") if p.is_file())
        for p in files:
            print(f"  {p.stat().st_size / 1e6:8.2f} MB  {p.relative_to(out)}")
        if args.dry_run:
            print("dry run: nothing uploaded")
            return
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=str(out), commit_message="Upload local-decision-model")
        print(f"published: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
