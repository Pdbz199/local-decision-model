"""Trains the local decision model on data/train.jsonl, then fits calibration temperatures on data/val.jsonl.

All three modes train with proper scoring rules (log loss), which reward honest
probabilities: cross entropy over options for "one", per-option binary cross
entropy for "any", and cross entropy over start/end positions for "span".

Usage: uv run python scripts/train.py [--backbone answerdotai/ModernBERT-base] [--epochs 1]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# Batch shapes vary a lot, which fragments the default CUDA allocator until the card is full.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from decisionmodel.encoding import MODES, Encoder, Query
from decisionmodel.model import DEFAULT_BACKBONE, DecisionNet


class JsonlDataset(Dataset):
    """Random access into a jsonl file through byte offsets, so worker processes share no big Python objects."""

    def __init__(self, path: str, tokenizer_name: str, max_len: int):
        self.path, self.tokenizer_name, self.max_len = path, tokenizer_name, max_len
        offsets, lengths, pos = [], [], 0
        with open(path, "rb") as f:
            for line in f:
                offsets.append(pos)
                lengths.append(len(line))
                pos += len(line)
        self.offsets = np.array(offsets, dtype=np.int64)
        self.lengths = np.array(lengths, dtype=np.int64)
        self._f = self._enc = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        if self._f is None:
            self._f = open(self.path, "rb")
            self._enc = Encoder(AutoTokenizer.from_pretrained(self.tokenizer_name), self.max_len)
        self._f.seek(self.offsets[i])
        r = json.loads(self._f.readline())
        enc = self._enc.encode(Query(r["mode"], r["q"], r["options"], r["state"]))
        start = end = 0
        valid = True
        if r["mode"] == "span" and r["span"][0] >= 0:
            s, e = r["span"]
            toks = [k for k, (a, b) in enumerate(enc.offsets) if b > s and a < e]
            if not toks or not enc.offsets or enc.offsets[-1][1] < e:
                valid = False  # the answer was truncated away
            else:
                start, end = enc.state_start + toks[0], enc.state_start + toks[-1]
        return enc, r["mode"], r["labels"], start, end, valid, r["src"]


def make_collate(encoder: Encoder):
    def collate(items):
        encs = [it[0] for it in items]
        batch = encoder.collate(encs)
        B, K = batch["opt_pos"].shape
        batch["mode"] = torch.tensor([MODES.index(it[1]) for it in items])
        multi = torch.zeros((B, K))
        target = torch.zeros(B, dtype=torch.long)
        for i, it in enumerate(items):
            if it[1] == "one":
                target[i] = it[2][0]
            for j in it[2]:
                multi[i, j] = 1.0
        batch["target"], batch["multi"] = target, multi
        batch["start"] = torch.tensor([it[3] for it in items])
        batch["end"] = torch.tensor([it[4] for it in items])
        batch["valid"] = torch.tensor([it[5] for it in items])
        batch["src"] = [it[6] for it in items]
        return batch
    return collate


def token_budget_batches(lengths: np.ndarray, budget: int, max_len_chars: int, seed: int, max_bs: int = 96):
    """Groups examples of similar length so little compute is wasted on padding."""
    rnd = random.Random(seed)
    idx = list(range(len(lengths)))
    rnd.shuffle(idx)
    batches = []
    for c in range(0, len(idx), 4096):
        chunk = sorted(idx[c:c + 4096], key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in chunk:
            est = min(lengths[i], max_len_chars) / 3.6 + 16  # rough chars -> tokens
            new_max = max(cur_max, est)
            if cur and (new_max * (len(cur) + 1) > budget or len(cur) >= max_bs):
                batches.append(cur)
                cur, new_max = [], est
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    rnd.shuffle(batches)
    return batches


def losses(model_out, batch):
    """Per-example loss vector (zeros for invalid rows)."""
    opt_logits, start_logits, end_logits = model_out
    mode, valid = batch["mode"], batch["valid"]
    loss = torch.zeros(len(mode), device=opt_logits.device)
    one, any_, span = mode == 0, mode == 1, mode == 2
    if one.any():
        loss[one] = F.cross_entropy(opt_logits[one], batch["target"][one], reduction="none")
    if any_.any():
        m = batch["opt_mask"][any_]
        lg = opt_logits[any_].masked_fill(~m, 0.0)
        bce = F.binary_cross_entropy_with_logits(lg, batch["multi"][any_], reduction="none") * m
        loss[any_] = bce.sum(1) / m.sum(1).clamp(min=1)
    if span.any():
        loss[span] = 0.5 * (F.cross_entropy(start_logits[span], batch["start"][span], reduction="none")
                            + F.cross_entropy(end_logits[span], batch["end"][span], reduction="none"))
    return loss * valid


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


FWD_KEYS = ("input_ids", "attention_mask", "opt_pos", "opt_mask", "span_mask")


@torch.no_grad()
def run_eval(model, loader, device, collect=False):
    model.eval()
    stats, store = {}, {m: [] for m in MODES}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**{k: batch[k] for k in FWD_KEYS})
        l = losses(out, batch)
        opt_logits, sl, el = out
        for i, src in enumerate(batch["src"]):
            if not batch["valid"][i]:
                continue
            m = MODES[batch["mode"][i]]
            if m == "one":
                correct = float(opt_logits[i].argmax() == batch["target"][i])
            elif m == "any":
                k = batch["opt_mask"][i]
                correct = float(((opt_logits[i][k] > 0).float() == batch["multi"][i][k]).float().mean())
            else:
                correct = float(sl[i].argmax() == batch["start"][i] and el[i].argmax() == batch["end"][i])
            s = stats.setdefault(src, [0, 0.0, 0.0])
            s[0] += 1
            s[1] += correct
            s[2] += float(l[i])
            if collect:
                if m == "one":
                    store[m].append((opt_logits[i][batch["opt_mask"][i]].cpu(), int(batch["target"][i])))
                elif m == "any":
                    k = batch["opt_mask"][i]
                    store[m].append((opt_logits[i][k].cpu(), batch["multi"][i][k].cpu()))
                else:
                    sm = batch["span_mask"][i]
                    # remap absolute positions to indices within the masked (valid) positions
                    pos = torch.cumsum(sm.long(), 0) - 1
                    store[m].append((sl[i][sm].cpu(), el[i][sm].cpu(), int(pos[batch["start"][i]]), int(pos[batch["end"][i]])))
    model.train()
    return stats, store


def _nll(m, rows, T_of):
    if m == "one":
        return sum(float(F.cross_entropy(lg[None] / T_of(len(lg)), torch.tensor([y]))) for lg, y in rows) / len(rows)
    if m == "any":
        return sum(float(F.binary_cross_entropy_with_logits(lg / T_of(0), y, reduction="mean")) for lg, y in rows) / len(rows)
    return sum(float(F.cross_entropy(s[None] / T_of(0), torch.tensor([a])) + F.cross_entropy(e[None] / T_of(0), torch.tensor([b])))
               for s, e, a, b in rows) / len(rows)


def fit_temperatures(store, unseen_one=None) -> tuple[dict[str, float], float]:
    """Grid-searches calibration parameters that minimize held-out log loss.

    "any" and "span" get one temperature each, fit on validation data. "one" is fit on `unseen_one`
    when given: examples from tasks that were never trained on, because that is how the model is
    used in practice. There the temperature is a + b * ln(number of options).
    """
    grid = np.exp(np.linspace(math.log(0.5), math.log(4.0), 61))
    temps, slope = {}, 0.0
    for m in MODES:
        rows = unseen_one if (m == "one" and unseen_one) else store[m]
        if not rows:
            temps[m] = 1.0
            continue
        before = _nll(m, rows, lambda k: 1.0)
        if m == "one" and unseen_one:
            best = min((_nll(m, rows, lambda k, a=a, b=b: a + b * math.log(max(k, 2))), float(a), float(b))
                       for a in np.linspace(0.6, 3.0, 25) for b in np.linspace(0.0, 0.8, 17))
            temps[m], slope = best[1], best[2]
            print(f"  temperature[one] = {best[1]:.2f} + {best[2]:.2f} * ln(options)  (unseen-task log loss {before:.4f} -> {best[0]:.4f})")
        else:
            best = min((_nll(m, rows, lambda k, T=T: T), float(T)) for T in grid)
            temps[m] = best[1]
            print(f"  temperature[{m}] = {best[1]:.3f}  (val log loss {before:.4f} -> {best[0]:.4f})")
    return temps, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints/decision-model")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--token-budget", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for smoke tests)")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--resume-from", default="", help="checkpoint directory saved by an interrupted run")
    ap.add_argument("--init-from", default="", help="start a fresh schedule from the weights of an existing checkpoint")
    ap.add_argument("--micro-tokens", type=int, default=11000, help="padded batches larger than this are split in two")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(args.backbone)
    encoder = Encoder(tok, args.max_len)
    collate = make_collate(encoder)

    val_ds = JsonlDataset(f"{args.data}/val.jsonl", args.backbone, args.max_len)
    val_batches = token_budget_batches(val_ds.lengths, args.token_budget * 2, args.max_len * 4, seed=1)
    val_loader = DataLoader(val_ds, batch_sampler=val_batches, collate_fn=collate, num_workers=2)

    if args.calibrate_only:
        model, tok, meta = DecisionNet.load(args.out, device=device, dtype=torch.float32)
    else:
        start_step = 0
        if args.resume_from:  # weights only; the batch order is deterministic, so we skip what was already seen
            model, _, old = DecisionNet.load(args.resume_from, device=device, dtype=torch.float32)
            start_step = int(old["step"])
            print(f"resuming from {args.resume_from} at step {start_step}")
        elif args.init_from:
            model, _, _ = DecisionNet.load(args.init_from, device=device, dtype=torch.float32)
            print(f"continuing from the weights in {args.init_from}")
        else:
            model = DecisionNet.from_backbone(args.backbone).to(device)
        meta = {}
        train_ds = JsonlDataset(f"{args.data}/train.jsonl", args.backbone, args.max_len)
        batches = []
        for ep in range(math.ceil(args.epochs)):
            b = token_budget_batches(train_ds.lengths, args.token_budget, args.max_len * 4, seed=ep)
            batches += b[: int(len(b) * min(1.0, args.epochs - ep))]
        if args.max_steps:
            batches = batches[: args.max_steps]
        total = len(batches)
        batches = batches[start_step:]
        print(f"{len(train_ds)} examples, {total} steps")
        loader = DataLoader(train_ds, batch_sampler=batches, collate_fn=collate, num_workers=args.workers,
                            pin_memory=True, prefetch_factor=4, persistent_workers=False)

        head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        decay = [p for n, p in model.encoder.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in model.encoder.named_parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([{"params": decay, "lr": args.lr, "weight_decay": 0.01},
                                 {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
                                 {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}], betas=(0.9, 0.98), eps=1e-6, fused=True)
        base_lrs = [g["lr"] for g in opt.param_groups]
        warmup = max(50, int(0.04 * total))

        def lr_scale(step):
            if step < warmup:
                return step / warmup
            if step - start_step < 100 and start_step:  # the optimizer state is fresh after a resume
                return (step - start_step + 1) / 100 * lr_scale(step + 100)
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

        model.train()
        t0, run_loss, run_n, n_tok = time.time(), 0.0, 0, 0
        for step, batch in enumerate(loader, start=start_step):
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_scale(step)
            batch = to_device(batch, device)
            n_valid = batch["valid"].sum().clamp(min=1)
            B = batch["input_ids"].size(0)
            # Bound peak memory: an unusually large padded batch is processed as two micro-batches.
            cuts = [0, B // 2, B] if batch["input_ids"].numel() > args.micro_tokens and B > 1 else [0, B]
            loss = 0.0
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                mb = {k: v[lo:hi] for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**{k: mb[k] for k in FWD_KEYS})
                part = losses(out, mb).sum() / n_valid
                part.backward()
                loss += float(part)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss
            run_n += 1
            n_tok += int(batch["attention_mask"].sum())
            if (step + 1) % 100 == 0:
                el = time.time() - t0
                print(f"step {step + 1}/{total} loss {run_loss / run_n:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{n_tok / el:,.0f} tok/s  eta {(total - step - 1) * el / (step + 1 - start_step) / 60:.0f} min  "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G/{torch.cuda.memory_reserved() / 2**30:.1f}G", flush=True)
                run_loss, run_n = 0.0, 0
            if (step + 1) % args.eval_every == 0 and step + 1 < total:
                stats, _ = run_eval(model, val_loader, device)
                n = sum(s[0] for s in stats.values())
                print(f"  [val] acc {sum(s[1] for s in stats.values()) / n:.4f} loss {sum(s[2] for s in stats.values()) / n:.4f}", flush=True)
                model.save(args.out, tok, {"backbone": args.backbone, "step": step + 1, "temperature": {m: 1.0 for m in MODES}})
        meta = {"backbone": args.backbone, "steps": total, "init_from": args.init_from or None, "epochs": args.epochs, "lr": args.lr, "train_examples": len(train_ds), "train_max_len": args.max_len,
                "train_minutes": round((time.time() - t0) / 60, 1)}

    print("final validation:")
    stats, store = run_eval(model, val_loader, device, collect=True)
    for src, s in sorted(stats.items()):
        print(f"  {src:20s} n={s[0]:4d} acc={s[1] / s[0]:.3f} loss={s[2] / s[0]:.3f}")
    n = sum(s[0] for s in stats.values())
    print(f"  overall acc {sum(s[1] for s in stats.values()) / n:.4f}")
    unseen = None
    if Path(f"{args.data}/calib.jsonl").exists():
        cal_ds = JsonlDataset(f"{args.data}/calib.jsonl", args.backbone, args.max_len)
        cal_loader = DataLoader(cal_ds, batch_sampler=token_budget_batches(cal_ds.lengths, args.token_budget, args.max_len * 4, seed=2),
                                collate_fn=collate, num_workers=2)
        cal_stats, cal_store = run_eval(model, cal_loader, device, collect=True)
        print("unseen calibration tasks (zero-shot):")
        for src, st in sorted(cal_stats.items()):
            print(f"  {src:40s} n={st[0]:4d} acc={st[1] / st[0]:.3f}")
        unseen = cal_store["one"]
        meta["calibration_task_accuracy"] = {src: round(st[1] / st[0], 4) for src, st in sorted(cal_stats.items())}
    temps, slope = fit_temperatures(store, unseen)
    model.temperature.copy_(torch.tensor([temps[m] for m in MODES]))
    model.k_slope.fill_(slope)
    meta["temperature"], meta["temperature_k_slope"] = temps, slope
    meta["val_accuracy"] = {src: round(s[1] / s[0], 4) for src, s in sorted(stats.items())}
    model.save(args.out, tok, meta)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
