"""Turns (mode, question, options, state) queries into model inputs.

Sequence layout:

    [CLS] [MODE] question ([OPT] option)* [SEP] state [SEP]

The schema side comes first so that truncating a long state never drops an
option. Control tokens are inserted by id, never parsed out of text, so a state
that contains the literal string "[OPT]" cannot forge an option.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ModernBERT ships reserved, never-trained vocabulary slots; we claim four.
CONTROL_TOKENS = {"opt": "[unused0]", "one": "[unused1]", "any": "[unused2]", "span": "[unused3]"}
MODES = ("one", "any", "span")


@dataclass
class Query:
    mode: str
    question: str
    options: list[str]
    state: str


@dataclass
class Encoded:
    input_ids: list[int]
    opt_pos: list[int]  # index of each option's [OPT] token
    state_start: int  # index of the first state token
    offsets: list[tuple[int, int]]  # char offsets of each (kept) state token


class Encoder:
    def __init__(self, tokenizer, max_len: int = 8192):
        self.tok = tokenizer
        self.max_len = max_len
        self.cls, self.sep, self.pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        self.ctrl = {k: tokenizer.convert_tokens_to_ids(v) for k, v in CONTROL_TOKENS.items()}
        assert all(i is not None and i != tokenizer.unk_token_id for i in self.ctrl.values()), "control tokens missing"
        # The tokenizer would happily parse "[SEP]" or "[unused0]" out of user text. Every reserved id that
        # shows up in user-provided text is replaced with [UNK], so text can never forge structure.
        reserved = set(tokenizer.all_special_ids) | set(self.ctrl.values())
        reserved |= {i for t, i in tokenizer.get_added_vocab().items()
                     if t.startswith("[") and t.endswith("]")}  # padding-only added tokens (runs of spaces) stay usable
        self._reserved, self._unk = reserved, tokenizer.unk_token_id
        self._opt_cache: dict[str, list[int]] = {}

    def _clean(self, ids: list[int]) -> list[int]:
        return [self._unk if i in self._reserved else i for i in ids]

    def _ids(self, text: str) -> list[int]:
        return self._clean(self.tok(text, add_special_tokens=False)["input_ids"])

    def _opt_ids(self, text: str) -> list[int]:
        ids = self._opt_cache.get(text)
        if ids is None:
            ids = self._ids(" " + text.strip())[:24]
            if len(self._opt_cache) < 100_000:
                self._opt_cache[text] = ids
        return ids

    def tokenize_state(self, state: str) -> tuple[list[int], list[tuple[int, int]]]:
        st = self.tok(state, add_special_tokens=False, return_offsets_mapping=True)
        return self._clean(st["input_ids"]), st["offset_mapping"]

    def encode(self, q: Query, max_len: int | None = None, state_tok=None) -> Encoded:
        """`state_tok` lets callers tokenize a state once and reuse it across fields."""
        max_len = max_len or self.max_len
        ids = [self.cls, self.ctrl[q.mode]] + self._ids(q.question.strip())[:256]
        opt_pos = []
        for o in q.options:
            opt_pos.append(len(ids))
            ids.append(self.ctrl["opt"])
            ids.extend(self._opt_ids(o))
        ids.append(self.sep)
        budget = max_len - len(ids) - 1
        if budget < 8:
            raise ValueError("question and options leave no room for the state; raise max_len or use fewer options")
        all_ids, all_offsets = state_tok if state_tok is not None else self.tokenize_state(q.state)
        state_ids, offsets = all_ids[:budget], all_offsets[:budget]
        state_start = len(ids)
        ids = ids + state_ids + [self.sep]
        return Encoded(ids, opt_pos, state_start, [tuple(o) for o in offsets])

    def collate(self, encs: list[Encoded], device=None) -> dict[str, torch.Tensor]:
        B, L = len(encs), max(len(e.input_ids) for e in encs)
        K = max(1, max(len(e.opt_pos) for e in encs))
        input_ids = torch.full((B, L), self.pad, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        opt_pos = torch.zeros((B, K), dtype=torch.long)
        opt_mask = torch.zeros((B, K), dtype=torch.bool)
        span_mask = torch.zeros((B, L), dtype=torch.bool)
        for i, e in enumerate(encs):
            n = len(e.input_ids)
            input_ids[i, :n] = torch.tensor(e.input_ids)
            attn[i, :n] = 1
            k = len(e.opt_pos)
            if k:
                opt_pos[i, :k] = torch.tensor(e.opt_pos)
                opt_mask[i, :k] = True
            # A span may start/end on any state token; position 0 ([CLS]) means "no answer".
            span_mask[i, 0] = True
            span_mask[i, e.state_start : e.state_start + len(e.offsets)] = True
        batch = dict(input_ids=input_ids, attention_mask=attn, opt_pos=opt_pos, opt_mask=opt_mask, span_mask=span_mask)
        if device is not None:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return batch
