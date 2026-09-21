"""The network: a bidirectional encoder plus two small decision heads.

One forward pass reads the whole query (question, every option, the state) and
emits a logit for every option at once, plus start/end logits over the state
for extraction. There is no decoding loop and no token generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .encoding import MODES

DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"


class DecisionNet(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.opt_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.span_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        # Post-hoc calibration, fit after training (see scripts/train.py). Logits are divided by
        # temperature[mode], plus k_slope * ln(number of options) for "one" queries: many-way choices on
        # unfamiliar tasks tend to be more overconfident than two-way ones.
        self.register_buffer("temperature", torch.ones(len(MODES)))
        self.register_buffer("k_slope", torch.zeros(()))

    @classmethod
    def from_backbone(cls, name: str = DEFAULT_BACKBONE) -> "DecisionNet":
        enc = AutoModel.from_pretrained(name, attn_implementation="sdpa")
        return cls(enc, enc.config.hidden_size)

    def forward(self, input_ids, attention_mask, opt_pos, opt_mask, span_mask):
        """Returns raw (uncalibrated) logits.

        opt_logits    [B, K]  one per option, -inf on padding
        start_logits  [B, L]  -inf outside {[CLS], state tokens}
        end_logits    [B, L]
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        opt_h = torch.gather(h, 1, opt_pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        opt_logits = self.opt_head(opt_h).squeeze(-1).float()
        opt_logits = opt_logits.masked_fill(~opt_mask, float("-inf"))
        se = self.span_head(h).float().masked_fill(~span_mask.unsqueeze(-1), float("-inf"))
        return opt_logits, se[..., 0], se[..., 1]

    def temp(self, mode: str, n_options: int = 0) -> float:
        t = float(self.temperature[MODES.index(mode)])
        if mode == "one" and n_options > 1:
            t += float(self.k_slope) * math.log(n_options)
        return t

    def save(self, path: str | Path, tokenizer, meta: dict | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "encoder")
        heads = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
        save_file({k: v.contiguous() for k, v in heads.items()}, str(path / "heads.safetensors"))  # no pickle: safe to share
        (path / "meta.json").write_text(json.dumps(meta or {}, indent=2))

    @classmethod
    def load(cls, path: str | Path, device="cuda", dtype=torch.bfloat16):
        path = Path(path)
        cfg = AutoConfig.from_pretrained(path / "encoder")
        enc = AutoModel.from_pretrained(path / "encoder", attn_implementation="sdpa", dtype=dtype)
        model = cls(enc, cfg.hidden_size)
        if (path / "heads.safetensors").exists():
            heads = load_file(str(path / "heads.safetensors"))
        else:  # checkpoints written before the switch to safetensors
            heads = torch.load(path / "heads.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(heads, strict=False)
        model.opt_head.to(dtype)
        model.span_head.to(dtype)
        tok = AutoTokenizer.from_pretrained(path / "encoder")
        meta = json.loads((path / "meta.json").read_text())
        return model.to(device).eval(), tok, meta
