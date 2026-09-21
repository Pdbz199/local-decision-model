"""Public API: unstructured state in, typed probabilistic decisions out.

    dm = DecisionModel.load("checkpoints/decision-model")
    d = dm.decide(ticket_text, {
        "urgent": Bool("Is this urgent?"),
        "team": Enum("Which team should handle this?", ["billing", "shipping", "tech support"]),
    })
    if d.urgent.p() > 0.8: ...

Every field of every state becomes one row of a single batched forward pass, so
a whole schema is answered in parallel rather than token by token.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .encoding import Encoder, Query
from .model import DecisionNet
from .types import Bool, Decision, Decisions, Enum, Extract, Float, Int, MultiLabel, Spec


def render_state(state: Any) -> str:
    """Strings pass through; any other program state is rendered as JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2, default=str)


class DecisionModel:
    def __init__(self, model: DecisionNet, tokenizer, meta: dict, device: str, max_len: int = 8192, batch_tokens: int = 65536):
        self.model, self.meta, self.device = model, meta, device
        self.encoder = Encoder(tokenizer, max_len)
        self.batch_tokens = batch_tokens

    @classmethod
    def load(cls, path: str | Path | None = None, device: str | None = None, warmup: bool = True, **kw) -> "DecisionModel":
        """Loads a trained checkpoint. `path` defaults to $DECISION_MODEL_PATH, then to checkpoints/decision-model."""
        path = Path(path or os.environ.get("DECISION_MODEL_PATH", "checkpoints/decision-model"))
        if not path.exists() and not path.is_absolute():  # allow running from any directory inside the repo
            path = Path(__file__).resolve().parents[2] / path
        if not (path / "meta.json").exists():
            raise FileNotFoundError(f"no trained model at {path}; run scripts/build_data.py and scripts/train.py first")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model, tok, meta = DecisionNet.load(path, device=device, dtype=dtype)
        dm = cls(model, tok, meta, device, **kw)
        if warmup:  # pay one-time CUDA kernel selection costs up front, not on the first real call
            for n in (8, 64, 512):
                dm.decide("warm up " * n, {"a": Bool("Is this a test?"), "b": Extract("Which word repeats?")})
        return dm

    # ------------------------------------------------------------------ public

    def decide(self, state: Any, schema: Mapping[str, Spec]) -> Decisions:
        """Answers every field of `schema` about one state, in one forward pass."""
        return self.decide_many([state], schema)[0]

    def decide_many(self, states: Sequence[Any], schema: Mapping[str, Spec]) -> list[Decisions]:
        """Map a schema over many states. Rows are length-sorted and packed into large batches."""
        for name, spec in schema.items():
            if not isinstance(spec, Spec):
                raise TypeError(f"schema field {name!r} must be a decisionmodel type spec, got {type(spec).__name__}")
        texts = [render_state(s) for s in states]
        state_toks = [self.encoder.tokenize_state(t) for t in texts]
        rows = []  # (state index, field name, spec, Encoded)
        for i, text in enumerate(texts):
            for name, spec in schema.items():
                enc = self.encoder.encode(Query(spec.mode, spec.question, spec.options(), text), state_tok=state_toks[i])
                rows.append((i, name, spec, enc))

        results = [Decisions() for _ in states]
        order = sorted(range(len(rows)), key=lambda r: len(rows[r][3].input_ids))
        batch, longest = [], 0
        for r in order + [None]:
            n = len(rows[r][3].input_ids) if r is not None else 0
            if batch and (r is None or max(longest, n) * (len(batch) + 1) > self.batch_tokens):
                self._run([rows[b] for b in batch], texts, results)
                batch, longest = [], 0
            if r is not None:
                batch.append(r)
                longest = max(longest, n)
        # restore the schema's field order
        return [Decisions((k, d[k]) for k in schema) for d in results]

    def check(self, state: Any, question: str) -> float:
        """Shorthand for a single yes/no predicate. Returns P(yes)."""
        return self.decide(state, {"x": Bool(question)})["x"].p()

    # ------------------------------------------------------------------ internals

    @torch.inference_mode()
    def _run(self, rows, texts, results) -> None:
        encs = [r[3] for r in rows]
        batch = self.encoder.collate(encs, device=self.device)
        opt_logits, start_logits, end_logits = self.model(**batch)
        opt_logits = opt_logits.cpu()
        for b, (i, name, spec, enc) in enumerate(rows):
            T = self.model.temp(spec.mode, len(enc.opt_pos))
            if spec.mode == "span":
                results[i][name] = self._decode_span(spec, enc, texts[i], start_logits[b] / T, end_logits[b] / T)
            else:
                results[i][name] = _decode_choice(spec, opt_logits[b, : len(enc.opt_pos)] / T)

    def _decode_span(self, spec: Extract, enc, text: str, start_logits, end_logits) -> Decision:
        """Scores every (start, end) pair up to max_tokens long, plus the "no answer" pair at [CLS]."""
        ps, pe = torch.softmax(start_logits, -1), torch.softmax(end_logits, -1)
        p_null = float(ps[0] * pe[0])
        s0, n = enc.state_start, len(enc.offsets)
        if n == 0:
            return Decision(None, 1.0, {None: 1.0})
        ps, pe = ps[s0 : s0 + n], pe[s0 : s0 + n]
        W = min(spec.max_tokens, n)
        # joint[w, i] = P(start=i) * P(end=i+w)
        pe_pad = torch.cat([pe, pe.new_zeros(W)])
        joint = torch.stack([ps * pe_pad[w : w + n] for w in range(W)])
        total = p_null + float(joint.sum())
        top = torch.topk(joint.flatten(), k=min(5, joint.numel()))
        probs: dict[Any, float] = {None: p_null / total}
        best = None
        for val, flat in zip(top.values.tolist(), top.indices.tolist()):
            w, i = divmod(flat, n)
            if val <= 0.0 or i + w >= n:  # zero-probability filler from the padded region
                continue
            a, b = enc.offsets[i][0], enc.offsets[i + w][1]
            while a < b and text[a].isspace():
                a += 1
            frag = text[a:b]
            if not frag:
                continue
            if best is None:
                best = (frag, (a, b))
            probs[frag] = probs.get(frag, 0.0) + val / total
        if best is None or probs[None] >= 0.5:
            return Decision(None, probs[None], probs)
        return Decision(best[0], probs[best[0]], probs, span=best[1])


def _decode_choice(spec: Spec, logits: torch.Tensor) -> Decision:
    if isinstance(spec, MultiLabel):
        p = torch.sigmoid(logits.float()).tolist()
        probs = dict(zip(spec.choices, p))
        chosen = [c for c, pi in probs.items() if pi >= spec.threshold]
        return Decision(chosen, min(max(pi, 1 - pi) for pi in p), probs)
    p = torch.softmax(logits.float(), -1).tolist()
    if isinstance(spec, Bool):
        value = p[0] >= spec.threshold
        return Decision(value, p[0] if value else p[1], {True: p[0], False: p[1]})
    if isinstance(spec, Enum):
        k = max(range(len(p)), key=p.__getitem__)
        return Decision(spec.choices[k], p[k], dict(zip(spec.choices, p)))
    if isinstance(spec, (Int, Float)):
        vals = spec.values()
        k = max(range(len(p)), key=p.__getitem__)
        expected = min(max(sum(v * pi for v, pi in zip(vals, p)), vals[0]), vals[-1])  # guard against float rounding
        value = vals[k] if isinstance(spec, Int) else expected
        return Decision(value, p[k], dict(zip(vals, p)), expected=expected)
    raise TypeError(f"unsupported spec {type(spec).__name__}")
