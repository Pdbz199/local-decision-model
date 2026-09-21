"""Output types for decisions.

A schema is a dict of field name -> type spec. Every spec compiles down to one
"query" for the model: a question, a mode, and (for choice modes) a closed list
of options. Because the model can only ever score the options it was given (or
point at a span of the input), the decoded value is always of the declared type.

Modes understood by the model:
    "one"   exactly one option is correct        -> softmax over options
    "any"   each option is independently yes/no  -> sigmoid per option
    "span"  answer is a substring of the state   -> softmax over start/end tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

MAX_OPTIONS = 255
BOOL_OPTIONS = ("yes", "no")


class Spec:
    """Base class for field type specs."""

    question: str
    mode: str

    def options(self) -> list[str]:
        return []


@dataclass(frozen=True)
class Bool(Spec):
    """A yes/no predicate about the state. Decodes to a Python bool."""

    question: str
    threshold: float = 0.5
    mode: str = field(default="one", init=False)

    def options(self) -> list[str]:
        return list(BOOL_OPTIONS)


@dataclass(frozen=True)
class Enum(Spec):
    """Exactly one of a closed set of string options. Decodes to one of them."""

    question: str
    choices: Sequence[str]
    mode: str = field(default="one", init=False)

    def __post_init__(self):
        _check_choices(self.choices)

    def options(self) -> list[str]:
        return list(self.choices)


@dataclass(frozen=True)
class MultiLabel(Spec):
    """Any subset of a closed set of options. Decodes to a list of options."""

    question: str
    choices: Sequence[str]
    threshold: float = 0.5
    mode: str = field(default="any", init=False)

    def __post_init__(self):
        _check_choices(self.choices)

    def options(self) -> list[str]:
        return list(self.choices)


@dataclass(frozen=True)
class Int(Spec):
    """An integer in [lo, hi] (inclusive). Decodes to a Python int.

    Each integer is an option, so the range must hold at most 255 values. The
    result also carries the expected value, which is often the more useful
    number for scoring tasks.
    """

    question: str
    lo: int
    hi: int
    mode: str = field(default="one", init=False)

    def __post_init__(self):
        if self.hi <= self.lo:
            raise ValueError("Int needs hi > lo")
        if self.hi - self.lo + 1 > MAX_OPTIONS:
            raise ValueError(f"Int range may hold at most {MAX_OPTIONS} values; use Float for wider ranges")

    def values(self) -> list[int]:
        return list(range(self.lo, self.hi + 1))

    def options(self) -> list[str]:
        return [str(v) for v in self.values()]


@dataclass(frozen=True)
class Float(Spec):
    """A float in [lo, hi], discretized into evenly spaced grid points.

    Decodes to the expected value under the predicted distribution, which is
    always inside [lo, hi].
    """

    question: str
    lo: float
    hi: float
    bins: int = 11
    mode: str = field(default="one", init=False)

    def __post_init__(self):
        if self.hi <= self.lo:
            raise ValueError("Float needs hi > lo")
        if not 2 <= self.bins <= MAX_OPTIONS:
            raise ValueError(f"bins must be in [2, {MAX_OPTIONS}]")

    def values(self) -> list[float]:
        step = (self.hi - self.lo) / (self.bins - 1)
        return [self.lo + i * step for i in range(self.bins)]

    def options(self) -> list[str]:
        return [f"{v:g}" for v in self.values()]


@dataclass(frozen=True)
class Extract(Spec):
    """A verbatim substring of the state, or None when the state has no answer.

    The value is copied out of the input by character offsets, so it cannot
    contain text that is not in the state.
    """

    question: str
    max_tokens: int = 32
    mode: str = field(default="span", init=False)


def _check_choices(choices: Sequence[str]) -> None:
    if isinstance(choices, str):
        raise TypeError("choices must be a sequence of strings, not a single string")
    if not 2 <= len(choices) <= MAX_OPTIONS:
        raise ValueError(f"need between 2 and {MAX_OPTIONS} choices, got {len(choices)}")
    if len(set(choices)) != len(choices):
        raise ValueError("choices must be unique")
    if not all(isinstance(c, str) and c.strip() for c in choices):
        raise ValueError("choices must be non-empty strings")


@dataclass
class Decision:
    """One decoded field.

    value        the typed value (bool, str, list[str], int, float, str or None)
    confidence   probability that `value` is right, as estimated by the model.
                 For MultiLabel it is the probability of the least certain
                 per-option call. For Float it is the mass of the modal bin.
    probs        full distribution: option -> probability. For Extract it maps
                 the top candidate spans (and None) to their probabilities.
    expected     expected value for Int and Float fields, else None.
    span         (start, end) character offsets into the state for Extract.
    """

    value: Any
    confidence: float
    probs: dict[Any, float]
    expected: float | None = None
    span: tuple[int, int] | None = None

    def p(self, option: Any = True) -> float:
        """Probability of one option. For Bool fields, p() is P(True)."""
        return self.probs[option]


class Decisions(dict):
    """Dict of field name -> Decision, with attribute access for convenience."""

    def __getattr__(self, name: str) -> Decision:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def values_dict(self) -> dict[str, Any]:
        return {k: d.value for k, d in self.items()}
