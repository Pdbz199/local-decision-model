"""The JevBench question mapping needs no GPU and no checkpoint, so it is tested here without downloading anything."""

import importlib.util
from pathlib import Path

import pytest

from decisionmodel import Bool, Enum

spec = importlib.util.spec_from_file_location("eval_jevbench", Path(__file__).parents[1] / "scripts" / "eval_jevbench.py")
eval_jevbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_jevbench)
to_spec = eval_jevbench.to_spec

NOUL = {"type": "noul", "instructions": "Has the order shipped?",
        "criteria": {"true": "The text says so", "false": "The text says it is not so"}}
CHOICE = {"type": "choice", "instructions": "Which intent?",
          "criteria": {"track": "Wants to know where an order is", "cancel": "Wants to cancel"}}
SCORE = {"type": "score", "instructions": "Rate the impact.", "criteria": ["cosmetic", "one user blocked", "data loss"]}


def test_noul_maps_to_bool_and_label_order_matches_jevbench():
    s, opts = to_spec(NOUL, ["no", "yes"], "plain")
    assert isinstance(s, Bool) and s.question == NOUL["instructions"]
    assert opts == ["no", "yes"]
    s, _ = to_spec(NOUL, ["no", "yes"], "criteria")
    assert "Yes means: The text says so" in s.question and "No means:" in s.question


def test_choice_keeps_label_order_so_positions_map_back():
    s, opts = to_spec(CHOICE, ["track", "cancel"], "plain")
    assert isinstance(s, Enum) and list(s.choices) == ["track", "cancel"] == opts
    s, opts = to_spec(CHOICE, ["track", "cancel"], "criteria")
    assert opts == ["track: Wants to know where an order is", "cancel: Wants to cancel"]
    assert list(s.choices) == opts


def test_score_levels_become_readable_options():
    _, opts = to_spec(SCORE, ["0", "1", "2"], "plain")
    assert opts == ["level 0", "level 1", "level 2"]
    _, opts = to_spec(SCORE, ["0", "1", "2"], "criteria")
    assert opts == ["level 0: cosmetic", "level 1: one user blocked", "level 2: data loss"]


def test_criteria_variant_falls_back_when_criteria_are_missing():
    _, opts = to_spec({"type": "choice", "instructions": "q", "criteria": None}, ["a", "b"], "criteria")
    assert opts == ["a", "b"]
    _, opts = to_spec({"type": "score", "instructions": "q", "criteria": ["only one"]}, ["0", "1"], "criteria")
    assert opts == ["level 0", "level 1"]


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        to_spec({"type": "essay", "instructions": "q"}, ["a", "b"], "plain")
