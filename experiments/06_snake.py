"""Experiment 6: the decision model plays Snake.

The model has never seen Snake and cannot read a grid. It reads text. So the game writes one short English
"sensor report" per candidate move (does the snake crash, does it get closer to the food, is the path ahead a
dead end), and the model answers two typed questions about each report. All four moves and both questions go
through one batched forward pass, and the snake moves wherever P(yes) for "is this a good move" is highest.

The division of labor is deliberate and worth stating plainly: the geometry (collision test, flood fill) is
ordinary code, and the judgment (reading the report and weighing it) is the model. Nothing about Snake was
trained in; the question below is a plain string passed at call time, like every other experiment here.

The idea for this demo comes from the Snake demo in laya-mlx (https://github.com/mizorewww/laya-mlx).
This is an independent implementation that shares no code or assets with it.

Usage:
    uv run python 06_snake.py                 play 6 seeded games, compare with baselines, run the checks
    uv run python 06_snake.py --watch         watch one game in the terminal
    uv run python 06_snake.py --record        also save a full game trace to assets/snake.json (for make_visuals.py)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import deque
from pathlib import Path

from common import SINGLE_CALL_MS, check, finish, header

from decisionmodel import Bool, DecisionModel

COLS, ROWS = 12, 10
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
MAX_IDLE = 150  # a game is stopped if the snake goes this many steps without eating
SEEDS = range(6)
RECORD_SEED = 3

SCHEMA = {
    "good": Bool("Is this move safe and does it bring the snake closer to the food?"),
    "dies": Bool("Does the snake die if it makes this move?"),
}


class Snake:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        cx, cy = COLS // 2, ROWS // 2
        self.body = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])  # head first
        self.alive, self.score, self.steps, self.idle = True, 0, 0, 0
        self.food = self._place_food()

    def _place_food(self):
        free = [(x, y) for x in range(COLS) for y in range(ROWS) if (x, y) not in self.body]
        return self.rng.choice(free) if free else None

    def solid(self) -> set:
        """Squares that kill. The tail tip is excluded because it moves out of the way."""
        return set(list(self.body)[:-1])

    def step(self, move: str) -> None:
        (hx, hy), (dx, dy) = self.body[0], MOVES[move]
        nxt = (hx + dx, hy + dy)
        self.steps += 1
        self.idle += 1
        if not in_bounds(nxt) or nxt in self.solid():
            self.alive = False
            return
        self.body.appendleft(nxt)
        if nxt == self.food:
            self.score, self.idle = self.score + 1, 0
            self.food = self._place_food()
        else:
            self.body.pop()

    def running(self) -> bool:
        return self.alive and self.food is not None and self.idle < MAX_IDLE


def in_bounds(c) -> bool:
    return 0 <= c[0] < COLS and 0 <= c[1] < ROWS


def sense(g: Snake) -> dict[str, dict]:
    """Plain geometry for each move: what is hit, whether the food gets closer, how much room is left."""
    (hx, hy), (fx, fy), solid = g.body[0], g.food, g.solid()
    out = {}
    for move, (dx, dy) in MOVES.items():
        c = (hx + dx, hy + dy)
        hit = "wall" if not in_bounds(c) else "body" if c in solid else None
        room = 0
        if hit is None:  # flood fill: squares reachable after the move
            seen, todo = {c}, deque([c])
            while todo:
                x, y = todo.popleft()
                for ax, ay in MOVES.values():
                    n = (x + ax, y + ay)
                    if in_bounds(n) and n not in solid and n not in seen:
                        seen.add(n)
                        todo.append(n)
            room = len(seen)
        closer = abs(c[0] - fx) + abs(c[1] - fy) < abs(hx - fx) + abs(hy - fy)
        pocket = hit is None and room < min(len(g.body) + 2, COLS * ROWS - len(g.body))
        out[move] = {"hit": hit, "room": room, "closer": closer, "pocket": pocket}
    return out


def report(move: str, s: dict) -> str:
    """The text the model actually reads for one candidate move."""
    if s["hit"] == "wall":
        return f"If the snake moves {move}, it crashes into the wall and dies."
    if s["hit"] == "body":
        return f"If the snake moves {move}, it bites its own body and dies."
    text = f"If the snake moves {move}, it survives and {'gets closer to' if s['closer'] else 'moves away from'} the food. "
    if s["pocket"]:
        return text + f"Warning: this path leads into a small closed pocket of {s['room']} squares where the snake will be trapped and die."
    return text + "The path ahead is wide open."


class ModelPolicy:
    """Four reports, two questions each, one forward pass. Move where P(good) is highest."""

    def __init__(self, dm: DecisionModel):
        self.dm, self.ms, self.last = dm, [], None

    def __call__(self, g: Snake) -> str:
        senses = sense(g)
        reports = {m: report(m, s) for m, s in senses.items()}
        t0 = time.perf_counter()
        decisions = self.dm.decide_many(list(reports.values()), SCHEMA)
        self.ms.append((time.perf_counter() - t0) * 1000)
        self.last = {m: {"report": reports[m], "p_good": d.good.p(), "p_dies": d.dies.p(), **senses[m]}
                     for m, d in zip(reports, decisions)}
        return max(self.last, key=lambda m: self.last[m]["p_good"])


def random_safe_policy(rng: random.Random):
    """Baseline: never crashes on purpose, otherwise has no idea where the food is."""
    return lambda g: rng.choice([m for m, s in sense(g).items() if s["hit"] is None] or ["up"])


def hand_coded_policy(g: Snake) -> str:
    """Reference: the same sensor facts, combined by an explicit rule instead of by the model."""
    s = sense(g)
    safe = [m for m in s if s[m]["hit"] is None] or ["up"]
    roomy = [m for m in safe if not s[m]["pocket"]] or safe
    return max(roomy, key=lambda m: (s[m]["closer"], s[m]["room"]))


def play(policy, seed: int, on_step=None) -> Snake:
    g = Snake(seed)
    while g.running():
        move = policy(g)
        if on_step:
            on_step(g, move)
        g.step(move)
    return g


def draw(g: Snake, policy: ModelPolicy, move: str) -> None:
    cells = {c: "o" for c in g.body}
    cells[g.body[0]], cells[g.food] = "@", "*"
    rows = ["+" + "-" * (COLS * 2) + "+"]
    rows += ["|" + "".join(cells.get((x, y), ".") + " " for x in range(COLS)) + "|" for y in range(ROWS)]
    rows.append(rows[0])
    side = [f"score {g.score}   length {len(g.body)}   step {g.steps}   {policy.ms[-1]:.0f} ms", ""]
    for m, v in policy.last.items():
        side.append(f"{'>' if m == move else ' '} {m:5} P(good) {v['p_good']:.2f} {'#' * round(v['p_good'] * 20):20}  P(dies) {v['p_dies']:.2f}")
    side += ["", "the model read, for the chosen move:", "  " + policy.last[move]["report"]]
    lines = [r + "   " + (side[i] if i < len(side) else "") for i, r in enumerate(rows)]
    print("\x1b[H\x1b[J" + "\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="show one game in the terminal and exit")
    ap.add_argument("--record", action="store_true", help="save a full game trace to assets/snake.json")
    ap.add_argument("--seed", type=int, default=RECORD_SEED, help="game seed for --watch and --record")
    args = ap.parse_args()
    dm = DecisionModel.load()

    if args.watch:
        policy = ModelPolicy(dm)

        def show(g, move):
            draw(g, policy, move)
            time.sleep(0.06)
        g = play(policy, args.seed, show)
        print(f"\ngame over: score {g.score} in {g.steps} steps ({'crashed' if not g.alive else 'stopped'})")
        return

    header("Experiment 6: playing Snake by reading sensor reports")
    g = Snake(0)
    print("\nWhat the model sees for one board position (seed 0, first move):")
    policy = ModelPolicy(dm)
    policy(g)
    for m, v in policy.last.items():
        print(f"  P(good)={v['p_good']:.2f}  P(dies)={v['p_dies']:.2f}  {v['report']}")

    policy, audit = ModelPolicy(dm), {"steps": 0, "fatal_when_avoidable": 0, "pocket_when_avoidable": 0, "dies_right": 0, "dies_total": 0}

    def watch_choices(g, move):
        v = policy.last
        audit["steps"] += 1
        if v[move]["hit"] and any(not x["hit"] for x in v.values()):
            audit["fatal_when_avoidable"] += 1
        elif v[move]["pocket"] and any(not x["hit"] and not x["pocket"] for x in v.values()):
            audit["pocket_when_avoidable"] += 1
        for x in v.values():
            audit["dies_total"] += 1
            audit["dies_right"] += (x["p_dies"] > 0.5) == bool(x["hit"] or x["pocket"])

    games = [play(policy, s, watch_choices) for s in SEEDS]
    rng = random.Random(0)
    baseline = [play(random_safe_policy(rng), s) for s in SEEDS]
    reference = [play(hand_coded_policy, s) for s in SEEDS]
    mean = lambda gs: statistics.mean(g.score for g in gs)
    print(f"\nFood eaten per game on a {COLS}x{ROWS} board, seeds {list(SEEDS)}:")
    print(f"  random safe moves (baseline)       mean {mean(baseline):5.1f}   {[g.score for g in baseline]}")
    print(f"  decision model                     mean {mean(games):5.1f}   {[g.score for g in games]}")
    print(f"  hand-coded rule on the same facts  mean {mean(reference):5.1f}   {[g.score for g in reference]}")
    print(f"  model decisions: {audit['steps']}, chose a fatal move when a safe one existed: {audit['fatal_when_avoidable']}, "
          f"entered a pocket when open space existed: {audit['pocket_when_avoidable']}")
    ms = sorted(policy.ms)
    p50, p95 = statistics.median(ms), ms[int(0.95 * len(ms)) - 1]
    print(f"  latency per move (4 reports x 2 questions, one pass): p50 {p50:.1f} ms, p95 {p95:.1f} ms")

    if args.record:
        rec, frames = ModelPolicy(dm), []
        g = play(rec, args.seed, lambda g, move: frames.append({
            "body": list(g.body), "food": g.food, "score": g.score, "move": move, "ms": rec.ms[-1],
            "moves": {m: {k: v[k] for k in ("report", "p_good", "p_dies")} for m, v in rec.last.items()}}))
        path = Path(__file__).resolve().parents[1] / "assets" / "snake.json"
        path.write_text(json.dumps({"cols": COLS, "rows": ROWS, "seed": args.seed, "final_score": g.score, "steps": g.steps,
                                    "schema": {k: s.question for k, s in SCHEMA.items()}, "frames": frames}))
        print(f"  recorded seed {args.seed}: score {g.score} in {g.steps} steps -> {path}")

    print("\nChecks:")
    check("model scores at least 10x the random baseline", mean(games) >= 10 * max(mean(baseline), 1.0), f"{mean(games):.1f} vs {mean(baseline):.1f}")
    check("every game reaches at least 15 food", min(g.score for g in games) >= 15, str([g.score for g in games]))
    check("picks an avoidable fatal move in under 0.5% of decisions", audit["fatal_when_avoidable"] < 0.005 * audit["steps"],
          f"{audit['fatal_when_avoidable']} of {audit['steps']}")
    check("P(dies) lands on the right side of 0.5 for at least 97% of reports", audit["dies_right"] >= 0.97 * audit["dies_total"],
          f"{audit['dies_right']} of {audit['dies_total']}")
    check(f"one move decision p95 under {SINGLE_CALL_MS:.0f} ms", p95 < SINGLE_CALL_MS, f"{p95:.1f} ms")
    finish()


if __name__ == "__main__":
    main()
