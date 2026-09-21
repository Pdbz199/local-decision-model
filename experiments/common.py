"""Shared helpers for the experiments."""

from __future__ import annotations

import statistics
import sys
import time

# Jev's advertised end-to-end response time is 70 ms to 500 ms. We hold the local model to the fast end.
JEV_FAST_MS = 70.0
JEV_SLOW_MS = 500.0
# Budget for one small call: half of Jev's fastest figure, which leaves the other half for a network hop.
# Median latency is about 10 ms; the tail moves with desktop GPU sharing and CPU scheduling, so we budget for p95.
SINGLE_CALL_MS = JEV_FAST_MS / 2

_failures: list[str] = []


def latency(fn, runs: int = 100, warmup: int = 5) -> dict[str, float]:
    """Wall-clock latency of fn() in milliseconds, including tokenization and decoding."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"p50": statistics.median(times), "p95": times[int(0.95 * len(times)) - 1], "max": times[-1]}


def fmt(lat: dict[str, float]) -> str:
    return f"p50 {lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms, max {lat['max']:.1f} ms"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def finish() -> None:
    if _failures:
        print(f"\n{len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print("\nAll checks passed.")


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
