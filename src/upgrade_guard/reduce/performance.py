"""Repeated-confidence performance evidence reduction."""

from __future__ import annotations

import time
from dataclasses import dataclass

from upgrade_guard.compare.performance import (
    AcceptedPair,
    GateOutcome,
    RatioEstimate,
    paired_ratio_gate,
)
from upgrade_guard.errors import InvalidInputError


@dataclass(frozen=True, slots=True)
class ReducedPerformanceFailure:
    """Smallest prefix that retains the statistical regression decision."""

    pairs: tuple[AcceptedPair, ...]
    estimate: RatioEstimate
    original_pairs: int
    evaluated_candidates: int
    budget_exhausted: bool


def reduce_performance_failure(
    pairs: tuple[AcceptedPair, ...],
    *,
    allowance: float,
    seed: int,
    replicates: int,
    minimum_pairs: int = 20,
    maximum_candidates: int | None = None,
    maximum_seconds: float | None = None,
) -> ReducedPerformanceFailure:
    """Remove paired blocks only when the repeated bootstrap regression remains."""

    original = paired_ratio_gate(
        pairs,
        allowance=allowance,
        seed=seed,
        replicates=replicates,
        minimum_pairs=minimum_pairs,
    )
    if original.outcome is not GateOutcome.REGRESSION:
        raise InvalidInputError("performance evidence does not satisfy the regression predicate")
    if maximum_candidates is not None and maximum_candidates < 1:
        raise InvalidInputError("performance reduction candidate budget must be positive")
    if maximum_seconds is not None and maximum_seconds <= 0:
        raise InvalidInputError("performance reduction time budget must be positive")
    started = time.monotonic()
    best_pairs = pairs
    best_estimate = original
    evaluated = 1
    exhausted = False
    for length in range(minimum_pairs, len(pairs)):
        if (maximum_candidates is not None and evaluated >= maximum_candidates) or (
            maximum_seconds is not None and time.monotonic() - started >= maximum_seconds
        ):
            exhausted = True
            break
        candidate = pairs[:length]
        estimate = paired_ratio_gate(
            candidate,
            allowance=allowance,
            seed=seed,
            replicates=replicates,
            minimum_pairs=minimum_pairs,
        )
        evaluated += 1
        if estimate.outcome is GateOutcome.REGRESSION:
            best_pairs = candidate
            best_estimate = estimate
            break
    return ReducedPerformanceFailure(best_pairs, best_estimate, len(pairs), evaluated, exhausted)
