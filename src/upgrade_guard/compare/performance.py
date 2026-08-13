"""Seeded paired-bootstrap performance gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from upgrade_guard.errors import InvalidInputError


class GateOutcome(StrEnum):
    """A statistical or infrastructure gate outcome."""

    PASSED = "passed"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"


@dataclass(frozen=True)
class AcceptedPair:
    """Adjacent valid baseline and candidate medians."""

    baseline_milliseconds: float
    candidate_milliseconds: float


@dataclass(frozen=True)
class RatioEstimate:
    """Ratio estimate, confidence bounds, and decision."""

    point: float
    two_sided_lower: float
    two_sided_upper: float
    one_sided_lower: float
    one_sided_upper: float
    accepted_pairs: int
    allowance: float
    outcome: GateOutcome


@dataclass(frozen=True)
class PerformanceGate:
    """Per-shape and weighted workload evidence."""

    shapes: dict[str, RatioEstimate]
    aggregate: RatioEstimate
    outcome: GateOutcome


def paired_ratio_gate(
    pairs: tuple[AcceptedPair, ...],
    *,
    allowance: float,
    seed: int,
    replicates: int = 5_000,
    minimum_pairs: int = 20,
) -> RatioEstimate:
    """Apply the locked log-ratio paired-bootstrap gate."""

    if allowance < 0 or replicates < 1 or minimum_pairs < 1:
        raise InvalidInputError("performance policy values are invalid")
    if len(pairs) < minimum_pairs:
        return RatioEstimate(
            point=math.nan,
            two_sided_lower=math.nan,
            two_sided_upper=math.nan,
            one_sided_lower=math.nan,
            one_sided_upper=math.nan,
            accepted_pairs=len(pairs),
            allowance=allowance,
            outcome=GateOutcome.INFRASTRUCTURE_INVALID,
        )
    logs = _log_ratios(pairs)
    rng = np.random.Generator(np.random.PCG64(seed))
    indexes = rng.integers(0, len(logs), size=(replicates, len(logs)))
    bootstrap = np.exp(np.mean(logs[indexes], axis=1))
    return _estimate(logs, bootstrap, allowance)


def weighted_performance_gate(
    pairs_by_shape: dict[str, tuple[AcceptedPair, ...]],
    weights: dict[str, float],
    allowances: dict[str, float],
    *,
    aggregate_allowance: float,
    seed: int,
    replicates: int = 5_000,
    minimum_pairs: int = 20,
) -> PerformanceGate:
    """Resample every shape within complete pairs for a weighted gate."""

    if set(pairs_by_shape) != set(weights) or set(weights) != set(allowances):
        raise InvalidInputError("pairs, weights, and allowances must name the same shapes")
    if abs(sum(weights.values()) - 1.0) > 1e-9 or any(weight <= 0 for weight in weights.values()):
        raise InvalidInputError("positive performance weights must sum to one")
    shape_results = {
        shape: paired_ratio_gate(
            pairs,
            allowance=allowances[shape],
            seed=seed + index + 1,
            replicates=replicates,
            minimum_pairs=minimum_pairs,
        )
        for index, (shape, pairs) in enumerate(sorted(pairs_by_shape.items()))
    }
    if any(item.outcome is GateOutcome.INFRASTRUCTURE_INVALID for item in shape_results.values()):
        aggregate = RatioEstimate(
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            min(len(item) for item in pairs_by_shape.values()),
            aggregate_allowance,
            GateOutcome.INFRASTRUCTURE_INVALID,
        )
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        weighted_means = np.zeros(replicates, dtype=np.float64)
        point = 0.0
        for shape in sorted(pairs_by_shape):
            logs = _log_ratios(pairs_by_shape[shape])
            indexes = rng.integers(0, len(logs), size=(replicates, len(logs)))
            weighted_means += weights[shape] * np.mean(logs[indexes], axis=1)
            point += weights[shape] * float(np.mean(logs))
        aggregate = _estimate(
            np.asarray([point]),
            np.exp(weighted_means),
            aggregate_allowance,
            point_override=math.exp(point),
            accepted_pairs=sum(len(item) for item in pairs_by_shape.values()),
        )
    outcomes = [result.outcome for result in shape_results.values()] + [aggregate.outcome]
    overall = _overall(outcomes)
    return PerformanceGate(shape_results, aggregate, overall)


def coefficient_of_variation(values: tuple[float, ...]) -> float:
    """Population coefficient of variation for a benchmark pilot."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or np.any(array <= 0):
        raise InvalidInputError("coefficient of variation needs two positive samples")
    return float(np.std(array) / np.mean(array))


def _log_ratios(pairs: tuple[AcceptedPair, ...]) -> np.ndarray:
    baseline = np.asarray([pair.baseline_milliseconds for pair in pairs], dtype=np.float64)
    candidate = np.asarray([pair.candidate_milliseconds for pair in pairs], dtype=np.float64)
    if np.any(baseline <= 0) or np.any(candidate <= 0):
        raise InvalidInputError("paired timings must be positive")
    return np.log(candidate / baseline)


def _estimate(
    logs: np.ndarray,
    bootstrap: np.ndarray,
    allowance: float,
    *,
    point_override: float | None = None,
    accepted_pairs: int | None = None,
) -> RatioEstimate:
    point = point_override if point_override is not None else math.exp(float(np.mean(logs)))
    lower_two, lower_one, upper_one, upper_two = np.percentile(bootstrap, [2.5, 5, 95, 97.5])
    boundary = 1 + allowance
    if upper_one <= boundary:
        outcome = GateOutcome.PASSED
    elif lower_one > boundary:
        outcome = GateOutcome.REGRESSION
    else:
        outcome = GateOutcome.INCONCLUSIVE
    return RatioEstimate(
        point=point,
        two_sided_lower=float(lower_two),
        two_sided_upper=float(upper_two),
        one_sided_lower=float(lower_one),
        one_sided_upper=float(upper_one),
        accepted_pairs=accepted_pairs if accepted_pairs is not None else len(logs),
        allowance=allowance,
        outcome=outcome,
    )


def _overall(outcomes: list[GateOutcome]) -> GateOutcome:
    for outcome in (
        GateOutcome.INFRASTRUCTURE_INVALID,
        GateOutcome.REGRESSION,
        GateOutcome.INCONCLUSIVE,
    ):
        if outcome in outcomes:
            return outcome
    return GateOutcome.PASSED
