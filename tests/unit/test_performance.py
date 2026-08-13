"""Paired bootstrap, A/A, seeded slowdown, and memory policy tests."""

from __future__ import annotations

import math

import pytest

from upgrade_guard.compare.memory import MIB, device_memory_gate, engine_size_gate
from upgrade_guard.compare.performance import (
    AcceptedPair,
    GateOutcome,
    paired_ratio_gate,
    weighted_performance_gate,
)


def pairs(ratio: float, count: int = 24) -> tuple[AcceptedPair, ...]:
    return tuple(
        AcceptedPair(10.0 + index * 0.01, (10.0 + index * 0.01) * ratio) for index in range(count)
    )


def test_aa_passes_and_seeded_ten_percent_slowdown_is_detected() -> None:
    aa = paired_ratio_gate(pairs(1.0), allowance=0.03, seed=7)
    assert aa.outcome is GateOutcome.PASSED
    assert aa.point == pytest.approx(1.0)
    slowed = paired_ratio_gate(pairs(1.10), allowance=0.03, seed=7)
    assert slowed.outcome is GateOutcome.REGRESSION
    assert slowed.one_sided_lower > 1.03


def test_insufficient_pairs_are_infrastructure_invalid() -> None:
    result = paired_ratio_gate(pairs(1.0, 19), allowance=0.03, seed=7)
    assert result.outcome is GateOutcome.INFRASTRUCTURE_INVALID
    assert math.isnan(result.point)


def test_weighted_gate_resamples_each_shape_and_any_regression_fails() -> None:
    result = weighted_performance_gate(
        {"small": pairs(1.0), "large": pairs(1.10)},
        {"small": 0.7, "large": 0.3},
        {"small": 0.03, "large": 0.03},
        aggregate_allowance=0.03,
        seed=42,
    )
    assert result.shapes["small"].outcome is GateOutcome.PASSED
    assert result.shapes["large"].outcome is GateOutcome.REGRESSION
    assert result.outcome is GateOutcome.REGRESSION


def test_seeded_64_mib_device_increase_and_aa_instability() -> None:
    baseline = (128 * MIB, 128 * MIB, 128 * MIB)
    regression = device_memory_gate(baseline, (192 * MIB, 192 * MIB, 192 * MIB))
    assert regression.outcome is GateOutcome.REGRESSION
    stable = engine_size_gate((10 * MIB,) * 3, (10 * MIB + 512,) * 3)
    assert stable.outcome is GateOutcome.PASSED
    unstable = device_memory_gate((128 * MIB, 140 * MIB, 128 * MIB), (128 * MIB,) * 3)
    assert unstable.outcome is GateOutcome.INFRASTRUCTURE_INVALID
