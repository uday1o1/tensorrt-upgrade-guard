"""Confirmed engine-size and device-memory regression gates."""

from __future__ import annotations

from dataclasses import dataclass

from upgrade_guard.compare.performance import GateOutcome
from upgrade_guard.errors import InvalidInputError

MIB = 1024 * 1024


@dataclass(frozen=True)
class MemoryGate:
    """A/A validity, effective allowance, and confirmation outcome."""

    baseline_bytes: int
    candidate_bytes: tuple[int, ...]
    allowance_bytes: int
    aa_spread_bytes: int
    outcome: GateOutcome


def confirmed_memory_gate(
    baseline_confirmations: tuple[int, ...],
    candidate_confirmations: tuple[int, ...],
    *,
    fixed_allowance_bytes: int,
    proportional_allowance: float = 0.05,
) -> MemoryGate:
    """Require three consistent comparisons and stable A/A behavior."""

    if len(baseline_confirmations) < 3 or len(candidate_confirmations) < 3:
        raise InvalidInputError("memory gate requires three baseline and candidate builds")
    if any(value < 0 for value in baseline_confirmations + candidate_confirmations):
        raise InvalidInputError("memory observations cannot be negative")
    baseline = round(sum(baseline_confirmations) / len(baseline_confirmations))
    allowance = max(fixed_allowance_bytes, round(baseline * proportional_allowance))
    aa_spread = max(baseline_confirmations) - min(baseline_confirmations)
    if aa_spread > allowance:
        outcome = GateOutcome.INFRASTRUCTURE_INVALID
    else:
        exceeded = tuple(value > baseline + allowance for value in candidate_confirmations)
        if all(exceeded):
            outcome = GateOutcome.REGRESSION
        elif any(exceeded):
            outcome = GateOutcome.INCONCLUSIVE
        else:
            outcome = GateOutcome.PASSED
    return MemoryGate(baseline, candidate_confirmations, allowance, aa_spread, outcome)


def engine_size_gate(
    baseline_confirmations: tuple[int, ...], candidate_confirmations: tuple[int, ...]
) -> MemoryGate:
    """Use the locked max(1 MiB, 5 percent) engine-size policy."""

    return confirmed_memory_gate(
        baseline_confirmations,
        candidate_confirmations,
        fixed_allowance_bytes=MIB,
    )


def device_memory_gate(
    baseline_confirmations: tuple[int, ...], candidate_confirmations: tuple[int, ...]
) -> MemoryGate:
    """Use the locked max(8 MiB, 5 percent) device-memory policy."""

    return confirmed_memory_gate(
        baseline_confirmations,
        candidate_confirmations,
        fixed_allowance_bytes=8 * MIB,
    )
