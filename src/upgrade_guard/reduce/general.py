"""Bounded confirmed reducers for outputs, options, inputs, and environments."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from upgrade_guard.errors import InvalidInputError

T = TypeVar("T")
Array = npt.NDArray[np.generic]


class TrialOutcome(StrEnum):
    """One predicate trial outcome with infrastructure separated from behavior."""

    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ReductionLimits:
    """Trial, wall-clock, and repeated-confirmation bounds."""

    maximum_trials: int
    maximum_seconds: float
    confirmation_count: int

    def __post_init__(self) -> None:
        if self.maximum_trials < self.confirmation_count or self.maximum_seconds <= 0:
            raise InvalidInputError("reduction limits cannot satisfy one confirmation")
        if self.confirmation_count < 2:
            raise InvalidInputError("reduction requires at least two confirmations")


@dataclass(frozen=True, slots=True)
class ReductionTrace:
    """Auditable reducer result and bounded search statistics."""

    original_items: int
    reduced_items: int
    trials: int
    inconclusive_trials: int
    budget_exhausted: bool


@dataclass(frozen=True, slots=True)
class SequenceReduction[T]:
    """Smallest confirmed sequence found within the configured limits."""

    items: tuple[T, ...]
    trace: ReductionTrace


class ConfirmedEvaluator[T]:
    """Count confirmed predicate executions under trial and time limits."""

    def __init__(
        self,
        predicate: Callable[[T], TrialOutcome],
        limits: ReductionLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._predicate = predicate
        self._limits = limits
        self._clock = clock
        self._started = clock()
        self.trials = 0
        self.inconclusive_trials = 0
        self.exhausted = False

    def confirms(self, candidate: T) -> bool:
        """Require the same reproduced outcome repeatedly and fail closed on noise."""

        for _ in range(self._limits.confirmation_count):
            if (
                self.trials >= self._limits.maximum_trials
                or self._clock() - self._started >= self._limits.maximum_seconds
            ):
                self.exhausted = True
                return False
            outcome = self._predicate(candidate)
            self.trials += 1
            if outcome is TrialOutcome.INCONCLUSIVE:
                self.inconclusive_trials += 1
                return False
            if outcome is not TrialOutcome.REPRODUCED:
                return False
        return True


def reduce_sequence(
    items: Sequence[T],
    predicate: Callable[[tuple[T, ...]], TrialOutcome],
    limits: ReductionLimits,
    *,
    minimum_items: int = 0,
) -> SequenceReduction[T]:
    """Apply deterministic delta debugging while preserving a confirmed predicate."""

    original = tuple(items)
    if minimum_items < 0 or minimum_items > len(original):
        raise InvalidInputError("minimum reducer item count is invalid")
    evaluator = ConfirmedEvaluator(predicate, limits)
    if not evaluator.confirms(original):
        raise InvalidInputError("original reduction candidate is not a stable confirmed failure")
    current = original
    granularity = 2
    while len(current) > minimum_items and not evaluator.exhausted:
        chunk_size = max(1, (len(current) + granularity - 1) // granularity)
        changed = False
        for start in range(0, len(current), chunk_size):
            candidate = current[:start] + current[start + chunk_size :]
            if len(candidate) < minimum_items:
                continue
            if evaluator.confirms(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                changed = True
                break
            if evaluator.exhausted:
                break
        if changed:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return SequenceReduction(
        current,
        ReductionTrace(
            original_items=len(original),
            reduced_items=len(current),
            trials=evaluator.trials,
            inconclusive_trials=evaluator.inconclusive_trials,
            budget_exhausted=evaluator.exhausted,
        ),
    )


@dataclass(frozen=True, slots=True)
class InputReduction:
    """Simplified finite input and its bounded search trace."""

    values: Array
    changed_elements: int
    trials: int
    inconclusive_trials: int
    budget_exhausted: bool


def simplify_finite_input(
    values: Array,
    predicate: Callable[[Array], TrialOutcome],
    limits: ReductionLimits,
) -> InputReduction:
    """Replace regions with zero, one, or one constant while preserving failure."""

    original = np.asarray(values)
    if original.size == 0 or not np.issubdtype(original.dtype, np.number):
        raise InvalidInputError("input reduction requires a nonempty numeric array")
    if not np.all(np.isfinite(original)):
        raise InvalidInputError("input reduction accepts finite values only")
    evaluator = ConfirmedEvaluator(predicate, limits)
    if not evaluator.confirms(original.copy()):
        raise InvalidInputError("original input is not a stable confirmed failure")
    current = original.copy()
    candidates = (0.0, 1.0, float(current.reshape(-1)[0]))
    for constant in candidates:
        candidate = np.full_like(current, constant)
        if np.array_equal(candidate, current):
            continue
        if evaluator.confirms(candidate):
            current = candidate
            return InputReduction(
                current,
                int(np.count_nonzero(current != original)),
                evaluator.trials,
                evaluator.inconclusive_trials,
                evaluator.exhausted,
            )
        if evaluator.exhausted:
            break
    chunk = max(1, current.size // 2)
    while chunk >= 1 and not evaluator.exhausted:
        changed = False
        for start in range(0, current.size, chunk):
            for constant in (0.0, 1.0):
                candidate = current.copy().reshape(-1)
                candidate[start : start + chunk] = constant
                reshaped = candidate.reshape(current.shape)
                if np.array_equal(reshaped, current):
                    continue
                if evaluator.confirms(reshaped):
                    current = reshaped.copy()
                    changed = True
                    break
                if evaluator.exhausted:
                    break
            if evaluator.exhausted:
                break
        if chunk == 1:
            break
        chunk //= 2
        if not changed and chunk == 0:
            break
    changed_elements = int(np.count_nonzero(current != original))
    return InputReduction(
        current,
        changed_elements,
        evaluator.trials,
        evaluator.inconclusive_trials,
        evaluator.exhausted,
    )


@dataclass(frozen=True, slots=True)
class EnvironmentBoundary:
    """First adjacent nonfailing and confirmed-failing environments."""

    last_passing: str
    first_failing: str
    trials: int
    inconclusive_trials: int


def reduce_environment_history(
    environment_ids: Sequence[str],
    predicate: Callable[[str], TrialOutcome],
    limits: ReductionLimits,
) -> EnvironmentBoundary:
    """Find an ordered adjacent pass-to-fail boundary without assuming noisy trials pass."""

    history = tuple(environment_ids)
    if len(history) < 2 or len(history) != len(set(history)):
        raise InvalidInputError("environment history requires at least two unique ordered entries")
    evaluator = ConfirmedEvaluator(predicate, limits)
    states: list[bool] = []
    for environment_id in history:
        reproduced = evaluator.confirms(environment_id)
        if evaluator.inconclusive_trials:
            raise InvalidInputError("environment boundary trial was inconclusive")
        if evaluator.exhausted:
            raise InvalidInputError("environment boundary reduction exhausted its budget")
        states.append(reproduced)
    for index in range(1, len(history)):
        if not states[index - 1] and states[index]:
            return EnvironmentBoundary(
                history[index - 1],
                history[index],
                evaluator.trials,
                evaluator.inconclusive_trials,
            )
    raise InvalidInputError("environment history has no adjacent pass-to-fail boundary")
