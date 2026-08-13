"""Three-way numerical comparison with bounded diagnostic evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from upgrade_guard.contracts.common import NumericalTolerance
from upgrade_guard.contracts.results import NumericalSummary
from upgrade_guard.errors import FailureCode, InvalidInputError

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class ThreeWayDecision:
    """Independent reference validity and upgrade drift decisions."""

    baseline_to_reference: NumericalSummary
    candidate_to_reference: NumericalSummary
    candidate_to_baseline: NumericalSummary
    passed: bool
    failure_code: FailureCode | None
    failed_gates: tuple[str, ...]


def compare_arrays(
    name: str,
    reference: Array,
    candidate: Array,
    policy: NumericalTolerance,
    *,
    maximum_failed_indexes: int = 16,
) -> NumericalSummary:
    """Compute the complete elementwise numerical evidence for one output."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise InvalidInputError(
            "output shape differs",
            details={
                "output": name,
                "reference": reference_array.shape,
                "candidate": candidate_array.shape,
            },
        )
    if reference_array.dtype != candidate_array.dtype:
        raise InvalidInputError(
            "output dtype differs",
            details={
                "output": name,
                "reference": str(reference_array.dtype),
                "candidate": str(candidate_array.dtype),
            },
        )
    reference_finite = np.isfinite(reference_array)
    candidate_finite = np.isfinite(candidate_array)
    reference_nonfinite = int(reference_array.size - np.count_nonzero(reference_finite))
    candidate_nonfinite = int(candidate_array.size - np.count_nonzero(candidate_finite))
    safe_reference = np.nan_to_num(reference_array.astype(np.float64), copy=False)
    safe_candidate = np.nan_to_num(candidate_array.astype(np.float64), copy=False)
    absolute = np.abs(safe_candidate - safe_reference)
    threshold = policy.atol + policy.rtol * np.abs(safe_reference)
    failed = np.flatnonzero((absolute > threshold).reshape(-1))
    guard = max(policy.atol, np.finfo(np.float64).eps)
    relative = absolute / np.maximum(np.abs(safe_reference), guard)
    reference_flat = safe_reference.reshape(-1)
    candidate_flat = safe_candidate.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine = (
        float(np.dot(reference_flat, candidate_flat) / denominator)
        if denominator > 0
        else float(reference_flat.size == 0 or np.array_equal(reference_flat, candidate_flat))
    )
    return NumericalSummary(
        output_name=name,
        element_count=int(reference_array.size),
        maximum_absolute_error=_maximum(absolute),
        mean_absolute_error=_mean(absolute),
        median_absolute_error=_percentile(absolute, 50),
        p99_absolute_error=_percentile(absolute, 99),
        maximum_relative_error=_maximum(relative),
        p99_relative_error=_percentile(relative, 99),
        cosine_similarity=max(-1.0, min(1.0, cosine)),
        l2_error=float(np.linalg.norm(absolute.reshape(-1))),
        reference_nonfinite_count=reference_nonfinite,
        candidate_nonfinite_count=candidate_nonfinite,
        failed_element_count=int(failed.size),
        failed_element_indexes=tuple(int(item) for item in failed[:maximum_failed_indexes]),
        elementwise_passed=not reference_nonfinite and not candidate_nonfinite and not failed.size,
        top1_agreement=None,
        top5_agreement=None,
    )


def decide_three_way(
    name: str,
    reference: Array,
    baseline: Array,
    candidate: Array,
    *,
    baseline_policy: NumericalTolerance,
    candidate_policy: NumericalTolerance,
    drift_policy: NumericalTolerance,
) -> ThreeWayDecision:
    """Apply the locked precedence and three-way numerical decision table."""

    baseline_result = compare_arrays(name, reference, baseline, baseline_policy)
    candidate_result = compare_arrays(name, reference, candidate, candidate_policy)
    drift_result = compare_arrays(name, baseline, candidate, drift_policy)
    if not baseline_result.elementwise_passed:
        return ThreeWayDecision(
            baseline_result,
            candidate_result,
            drift_result,
            False,
            FailureCode.CORPUS_INVALID,
            ("baseline_to_reference",),
        )
    failed = tuple(
        label
        for label, result in (
            ("candidate_to_reference", candidate_result),
            ("candidate_to_baseline", drift_result),
        )
        if not result.elementwise_passed
    )
    return ThreeWayDecision(
        baseline_result,
        candidate_result,
        drift_result,
        not failed,
        FailureCode.NUMERICAL_REGRESSION if failed else None,
        failed,
    )


def _maximum(values: Array) -> float:
    return float(np.max(values)) if values.size else 0.0


def _mean(values: Array) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _percentile(values: Array, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0
