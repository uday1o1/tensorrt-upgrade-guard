"""Three-way numerical comparison with bounded diagnostic evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from upgrade_guard.contracts.common import NumericalPolicy, NumericalTolerance
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


class ThreeWayPrecedenceError(Exception):
    """Typed case failure when schema-invalid arrays cannot produce numerical summaries."""

    def __init__(
        self,
        failure_code: FailureCode,
        failed_gates: tuple[str, ...],
        message: str,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.failed_gates = failed_gates


def compare_arrays(
    name: str,
    reference: Array,
    candidate: Array,
    policy: NumericalTolerance,
    *,
    relative_error_guard: float,
    semantics: Literal["classification"] | None = None,
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
    if not np.isfinite(relative_error_guard) or relative_error_guard <= 0:
        raise InvalidInputError("relative-error guard must be finite and positive")
    if semantics == "classification" and (
        reference_array.ndim < 1 or reference_array.shape[-1] < 5
    ):
        raise InvalidInputError("classification semantics require at least five classes")
    reference_finite = np.isfinite(reference_array)
    candidate_finite = np.isfinite(candidate_array)
    reference_nonfinite = int(reference_array.size - np.count_nonzero(reference_finite))
    candidate_nonfinite = int(candidate_array.size - np.count_nonzero(candidate_finite))
    safe_reference = np.nan_to_num(
        reference_array.astype(np.float64),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_candidate = np.nan_to_num(
        candidate_array.astype(np.float64),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    absolute = np.abs(safe_candidate - safe_reference)
    threshold = policy.atol + policy.rtol * np.abs(safe_reference)
    failed = np.flatnonzero((absolute > threshold).reshape(-1))
    relative = absolute / np.maximum(np.abs(safe_reference), relative_error_guard)
    reference_flat = safe_reference.reshape(-1)
    candidate_flat = safe_candidate.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine = (
        float(np.dot(reference_flat, candidate_flat) / denominator)
        if denominator > 0
        else float(reference_flat.size == 0 or np.array_equal(reference_flat, candidate_flat))
    )
    top1_agreement = None
    top5_agreement = None
    if semantics == "classification":
        top1_agreement = bool(
            np.array_equal(
                np.argmax(safe_reference, axis=-1),
                np.argmax(safe_candidate, axis=-1),
            )
        )
        reference_top5 = np.argsort(safe_reference, axis=-1)[..., -5:]
        candidate_top5 = np.argsort(safe_candidate, axis=-1)[..., -5:]
        top5_agreement = bool(
            all(
                set(reference_row) == set(candidate_row)
                for reference_row, candidate_row in zip(
                    reference_top5.reshape(-1, 5),
                    candidate_top5.reshape(-1, 5),
                    strict=True,
                )
            )
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
        top1_agreement=top1_agreement,
        top5_agreement=top5_agreement,
    )


def decide_three_way(
    name: str,
    reference: Array,
    baseline: Array,
    candidate: Array,
    *,
    policy: NumericalPolicy,
    semantics: Literal["classification"] | None = None,
) -> ThreeWayDecision:
    """Apply the locked precedence and three-way numerical decision table."""

    reference_array = np.asarray(reference)
    baseline_array = np.asarray(baseline)
    candidate_array = np.asarray(candidate)
    if (
        reference_array.shape != baseline_array.shape
        or reference_array.dtype != baseline_array.dtype
    ):
        raise ThreeWayPrecedenceError(
            FailureCode.CORPUS_INVALID,
            ("baseline_output_schema",),
            "baseline output schema differs from the locked reference",
        )
    if (
        reference_array.shape != candidate_array.shape
        or reference_array.dtype != candidate_array.dtype
    ):
        raise ThreeWayPrecedenceError(
            FailureCode.OUTPUT_SCHEMA_CHANGED,
            ("candidate_output_schema",),
            "candidate output schema differs from the locked reference",
        )
    if semantics == "classification" and (
        reference_array.ndim < 1 or reference_array.shape[-1] < 5
    ):
        raise ThreeWayPrecedenceError(
            FailureCode.CORPUS_INVALID,
            ("reference_output_schema",),
            "classification reference schema has fewer than five classes",
        )
    baseline_result = compare_arrays(
        name,
        reference_array,
        baseline_array,
        policy.baseline_to_reference,
        relative_error_guard=policy.relative_error_guard,
        semantics=semantics,
    )
    candidate_result = compare_arrays(
        name,
        reference_array,
        candidate_array,
        policy.candidate_to_reference,
        relative_error_guard=policy.relative_error_guard,
        semantics=semantics,
    )
    drift_result = compare_arrays(
        name,
        baseline_array,
        candidate_array,
        policy.candidate_to_baseline,
        relative_error_guard=policy.relative_error_guard,
        semantics=semantics,
    )
    if baseline_result.reference_nonfinite_count:
        return ThreeWayDecision(
            baseline_result,
            candidate_result,
            drift_result,
            False,
            FailureCode.CORPUS_INVALID,
            ("reference_nonfinite",),
        )
    if baseline_result.candidate_nonfinite_count:
        return ThreeWayDecision(
            baseline_result,
            candidate_result,
            drift_result,
            False,
            FailureCode.CORPUS_INVALID,
            ("baseline_nonfinite",),
        )
    if candidate_result.candidate_nonfinite_count:
        return ThreeWayDecision(
            baseline_result,
            candidate_result,
            drift_result,
            False,
            FailureCode.NONFINITE_OUTPUT,
            ("candidate_nonfinite",),
        )
    if not _gate_passed(baseline_result, policy, semantics):
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
        if not _gate_passed(result, policy, semantics)
    )
    return ThreeWayDecision(
        baseline_result,
        candidate_result,
        drift_result,
        not failed,
        FailureCode.NUMERICAL_REGRESSION if failed else None,
        failed,
    )


def _gate_passed(
    result: NumericalSummary,
    policy: NumericalPolicy,
    semantics: Literal["classification"] | None,
) -> bool:
    if not result.elementwise_passed:
        return False
    if semantics != "classification":
        return True
    return not (
        (policy.require_top1_agreement and result.top1_agreement is not True)
        or (policy.require_top5_agreement and result.top5_agreement is not True)
    )


def _maximum(values: Array) -> float:
    return float(np.max(values)) if values.size else 0.0


def _mean(values: Array) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _percentile(values: Array, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0
