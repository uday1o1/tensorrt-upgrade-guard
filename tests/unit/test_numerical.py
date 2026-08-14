"""Numerical policy and decision precedence tests."""

from __future__ import annotations

import numpy as np
import pytest

from upgrade_guard.compare.numerical import (
    ThreeWayPrecedenceError,
    compare_arrays,
    decide_three_way,
)
from upgrade_guard.contracts.common import NumericalPolicy, NumericalTolerance
from upgrade_guard.errors import FailureCode, InvalidInputError


def policy(atol: float = 1e-5, rtol: float = 1e-4) -> NumericalTolerance:
    return NumericalTolerance(atol=atol, rtol=rtol)


def authored_policy(
    atol: float = 1e-5,
    rtol: float = 1e-4,
    *,
    guard: float = 1e-12,
    top1: bool = False,
    top5: bool = False,
) -> NumericalPolicy:
    tolerance = policy(atol, rtol)
    return NumericalPolicy(
        baseline_to_reference=tolerance,
        candidate_to_reference=tolerance,
        candidate_to_baseline=tolerance,
        relative_error_guard=guard,
        require_top1_agreement=top1,
        require_top5_agreement=top5,
    )


def test_numerical_metrics_and_bounded_failed_indexes() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    candidate = np.asarray([0.1, 1.0, 2.2, 3.0], dtype=np.float32)
    result = compare_arrays(
        "output",
        reference,
        candidate,
        policy(),
        relative_error_guard=1e-12,
        maximum_failed_indexes=1,
    )
    assert not result.elementwise_passed
    assert result.failed_element_count == 2
    assert result.failed_element_indexes == (0,)
    assert result.maximum_absolute_error == pytest.approx(0.2)
    assert result.cosine_similarity > 0.99


def test_three_way_decision_distinguishes_corpus_and_upgrade_failure() -> None:
    reference = np.ones(8, dtype=np.float32)
    good = reference.copy()
    bad = reference + np.float32(0.1)
    invalid_baseline = decide_three_way(
        "output",
        reference,
        bad,
        good,
        policy=authored_policy(),
    )
    assert invalid_baseline.failure_code is FailureCode.CORPUS_INVALID
    regression = decide_three_way(
        "output",
        reference,
        good,
        bad,
        policy=authored_policy(),
    )
    assert regression.failure_code is FailureCode.NUMERICAL_REGRESSION
    assert regression.failed_gates == ("candidate_to_reference", "candidate_to_baseline")


def test_schema_and_nonfinite_fail_closed() -> None:
    reference = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(InvalidInputError, match="shape"):
        compare_arrays(
            "output",
            reference,
            np.ones(4, dtype=np.float32),
            policy(),
            relative_error_guard=1e-12,
        )
    with pytest.raises(InvalidInputError, match="dtype"):
        compare_arrays(
            "output",
            reference,
            reference.astype(np.float16),
            policy(),
            relative_error_guard=1e-12,
        )
    result = compare_arrays(
        "output",
        reference,
        np.asarray([[np.nan, 1], [1, 1]], dtype=np.float32),
        policy(),
        relative_error_guard=1e-12,
    )
    assert not result.elementwise_passed
    assert result.candidate_nonfinite_count == 1


def test_three_way_uses_specific_schema_and_nonfinite_precedence() -> None:
    reference = np.ones((2, 6), dtype=np.float32)
    candidate_nonfinite = reference.copy()
    candidate_nonfinite[0, 0] = np.nan
    nonfinite = decide_three_way(
        "output",
        reference,
        reference,
        candidate_nonfinite,
        policy=authored_policy(),
    )
    assert nonfinite.failure_code is FailureCode.NONFINITE_OUTPUT
    assert nonfinite.failed_gates == ("candidate_nonfinite",)

    baseline_nonfinite = reference.copy()
    baseline_nonfinite[0, 0] = np.inf
    invalid_baseline = decide_three_way(
        "output",
        reference,
        baseline_nonfinite,
        reference,
        policy=authored_policy(),
    )
    assert invalid_baseline.failure_code is FailureCode.CORPUS_INVALID
    assert invalid_baseline.failed_gates == ("baseline_nonfinite",)

    invalid_reference = reference.copy()
    invalid_reference[0, 0] = np.nan
    invalid_corpus = decide_three_way(
        "output",
        invalid_reference,
        reference,
        reference,
        policy=authored_policy(),
    )
    assert invalid_corpus.failure_code is FailureCode.CORPUS_INVALID
    assert invalid_corpus.failed_gates == ("reference_nonfinite",)

    with pytest.raises(ThreeWayPrecedenceError) as baseline_schema:
        decide_three_way(
            "output",
            reference,
            reference[:, :-1],
            reference,
            policy=authored_policy(),
        )
    assert baseline_schema.value.failure_code is FailureCode.CORPUS_INVALID

    with pytest.raises(ThreeWayPrecedenceError) as candidate_schema:
        decide_three_way(
            "output",
            reference,
            reference,
            reference.astype(np.float64),
            policy=authored_policy(),
        )
    assert candidate_schema.value.failure_code is FailureCode.OUTPUT_SCHEMA_CHANGED


def test_relative_error_uses_authored_guard_exactly() -> None:
    reference = np.asarray([0.0], dtype=np.float32)
    candidate = np.asarray([1e-6], dtype=np.float32)
    result = compare_arrays(
        "output",
        reference,
        candidate,
        policy(atol=1e-5, rtol=0),
        relative_error_guard=1e-3,
    )
    assert result.maximum_relative_error == pytest.approx(1e-3)
    with pytest.raises(InvalidInputError, match="relative-error guard"):
        compare_arrays(
            "output",
            reference,
            candidate,
            policy(),
            relative_error_guard=0,
        )


def test_classification_semantics_are_evidence_and_authored_gates() -> None:
    reference = np.asarray([[10, 9, 8, 7, 6, 0]], dtype=np.float32)
    baseline = reference.copy()
    candidate = np.asarray([[9, 10, 8, 7, 6, 0]], dtype=np.float32)
    permissive = authored_policy(atol=2, rtol=0)
    observed = decide_three_way(
        "logits",
        reference,
        baseline,
        candidate,
        policy=permissive,
        semantics="classification",
    )
    assert observed.candidate_to_reference.top1_agreement is False
    assert observed.candidate_to_reference.top5_agreement is True
    assert observed.passed
    required = decide_three_way(
        "logits",
        reference,
        baseline,
        candidate,
        policy=authored_policy(atol=2, rtol=0, top1=True, top5=True),
        semantics="classification",
    )
    assert required.failure_code is FailureCode.NUMERICAL_REGRESSION
    assert required.failed_gates == ("candidate_to_reference", "candidate_to_baseline")
