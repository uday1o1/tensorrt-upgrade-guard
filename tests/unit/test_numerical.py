"""Numerical policy and decision precedence tests."""

from __future__ import annotations

import numpy as np
import pytest

from upgrade_guard.compare.numerical import compare_arrays, decide_three_way
from upgrade_guard.contracts.common import NumericalTolerance
from upgrade_guard.errors import FailureCode, InvalidInputError


def policy(atol: float = 1e-5, rtol: float = 1e-4) -> NumericalTolerance:
    return NumericalTolerance(atol=atol, rtol=rtol)


def test_numerical_metrics_and_bounded_failed_indexes() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    candidate = np.asarray([0.1, 1.0, 2.2, 3.0], dtype=np.float32)
    result = compare_arrays("output", reference, candidate, policy(), maximum_failed_indexes=1)
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
        baseline_policy=policy(),
        candidate_policy=policy(),
        drift_policy=policy(),
    )
    assert invalid_baseline.failure_code is FailureCode.CORPUS_INVALID
    regression = decide_three_way(
        "output",
        reference,
        good,
        bad,
        baseline_policy=policy(),
        candidate_policy=policy(),
        drift_policy=policy(),
    )
    assert regression.failure_code is FailureCode.NUMERICAL_REGRESSION
    assert regression.failed_gates == ("candidate_to_reference", "candidate_to_baseline")


def test_schema_and_nonfinite_fail_closed() -> None:
    reference = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(InvalidInputError, match="shape"):
        compare_arrays("output", reference, np.ones(4, dtype=np.float32), policy())
    with pytest.raises(InvalidInputError, match="dtype"):
        compare_arrays("output", reference, reference.astype(np.float16), policy())
    result = compare_arrays(
        "output",
        reference,
        np.asarray([[np.nan, 1], [1, 1]], dtype=np.float32),
        policy(),
    )
    assert not result.elementwise_passed
    assert result.candidate_nonfinite_count == 1
