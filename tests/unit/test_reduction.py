"""Stable numerical and profile reduction tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from upgrade_guard.errors import InvalidInputError
from upgrade_guard.reduce.general import (
    ConfirmedEvaluator,
    ReductionLimits,
    TrialOutcome,
    reduce_environment_history,
    reduce_sequence,
    simplify_finite_input,
)
from upgrade_guard.reduce.inputs import reduce_numerical_failure
from upgrade_guard.reduce.performance import reduce_performance_failure
from upgrade_guard.reduce.polygraphy import reduction_commands
from upgrade_guard.reduce.predicate import ProfilePredicate
from upgrade_guard.reduce.session import reduce_failure_directory
from upgrade_guard.reduce.shapes import reduce_profile_failure


def test_numerical_reducer_retains_strongest_threshold_violation() -> None:
    reference = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 1] += 0.1
    candidate[1, 1] += 0.5
    reduced = reduce_numerical_failure(reference, candidate, atol=1e-5, rtol=1e-4)
    assert reduced.original_shape == (2, 2)
    assert reduced.multidimensional_index == (1, 1)
    assert reduced.reference.shape == (1,)
    assert reduced.absolute_error > reduced.threshold


def test_numerical_reducer_rejects_nonfailing_evidence() -> None:
    array = np.ones(4, dtype=np.float32)
    with pytest.raises(InvalidInputError, match="do not satisfy"):
        reduce_numerical_failure(array, array.copy(), atol=1e-5, rtol=1e-4)


def test_profile_reducer_keeps_one_minimal_violation() -> None:
    reduced = reduce_profile_failure(
        ProfilePredicate(
            kind="profile",
            input_name="tokens",
            observed_shape=(9, 513, 256),
            minimum_shape=(1, 8, 256),
            maximum_shape=(8, 512, 256),
        )
    )
    assert reduced.observed_shape == (9, 8, 256)
    assert reduced.maximum_shape == (8, 8, 256)
    assert reduced.violating_dimension == 0


def test_public_reduction_writes_hash_addressed_smaller_arrays(tmp_path: Path) -> None:
    source = tmp_path / "failure"
    source.mkdir()
    reference = np.arange(8, dtype=np.float32)
    candidate = reference.copy()
    candidate[5] += 1
    np.save(source / "reference.npy", reference, allow_pickle=False)
    np.save(source / "candidate.npy", candidate, allow_pickle=False)
    (source / "reduction-request.json").write_text(
        json.dumps(
            {
                "api_version": "upgradeguard.dev/v1alpha1",
                "kind": "ReductionRequest",
                "failure_code": "NUMERICAL_REGRESSION",
                "signature_sha256": "sha256:" + "1" * 64,
                "confirmation_count": 2,
                "maximum_trials": 20,
                "maximum_seconds": 60,
                "predicate": {
                    "kind": "numerical",
                    "output_name": "output",
                    "reference_path": "reference.npy",
                    "candidate_path": "candidate.npy",
                    "atol": 1e-5,
                    "rtol": 1e-4,
                },
            }
        ),
        encoding="utf-8",
    )
    result = reduce_failure_directory(source, tmp_path / "reduced")
    assert result["failure_code"] == "NUMERICAL_REGRESSION"
    assert np.load(tmp_path / "reduced" / "candidate.npy").shape == (1,)


def test_polygraphy_uses_bisect_then_linear_with_argument_arrays(tmp_path: Path) -> None:
    commands = reduction_commands(
        model=tmp_path / "model.onnx",
        output=tmp_path / "reduced.onnx",
        predicate_command=("upgrade-guard", "dev", "predicate", "predicate.json"),
    )
    assert commands[0][-1] == "--mode=bisect"
    assert commands[1][-1] == "--mode=linear"
    assert "polygraphy" == commands[0][0]


def test_confirmed_sequence_reduction_removes_unrelated_outputs_and_options() -> None:
    def predicate(items: tuple[str, ...]) -> TrialOutcome:
        return (
            TrialOutcome.REPRODUCED
            if "failing-output" in items and "required-option" in items
            else TrialOutcome.NOT_REPRODUCED
        )

    reduced = reduce_sequence(
        ("unused-output", "failing-output", "unused-option", "required-option"),
        predicate,
        ReductionLimits(maximum_trials=100, maximum_seconds=10, confirmation_count=2),
    )
    assert set(reduced.items) == {"failing-output", "required-option"}
    assert reduced.trace.reduced_items == 2
    assert reduced.trace.inconclusive_trials == 0


def test_sequence_reducer_fails_closed_on_unstable_original() -> None:
    outcomes = iter((TrialOutcome.REPRODUCED, TrialOutcome.INCONCLUSIVE))
    with pytest.raises(InvalidInputError, match="stable confirmed"):
        reduce_sequence(
            ("output",),
            lambda _: next(outcomes),
            ReductionLimits(maximum_trials=4, maximum_seconds=10, confirmation_count=2),
        )


def test_finite_input_reduction_retains_failure_and_simplifies_values() -> None:
    values = np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    def predicate(candidate: np.ndarray[tuple[int], np.dtype[np.float32]]) -> TrialOutcome:
        return (
            TrialOutcome.REPRODUCED
            if float(np.sum(candidate)) != 4.0
            else TrialOutcome.NOT_REPRODUCED
        )

    reduced = simplify_finite_input(
        values,
        predicate,
        ReductionLimits(maximum_trials=50, maximum_seconds=10, confirmation_count=2),
    )
    assert np.array_equal(reduced.values, np.zeros_like(values))
    assert reduced.changed_elements == 4


def test_environment_reducer_returns_first_adjacent_transition() -> None:
    failing = {"11.1", "11.2"}
    boundary = reduce_environment_history(
        ("10.13", "11.0", "11.1", "11.2"),
        lambda environment: (
            TrialOutcome.REPRODUCED if environment in failing else TrialOutcome.NOT_REPRODUCED
        ),
        ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2),
    )
    assert boundary.last_passing == "11.0"
    assert boundary.first_failing == "11.1"


def test_performance_reducer_never_uses_fewer_than_twenty_pairs() -> None:
    from upgrade_guard.compare.performance import AcceptedPair

    pairs = tuple(AcceptedPair(1.0, 1.2) for _ in range(30))
    reduced = reduce_performance_failure(
        pairs,
        allowance=0.10,
        seed=7,
        replicates=1000,
    )
    assert len(reduced.pairs) == 20
    assert reduced.estimate.one_sided_lower > 1.10
    with pytest.raises(InvalidInputError, match="does not satisfy"):
        reduce_performance_failure(
            (AcceptedPair(1.0, 1.2),),
            allowance=0.10,
            seed=7,
            replicates=1000,
        )


def test_public_performance_reduction_requires_repeated_pairs(tmp_path: Path) -> None:
    source = tmp_path / "performance-failure"
    source.mkdir()
    (source / "baseline.json").write_text(json.dumps([1.0] * 25), encoding="utf-8")
    (source / "candidate.json").write_text(json.dumps([1.2] * 25), encoding="utf-8")
    (source / "reduction-request.json").write_text(
        json.dumps(
            {
                "api_version": "upgradeguard.dev/v1alpha1",
                "kind": "ReductionRequest",
                "failure_code": "PERFORMANCE_REGRESSION",
                "signature_sha256": "sha256:" + "2" * 64,
                "confirmation_count": 2,
                "maximum_trials": 100,
                "maximum_seconds": 60,
                "predicate": {
                    "kind": "performance",
                    "baseline_path": "baseline.json",
                    "candidate_path": "candidate.json",
                    "allowance": 0.10,
                    "bootstrap_seed": 7,
                    "bootstrap_replicates": 1000,
                    "minimum_pairs": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    result = reduce_failure_directory(source, tmp_path / "performance-reduced")
    assert result["failure_code"] == "PERFORMANCE_REGRESSION"
    assert result["reduced_pairs"] == 20


def test_reduction_limits_and_evaluator_enforce_both_budgets() -> None:
    with pytest.raises(InvalidInputError, match="cannot satisfy"):
        ReductionLimits(maximum_trials=1, maximum_seconds=1, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="at least two"):
        ReductionLimits(maximum_trials=2, maximum_seconds=1, confirmation_count=1)

    times = iter((0.0, 2.0))
    evaluator = ConfirmedEvaluator(
        lambda _: TrialOutcome.REPRODUCED,
        ReductionLimits(maximum_trials=2, maximum_seconds=1, confirmation_count=2),
        clock=lambda: next(times),
    )
    assert not evaluator.confirms("candidate")
    assert evaluator.exhausted


def test_sequence_reduction_validates_minimum_and_records_exhaustion() -> None:
    limits = ReductionLimits(maximum_trials=2, maximum_seconds=10, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="minimum"):
        reduce_sequence(("a",), lambda _: TrialOutcome.REPRODUCED, limits, minimum_items=2)
    reduced = reduce_sequence(
        ("a", "b"),
        lambda _: TrialOutcome.REPRODUCED,
        limits,
    )
    assert reduced.items == ("a", "b")
    assert reduced.trace.budget_exhausted


def test_input_reducer_validates_values_and_uses_region_simplification() -> None:
    limits = ReductionLimits(maximum_trials=100, maximum_seconds=10, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="nonempty numeric"):
        simplify_finite_input(
            np.asarray([], dtype=np.float32), lambda _: TrialOutcome.REPRODUCED, limits
        )
    with pytest.raises(InvalidInputError, match="finite"):
        simplify_finite_input(
            np.asarray([np.inf], dtype=np.float32),
            lambda _: TrialOutcome.REPRODUCED,
            limits,
        )
    with pytest.raises(InvalidInputError, match="stable confirmed"):
        simplify_finite_input(
            np.asarray([1.0], dtype=np.float32),
            lambda _: TrialOutcome.NOT_REPRODUCED,
            limits,
        )

    original = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    def regional(candidate: np.ndarray[tuple[int], np.dtype[np.float32]]) -> TrialOutcome:
        reproduced = np.array_equal(candidate, original) or (
            np.all(candidate[:2] == 0) and np.array_equal(candidate[2:], original[2:])
        )
        return TrialOutcome.REPRODUCED if reproduced else TrialOutcome.NOT_REPRODUCED

    reduced = simplify_finite_input(original, regional, limits)
    assert np.array_equal(reduced.values, np.asarray([0.0, 0.0, 4.0, 5.0]))
    assert reduced.changed_elements == 2


def test_environment_history_rejects_invalid_noisy_and_missing_boundaries() -> None:
    limits = ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="unique ordered"):
        reduce_environment_history(("same", "same"), lambda _: TrialOutcome.REPRODUCED, limits)
    with pytest.raises(InvalidInputError, match="inconclusive"):
        reduce_environment_history(
            ("first", "second"),
            lambda _: TrialOutcome.INCONCLUSIVE,
            limits,
        )
    with pytest.raises(InvalidInputError, match="no adjacent"):
        reduce_environment_history(
            ("first", "second"),
            lambda _: TrialOutcome.NOT_REPRODUCED,
            limits,
        )
    with pytest.raises(InvalidInputError, match="exhausted"):
        reduce_environment_history(
            ("first", "second"),
            lambda _: TrialOutcome.REPRODUCED,
            ReductionLimits(maximum_trials=2, maximum_seconds=10, confirmation_count=2),
        )


def test_performance_reducer_reports_and_validates_candidate_budget() -> None:
    from upgrade_guard.compare.performance import AcceptedPair

    pairs = tuple(AcceptedPair(1.0, 1.2) for _ in range(25))
    exhausted = reduce_performance_failure(
        pairs,
        allowance=0.1,
        seed=5,
        replicates=1000,
        maximum_candidates=1,
    )
    assert exhausted.budget_exhausted
    assert exhausted.pairs == pairs
    with pytest.raises(InvalidInputError, match="candidate budget"):
        reduce_performance_failure(
            pairs,
            allowance=0.1,
            seed=5,
            replicates=1000,
            maximum_candidates=0,
        )
    with pytest.raises(InvalidInputError, match="time budget"):
        reduce_performance_failure(
            pairs,
            allowance=0.1,
            seed=5,
            replicates=1000,
            maximum_seconds=0,
        )


def test_public_profile_reduction_and_malformed_timings_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "profile-failure"
    source.mkdir()
    request = {
        "api_version": "upgradeguard.dev/v1alpha1",
        "kind": "ReductionRequest",
        "failure_code": "PROFILE_REJECTED",
        "signature_sha256": "sha256:" + "3" * 64,
        "confirmation_count": 2,
        "maximum_trials": 20,
        "maximum_seconds": 60,
        "predicate": {
            "kind": "profile",
            "input_name": "tokens",
            "observed_shape": [9, 128, 256],
            "minimum_shape": [1, 8, 256],
            "maximum_shape": [8, 512, 256],
        },
    }
    (source / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    reduced = reduce_failure_directory(source, tmp_path / "profile-reduced")
    assert reduced["kind"] == "profile"

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "baseline.json").write_text("{}", encoding="utf-8")
    (malformed / "candidate.json").write_text("[]", encoding="utf-8")
    request["failure_code"] = "PERFORMANCE_REGRESSION"
    request["predicate"] = {
        "kind": "performance",
        "baseline_path": "baseline.json",
        "candidate_path": "candidate.json",
        "allowance": 0.1,
        "bootstrap_seed": 7,
        "bootstrap_replicates": 1000,
        "minimum_pairs": 20,
    }
    (malformed / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="timing array"):
        reduce_failure_directory(malformed, tmp_path / "malformed-reduced")
