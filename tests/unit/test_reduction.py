"""Stable numerical and profile reduction tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from upgrade_guard.errors import InvalidInputError
from upgrade_guard.reduce.inputs import reduce_numerical_failure
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
