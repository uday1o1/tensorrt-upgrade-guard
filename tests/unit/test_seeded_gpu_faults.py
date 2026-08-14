"""Seeded GPU fault evidence validation tests."""

from __future__ import annotations

import copy

import pytest

from scripts.validate_seeded_gpu_faults import (
    ORDER_SCHEDULE,
    classify_seed_record,
    validate_records,
    validate_serialization_record,
)
from upgrade_guard.errors import FailureCode


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, order in enumerate(ORDER_SCHEDULE):
        baseline = 0.01 + index * 0.00001
        candidate = baseline * 1.10
        records.append(
            {
                "G2": {
                    "mechanism": "plugin_omits_residual_at_hidden_259",
                    "detected": True,
                    "control": "passed",
                    "observed_facts": {"numerical_valid": False},
                },
                "G3": {
                    "mechanism": "zero_epsilon_zero_input",
                    "detected": True,
                    "control": "passed",
                    "observed_facts": {"finite": False},
                },
                "G5": {
                    "mechanism": "controlled_device_delay",
                    "detected": True,
                    "control": "passed",
                    "calibrated": True,
                    "pair_index": index,
                    "order": order,
                    "order_seed": 20260813,
                    "target_ratio": 1.10,
                    "baseline_ms": baseline,
                    "candidate_ms": candidate,
                    "ratio": candidate / baseline,
                },
            }
        )
    return records


def test_seeded_slowdown_uses_balanced_locked_schedule_and_detects_boundary() -> None:
    result = validate_records(_records())
    assert result["status"] == "passed"
    assert result["G5"]["confirmed"] is True  # type: ignore[index]
    assert result["G2"]["classification"] == "NUMERICAL_REGRESSION"  # type: ignore[index]
    assert result["G3"]["classification"] == "NONFINITE_OUTPUT"  # type: ignore[index]
    assert result["G5"]["classification"] == "PERFORMANCE_REGRESSION"  # type: ignore[index]
    assert result["G5"]["accepted_pairs"] == 24  # type: ignore[index]
    assert list(ORDER_SCHEDULE).count("baseline_then_candidate") == 12


@pytest.mark.parametrize("mutation", ["order", "control", "ratio", "count"])
def test_seeded_slowdown_rejects_tampered_evidence(mutation: str) -> None:
    records = copy.deepcopy(_records())
    if mutation == "order":
        records[0]["G5"]["order"] = "candidate_then_baseline"  # type: ignore[index]
    elif mutation == "control":
        records[0]["G3"]["control"] = "failed"  # type: ignore[index]
    elif mutation == "ratio":
        records[0]["G5"]["ratio"] = 1.5  # type: ignore[index]
    else:
        records.pop()
    with pytest.raises(RuntimeError):
        validate_records(records)


def test_all_external_gpu_seed_facts_use_the_production_classifier() -> None:
    cases = {
        "G1": (
            "unsupported_custom_domain_onnx_node",
            {"parsed": False},
            FailureCode.ONNX_PARSE_FAILED,
        ),
        "G4": (
            "vectorized_tail_out_of_bounds",
            {"sanitizer_valid": False},
            FailureCode.SANITIZER_FAILURE,
        ),
        "G7": (
            "input_exceeds_optimization_profile",
            {"profile_accepted": False},
            FailureCode.PROFILE_REJECTED,
        ),
    }
    for seed, (mechanism, facts, expected) in cases.items():
        record = {
            "mechanism": mechanism,
            "detected": True,
            "control": "passed",
            "observed_facts": facts,
        }
        assert classify_seed_record(seed, record) is expected


def test_g6_requires_actual_epsilon_restoration_field_evidence() -> None:
    result = validate_serialization_record(
        {
            "mechanism": "creator_omits_serialized_epsilon",
            "detected": True,
            "control": "passed",
            "expected_epsilon": 0.0125,
            "control_epsilon": 0.0125,
            "fault_epsilon": 1e-5,
            "serialized_epsilon_present": True,
            "control_restore_epsilon_present": True,
            "fault_restore_epsilon_present": False,
            "observed_facts": {"numerical_valid": False},
        }
    )
    assert result["classification"] == "NUMERICAL_REGRESSION"
