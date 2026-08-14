"""Seeded GPU fault evidence validation tests."""

from __future__ import annotations

import copy

import pytest

from scripts.validate_seeded_gpu_faults import ORDER_SCHEDULE, validate_records


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, order in enumerate(ORDER_SCHEDULE):
        baseline = 0.01 + index * 0.00001
        candidate = baseline * 1.10
        records.append(
            {
                "G2": {"detected": True, "control": "passed"},
                "G3": {"detected": True, "control": "passed"},
                "G5": {
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
    assert result["G5"]["accepted_pairs"] == 20  # type: ignore[index]
    assert list(ORDER_SCHEDULE).count("baseline_then_candidate") == 10


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
