"""Fail-closed validation for the remote A/A pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_aa
from upgrade_guard.compare.performance import GateOutcome, RatioEstimate


def _write_pairs(root: Path, count: int = 20) -> None:
    for index in range(count):
        directory = root / f"pair-{index:02d}"
        directory.mkdir(parents=True)
        payload = json.dumps({"times": [{"computeMs": 1.0}, {"computeMs": 1.01}]})
        (directory / "a.json").write_text(payload, encoding="utf-8")
        (directory / "b.json").write_text(payload, encoding="utf-8")
        (directory / "validity.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (GateOutcome.PASSED, "passed"),
        (GateOutcome.REGRESSION, "failed"),
        (GateOutcome.INCONCLUSIVE, "failed"),
        (GateOutcome.INFRASTRUCTURE_INVALID, "failed"),
    ],
)
def test_aa_requires_exact_passing_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: GateOutcome,
    expected_status: str,
) -> None:
    _write_pairs(tmp_path)
    monkeypatch.setattr(
        validate_aa,
        "paired_ratio_gate",
        lambda *args, **kwargs: RatioEstimate(1, 1, 1, 1, 1, 20, 0.05, outcome),
    )
    result = validate_aa.validate_aa(tmp_path)
    assert result["status"] == expected_status


def test_aa_rejects_fewer_than_twenty_pairs(tmp_path: Path) -> None:
    _write_pairs(tmp_path, count=19)
    result = validate_aa.validate_aa(tmp_path)
    assert result["status"] == "failed"
    assert result["accepted_pairs"] == 19
