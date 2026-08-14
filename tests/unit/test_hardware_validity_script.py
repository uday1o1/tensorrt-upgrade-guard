"""Tests for shell-pilot hardware validity evidence."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.hardware_validity import transition
from tests.unit.test_qualification_edges import _specification


def test_transition_rejects_loaded_clock_and_power_drift(tmp_path: Path) -> None:
    specification = tmp_path / "qualification.yaml"
    specification.write_text(
        yaml.safe_dump(_specification().model_dump(mode="json")),
        encoding="utf-8",
    )
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    base = {
        "status": "passed",
        "rejection_reasons": [],
        "observed": {
            "graphics_clock_mhz": 2000,
            "power_watts": 200,
            "power_limit_watts": 300,
        },
    }
    before.write_text(json.dumps(base), encoding="utf-8")
    drifted = json.loads(json.dumps(base))
    drifted["observed"]["graphics_clock_mhz"] = 1500
    after.write_text(json.dumps(drifted), encoding="utf-8")
    output = tmp_path / "transition.json"

    assert transition(specification, before, after, output) is False
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "rejected"
    assert "graphics_clock_variation_exceeded" in result["rejection_reasons"]
