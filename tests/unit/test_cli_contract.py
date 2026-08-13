"""Stored public CLI contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from upgrade_guard.cli import app


def test_public_cli_matches_stored_contract() -> None:
    fixture = json.loads(Path("tests/fixtures/cli/public-contract.json").read_text())
    result = CliRunner().invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    for command in fixture["commands"]:
        root = command.split()[0]
        assert root in result.output
    assert "dev" not in result.output
    assert fixture["exit_codes"] == {
        "gate_failed": 1,
        "invalid_input": 2,
        "unsupported": 3,
        "infrastructure_or_inconclusive": 4,
        "internal": 5,
        "passed": 0,
    }
