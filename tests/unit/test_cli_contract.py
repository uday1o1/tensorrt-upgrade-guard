"""Stored public CLI contract fixture."""

from __future__ import annotations

import json
import subprocess
import sys
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


def test_cli_import_does_not_load_model_or_reference_runtimes() -> None:
    command = (
        "import sys; import upgrade_guard.cli; "
        "blocked={'numpy','onnx','onnxruntime','tensorrt','cuda'}; "
        "loaded={name.split('.',1)[0] for name in sys.modules}; "
        "assert not blocked & loaded, sorted(blocked & loaded)"
    )

    result = subprocess.run(  # noqa: S603 - fixed interpreter and authored test code
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_failure_classifier_and_public_contracts_import_in_a_fresh_process() -> None:
    command = (
        "import upgrade_guard.classify; "
        "from upgrade_guard.contracts import BuildManifest, RunResult; "
        "assert BuildManifest.__name__ == 'BuildManifest'; "
        "assert RunResult.__name__ == 'RunResult'"
    )

    result = subprocess.run(  # noqa: S603 - fixed interpreter and authored test code
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
