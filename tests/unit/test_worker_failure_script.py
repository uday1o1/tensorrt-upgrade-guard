"""Filesystem and process-boundary tests for worker failure capture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from upgrade_guard.containers.commands import command_sha256

SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_worker_failure.py"


def _run(kind: str, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        (sys.executable, str(SCRIPT), kind, str(path)),
        check=False,
        capture_output=True,
        text=True,
    )


def test_strict_worker_failure_crosses_real_process_boundary(tmp_path: Path) -> None:
    command = ("python3", "-m", "upgrade_guard.worker.run_correctness")
    result = tmp_path / "correctness.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-correctness/v1",
                "status": "failed",
                "command": command,
                "command_sha256": command_sha256(command),
                "started_unix_seconds": 1.0,
                "ended_unix_seconds": 2.0,
                "duration_seconds": 1.0,
                "failure_code": "EXECUTION_FAILED",
                "error_type": "RuntimeError",
                "message": "typed execution failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _run("correctness", result).returncode == 0

    result.write_text("{}\n", encoding="utf-8")
    malformed = _run("correctness", result)
    assert malformed.returncode == 4
    assert "missing or malformed" in malformed.stdout


def test_runner_routes_strict_worker_failures_through_extended_validators() -> None:
    runner = (SCRIPT.parent / "run_gpu_qualification.sh").read_text(encoding="utf-8")
    helper = runner[
        runner.index("run_extended_worker() {") : runner.index("run_core_qualification() {")
    ]
    plugin = runner[
        runner.index("run_plugin_matrix() {") : runner.index("write_mobilenet_profile() {")
    ]
    mobilenet = runner[runner.index("run_mobilenet_matrix() {") : runner.index("run_aa_pilot() {")]
    assert "scripts/validate_worker_failure.py" in helper
    assert "return 4" in helper
    assert "scripts/validate_plugin_outputs.py" in plugin
    assert "run_extended_worker" in plugin
    assert "scripts/validate_mobilenet_outputs.py" in mobilenet
    assert "run_extended_worker" in mobilenet
