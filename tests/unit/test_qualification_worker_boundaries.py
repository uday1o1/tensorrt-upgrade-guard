"""Focused host-side worker boundary and precedence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.common import Phase
from upgrade_guard.contracts.results import WorkerCorrectnessResult
from upgrade_guard.errors import FailureCode, InfrastructureError
from upgrade_guard.qualification import (
    _input_integrity_failure,
    _load_worker_result,
    _verify_worker_command,
    _worker_failure_record,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def test_qualification_loads_typed_worker_failure_instead_of_relabeling_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate" / "correctness.json"
    path.parent.mkdir(parents=True)
    command = ("python3", "-m", "upgrade_guard.worker.run_correctness")
    path.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-correctness/v1",
                "status": "failed",
                "command": command,
                "command_sha256": command_sha256(command),
                "failure_code": "PROFILE_REJECTED",
                "error_type": "RuntimeError",
                "message": "input shape was rejected for tokens",
                "started_unix_seconds": 1.0,
                "ended_unix_seconds": 2.0,
                "duration_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    worker = _load_worker_result(path, WorkerCorrectnessResult, "correctness")
    failure = _worker_failure_record(
        worker.failure_code,
        worker.message,
        phase=Phase.CORRECTNESS,
        environment_id="candidate",
        precision="fp32",
        shape_id="b1_s8",
        result_path=path,
        output_root=tmp_path,
    )

    assert failure is not None
    assert failure.code is FailureCode.PROFILE_REJECTED
    assert failure.evidence[0].sha256.startswith("sha256:")


def test_input_integrity_precedes_nondeterminism() -> None:
    stable = {"input_integrity_stable": True}
    unstable = {"input_integrity_stable": False}

    assert _input_integrity_failure(unstable, stable) is FailureCode.CORPUS_INVALID
    assert _input_integrity_failure(stable, unstable) is FailureCode.EXECUTION_FAILED
    assert _input_integrity_failure(stable, stable) is None


def test_worker_command_identity_is_exact() -> None:
    command = ("python3", "-m", "upgrade_guard.worker.build_engine")
    _verify_worker_command(command, command_sha256(command), command)
    with pytest.raises(InfrastructureError, match="command evidence"):
        _verify_worker_command(command, digest("0"), command)


def test_malformed_worker_transport_remains_infrastructure(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="result is invalid"):
        _load_worker_result(path, WorkerCorrectnessResult, "correctness")
