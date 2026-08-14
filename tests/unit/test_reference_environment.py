"""Independent reference image lock tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.factories import FIXED_TIME, resolved_image
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.errors import InfrastructureError, InvalidInputError
from upgrade_guard.matrix.digest import RegistryClient, ResolvedArtifact
from upgrade_guard.reference_environment import ReferenceEnvironmentLocker


class QueueRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = results
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        self.commands.append(command)
        if not self.results:
            raise AssertionError(f"unexpected command: {command}")
        result = self.results.pop(0)
        return CommandResult(command, result.returncode, result.stdout, result.stderr, 0.01)


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult((), returncode, stdout, stderr, 0.01)


def _probe() -> str:
    return json.dumps(
        {
            "schema_version": "upgradeguard.dev/reference-probe/v1",
            "operating_system": "Debian GNU/Linux 12",
            "architecture": "x86_64",
            "python": "3.12.13",
            "onnx": "1.22.0",
            "onnxruntime": "1.28.0",
            "numpy": "2.4.2",
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        }
    )


def _resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    image = resolved_image(reference="registry.example/reference:v1")
    monkeypatch.setattr(
        RegistryClient,
        "resolve_linux_amd64",
        lambda self, reference: ResolvedArtifact(image=image, config={}),
    )


def test_locks_exact_isolated_cpu_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _resolve(monkeypatch)
    runner = QueueRunner([_result(stdout="pulled"), _result(stdout=_probe())])
    output = tmp_path / "reference.lock.json"
    lock = ReferenceEnvironmentLocker(runner, clock=lambda: FIXED_TIME).lock(
        "registry.example/reference:v1",
        output,
    )
    assert lock.lock_sha256 == lock.computed_sha256()
    assert lock.execution_provider == "CPUExecutionProvider"
    assert output.is_file()
    command = runner.commands[1]
    assert ("--network", "none") == command[
        command.index("--network") : command.index("--network") + 2
    ]
    assert "--gpus" not in command
    assert "--read-only" in command
    assert ("--entrypoint", "") == command[
        command.index("--entrypoint") : command.index("--entrypoint") + 2
    ]


def test_invalid_reference_probe_cleans_exact_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _resolve(monkeypatch)
    runner = QueueRunner(
        [
            _result(stdout="pulled"),
            _result(stdout="not-json"),
            _result(),
        ]
    )
    with pytest.raises(InfrastructureError, match="invalid evidence"):
        ReferenceEnvironmentLocker(runner, clock=lambda: FIXED_TIME).lock(
            "registry.example/reference:v1",
            tmp_path / "reference.lock.json",
        )
    run_name = runner.commands[1][runner.commands[1].index("--name") + 1]
    assert runner.commands[2] == ("docker", "container", "rm", "--force", run_name)


def test_reference_lock_refuses_overwrite_before_registry_access(tmp_path: Path) -> None:
    output = tmp_path / "reference.lock.json"
    output.write_text("retained", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="refusing to overwrite"):
        ReferenceEnvironmentLocker(QueueRunner([])).lock(
            "registry.example/reference:v1",
            output,
        )


def test_reference_pull_failure_is_infrastructure_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _resolve(monkeypatch)
    with pytest.raises(InfrastructureError, match="could not pull"):
        ReferenceEnvironmentLocker(QueueRunner([_result(returncode=1)])).lock(
            "registry.example/reference:v1",
            tmp_path / "reference.lock.json",
        )


def test_reference_runtime_failure_cleans_exact_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _resolve(monkeypatch)
    runner = QueueRunner([_result(), _result(returncode=1), _result()])
    with pytest.raises(InfrastructureError, match="probe failed"):
        ReferenceEnvironmentLocker(runner).lock(
            "registry.example/reference:v1",
            tmp_path / "reference.lock.json",
        )
    run_name = runner.commands[1][runner.commands[1].index("--name") + 1]
    assert runner.commands[2] == ("docker", "container", "rm", "--force", run_name)


def test_reference_probe_requires_cpu_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _resolve(monkeypatch)
    payload = json.loads(_probe())
    payload["providers"] = ["CUDAExecutionProvider"]
    runner = QueueRunner([_result(), _result(stdout=json.dumps(payload)), _result()])
    with pytest.raises(InfrastructureError, match="invalid evidence"):
        ReferenceEnvironmentLocker(runner).lock(
            "registry.example/reference:v1",
            tmp_path / "reference.lock.json",
        )
