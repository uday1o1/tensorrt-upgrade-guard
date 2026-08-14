"""Exact Docker GPU preflight script tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.factories import supported_doctor
from upgrade_guard.containers.commands import CommandResult

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_docker_gpu_runtime.py"
SPEC = importlib.util.spec_from_file_location("check_docker_gpu_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gpu_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_preflight)

GPU = "GPU-11111111-1111-1111-1111-111111111111"
IMAGE = "registry@sha256:" + "1" * 64
TARGET_ERROR = (
    "docker: Error response from daemon: failed to discover GPU vendor from CDI: "
    "no known GPU vendor found"
)


class FakeRunner:
    def __init__(self, *, toolkit_present: bool, create_error: str | None) -> None:
        self.toolkit_present = toolkit_present
        self.create_error = create_error
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
        stdout = ""
        stderr = ""
        returncode = 0
        if command == ("nvidia-ctk", "--version") and self.toolkit_present:
            stdout = "NVIDIA Container Toolkit CLI version 1.20.0\n"
        elif command[0] in {
            "nvidia-container-cli",
            "nvidia-ctk",
            "nvidia-container-runtime",
            "dpkg-query",
            "rpm",
        }:
            returncode = 127
            stderr = f"command not found: {command[0]}"
        elif command[:3] == ("docker", "container", "create") and self.create_error:
            returncode = 1
            stderr = self.create_error
        elif command[:3] == ("docker", "container", "inspect"):
            stdout = json.dumps(
                [
                    {
                        "Driver": "nvidia",
                        "Count": 0,
                        "DeviceIDs": [GPU],
                        "Capabilities": [["gpu"]],
                        "Options": {},
                    }
                ]
            )
        elif command[:3] == ("docker", "image", "inspect"):
            stdout = "{}"
        return CommandResult(command, returncode, stdout, stderr, 0.01)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    toolkit_present: bool,
    create_error: str | None,
) -> tuple[int, dict[str, object], FakeRunner]:
    runner = FakeRunner(toolkit_present=toolkit_present, create_error=create_error)
    output = tmp_path / "gpu-runtime-preflight.json"
    monkeypatch.setattr(gpu_preflight, "CommandRunner", lambda: runner)
    monkeypatch.setattr(gpu_preflight, "run_doctor", lambda ignored: supported_doctor())
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--gpu", GPU, "--image", IMAGE, "--output", str(output)],
    )
    status = gpu_preflight.main()
    return status, json.loads(output.read_text(encoding="utf-8")), runner


def test_exact_target_failure_without_toolkit_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, payload, runner = _run(
        monkeypatch,
        tmp_path,
        toolkit_present=False,
        create_error=TARGET_ERROR,
    )
    assert status == 3
    assert payload["status"] == "unsupported"
    assert payload["error_code"] == "PREFLIGHT_UNSUPPORTED"
    assert payload["details"]["diagnosis_code"] == "NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE"  # type: ignore[index]
    create = next(
        command for command in runner.commands if command[:3] == ("docker", "container", "create")
    )
    assert ("--gpus", f"device={GPU}") == create[
        create.index("--gpus") : create.index("--gpus") + 2
    ]
    assert ("--entrypoint", "/bin/true") == create[
        create.index("--entrypoint") : create.index("--entrypoint") + 2
    ]
    name = create[create.index("--name") + 1]
    assert runner.commands[-1] == ("docker", "container", "rm", "--force", name)


def test_exact_target_failure_with_toolkit_is_infrastructure_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, payload, _ = _run(
        monkeypatch,
        tmp_path,
        toolkit_present=True,
        create_error=TARGET_ERROR,
    )
    assert status == 4
    assert payload["status"] == "infrastructure_invalid"
    assert payload["error_code"] == "INFRASTRUCTURE_INVALID"
    assert payload["details"]["diagnosis_code"] == "NVIDIA_CONTAINER_RUNTIME_MISCONFIGURED"  # type: ignore[index]


def test_success_requires_exact_retained_uuid_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, payload, _ = _run(
        monkeypatch,
        tmp_path,
        toolkit_present=True,
        create_error=None,
    )
    assert status == 0
    assert payload["status"] == "passed"
    assert payload["gpu_request_verified"] is True
    assert payload["gpu_injection_interface"] == "docker-gpus"
