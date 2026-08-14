"""Typed Docker GPU runtime failure diagnosis tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.gpu_runtime import (
    bounded_redacted_output,
    diagnose_docker_gpu_failure,
    docker_gpu_failure_error,
    observe_nvidia_container_toolkit_version,
)
from upgrade_guard.contracts.environment import (
    NvidiaContainerToolkitVersionAttempt,
    NvidiaContainerToolkitVersionObservation,
)
from upgrade_guard.errors import InfrastructureError, UnsupportedEnvironmentError

TARGET_ERROR = (
    "docker: Error response from daemon: failed to discover GPU vendor from CDI: "
    "no known GPU vendor found"
)


def unavailable_toolkit(*, inconclusive: bool = False) -> NvidiaContainerToolkitVersionObservation:
    class Runner:
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
            if inconclusive and command[0] == "nvidia-container-cli":
                return CommandResult(command, 126, "", "permission denied", 0.1)
            return CommandResult(command, 127, "", "command not found", 0.1)

    return observe_nvidia_container_toolkit_version(Runner())


def observed_toolkit() -> NvidiaContainerToolkitVersionObservation:
    attempt = NvidiaContainerToolkitVersionAttempt(
        source="nvidia-ctk",
        command=("nvidia-ctk", "--version"),
        outcome="observed",
        returncode=0,
        detail="NVIDIA Container Toolkit CLI version 1.18.1",
    )
    return NvidiaContainerToolkitVersionObservation(
        status="observed",
        version=attempt.detail,
        source="nvidia-ctk",
        attempts=(attempt,),
    )


def test_exact_cdi_failure_with_unavailable_toolkit_is_unsupported() -> None:
    error = docker_gpu_failure_error(
        "worker failed",
        stdout="",
        stderr=TARGET_ERROR,
        toolkit_observation=unavailable_toolkit(),
    )
    assert isinstance(error, UnsupportedEnvironmentError)
    assert error.error_code == "PREFLIGHT_UNSUPPORTED"
    assert error.details["diagnosis_code"] == "NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE"
    assert error.details["toolkit_evidence"] == "unavailable"


@pytest.mark.parametrize(
    "observation,evidence",
    [
        (observed_toolkit(), "observed"),
        (None, "unobserved"),
        (unavailable_toolkit(inconclusive=True), "inconclusive"),
    ],
)
def test_exact_cdi_failure_without_complete_absence_evidence_is_infrastructure(
    observation: NvidiaContainerToolkitVersionObservation | None,
    evidence: str,
) -> None:
    error = docker_gpu_failure_error(
        "worker failed",
        stdout="",
        stderr=TARGET_ERROR,
        toolkit_observation=observation,
    )
    assert isinstance(error, InfrastructureError)
    assert error.error_code == "INFRASTRUCTURE_INVALID"
    assert error.details["diagnosis_code"] == "NVIDIA_CONTAINER_RUNTIME_MISCONFIGURED"
    assert error.details["toolkit_evidence"] == evidence


def test_unrelated_container_failure_keeps_generic_infrastructure_diagnosis() -> None:
    diagnosis = diagnose_docker_gpu_failure(
        stdout="application output",
        stderr="process exited with status 17",
        toolkit_observation=unavailable_toolkit(),
    )
    assert diagnosis.code == "DOCKER_GPU_REQUEST_FAILED"
    assert diagnosis.outcome == "infrastructure"


def test_failure_output_is_bounded_and_redacts_credentials() -> None:
    output = bounded_redacted_output(
        "Authorization: Bearer abc123 password=hunter2",
        "https://user:pass@example.test/path",
        "x" * 5000,
        limit=200,
    )
    assert len(output) == 200
    assert "abc123" not in output
    assert "hunter2" not in output
    assert "user:pass" not in output
    assert "<redacted>" in output


def test_toolkit_observation_uses_first_truthful_version_source() -> None:
    class Runner:
        def __init__(self) -> None:
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
            if command[0] == "nvidia-container-cli":
                return CommandResult(command, 127, "", "command not found", 0.1)
            return CommandResult(command, 0, "NVIDIA Container Toolkit CLI 1.18.1\n", "", 0.1)

    runner = Runner()
    observation = observe_nvidia_container_toolkit_version(runner)
    assert observation.status == "observed"
    assert observation.source == "nvidia-ctk"
    assert observation.version == "NVIDIA Container Toolkit CLI 1.18.1"
    assert len(observation.attempts) == 2


def test_toolkit_observation_requires_every_source_before_unavailable() -> None:
    class Runner:
        def __init__(self) -> None:
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
            return CommandResult(command, 127, "", "command not found", 0.1)

    runner = Runner()
    observation = observe_nvidia_container_toolkit_version(runner)
    assert observation.status == "unavailable"
    assert len(observation.attempts) == 9
    assert all(attempt.outcome == "unavailable" for attempt in observation.attempts)


def test_toolkit_permission_failure_is_inconclusive_not_unavailable() -> None:
    class Runner:
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
            return CommandResult(command, 126, "", "permission denied", 0.1)

    observation = observe_nvidia_container_toolkit_version(Runner())
    assert observation.status == "unavailable"
    assert all(attempt.outcome == "error" for attempt in observation.attempts)
    error = docker_gpu_failure_error(
        "worker failed",
        stdout="",
        stderr=TARGET_ERROR,
        toolkit_observation=observation,
    )
    assert isinstance(error, InfrastructureError)
    assert error.details["toolkit_evidence"] == "inconclusive"
