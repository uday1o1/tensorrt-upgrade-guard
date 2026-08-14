"""Host doctor parsing and outcome tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.doctor import doctor_exit_code, run_doctor
from upgrade_guard.errors import InfrastructureError


class FakeRunner:
    def __init__(self, results: Mapping[tuple[str, ...], CommandResult]) -> None:
        self.results = results

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        key = tuple(args)
        return self.results.get(key, command_result(key, returncode=127, stderr="missing"))


def command_result(
    args: Sequence[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> CommandResult:
    return CommandResult(tuple(args), returncode, stdout, stderr, 0.01)


def docker_results(
    *, architecture: str = "x86_64", runtimes: tuple[str, ...] = ("nvidia",)
) -> dict[tuple[str, ...], CommandResult]:
    context = ("docker", "context", "show")
    version = ("docker", "version", "--format", "{{json .}}")
    info = ("docker", "info", "--format", "{{json .}}")
    return {
        context: command_result(context, stdout="default\n"),
        version: command_result(
            version,
            stdout=json.dumps(
                {
                    "Client": {"Version": "29.0.0"},
                    "Server": {"Version": "29.0.0"},
                }
            ),
        ),
        info: command_result(
            info,
            stdout=json.dumps(
                {
                    "OSType": "linux",
                    "Architecture": architecture,
                    "Runtimes": {runtime: {} for runtime in runtimes},
                }
            ),
        ),
    }


def gpu_command() -> tuple[str, ...]:
    return (
        "nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )


def test_doctor_passes_supported_linux_host(monkeypatch: object) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.machine", lambda: "x86_64")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.release", lambda: "6.8.0")  # type: ignore[attr-defined]
    results = docker_results()
    results[gpu_command()] = command_result(
        gpu_command(),
        stdout=(
            "NVIDIA RTX Test, GPU-11111111-1111-1111-1111-111111111111, 8.9, 24576, 580.80.01\n"
        ),
    )
    result = run_doctor(FakeRunner(results))
    assert result.outcome == "supported"
    assert doctor_exit_code(result) == 0
    assert result.gpus[0].compute_capability == "8.9"


def test_doctor_allows_cdi_only_host_for_exact_worker_probe(monkeypatch: object) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.machine", lambda: "x86_64")  # type: ignore[attr-defined]
    results = docker_results(runtimes=("io.containerd.runc.v2", "runc"))
    results[gpu_command()] = command_result(
        gpu_command(),
        stdout=(
            "NVIDIA RTX Test, GPU-11111111-1111-1111-1111-111111111111, 8.9, 24576, 580.80.01\n"
        ),
    )
    result = run_doctor(FakeRunner(results))
    assert result.outcome == "supported"
    assert not result.issues


def test_doctor_fails_closed_on_macos_arm64(monkeypatch: object) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.machine", lambda: "arm64")  # type: ignore[attr-defined]
    results = docker_results(architecture="aarch64", runtimes=("runc",))
    results[gpu_command()] = command_result(gpu_command(), returncode=127, stderr="missing")
    result = run_doctor(FakeRunner(results))
    assert result.outcome == "unsupported"
    assert doctor_exit_code(result) == 3
    codes = {issue.code for issue in result.issues}
    assert {
        "HOST_OS_UNSUPPORTED",
        "HOST_ARCHITECTURE_UNSUPPORTED",
        "DOCKER_SERVER_ARCHITECTURE_UNSUPPORTED",
        "NVIDIA_GPU_UNAVAILABLE",
    } <= codes


def test_doctor_distinguishes_docker_infrastructure_failure(monkeypatch: object) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.machine", lambda: "x86_64")  # type: ignore[attr-defined]
    results = docker_results()
    docker_info = ("docker", "info", "--format", "{{json .}}")
    results[docker_info] = command_result(docker_info, returncode=1, stderr="daemon denied")
    results[gpu_command()] = command_result(
        gpu_command(),
        stdout=(
            "NVIDIA RTX Test, GPU-11111111-1111-1111-1111-111111111111, 8.9, 24576, 580.80.01\n"
        ),
    )
    result = run_doctor(FakeRunner(results))
    assert result.outcome == "infrastructure_invalid"
    assert doctor_exit_code(result) == 4
    assert result.issues[0].code == "DOCKER_UNAVAILABLE"


def test_doctor_rejects_invalid_gpu_output(monkeypatch: object) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.machine", lambda: "x86_64")  # type: ignore[attr-defined]
    results = docker_results()
    results[gpu_command()] = command_result(gpu_command(), stdout="not,a,valid,row\n")
    result = run_doctor(FakeRunner(results))
    assert result.outcome == "infrastructure_invalid"
    assert {issue.code for issue in result.issues} == {"NVIDIA_GPU_PROBE_INVALID"}


def test_doctor_reports_probe_timeouts_as_typed_infrastructure(monkeypatch: object) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")  # type: ignore[attr-defined]
    monkeypatch.setattr("platform.machine", lambda: "x86_64")  # type: ignore[attr-defined]

    class TimeoutRunner(FakeRunner):
        def run(self, args: Sequence[str], **kwargs: object) -> CommandResult:
            del kwargs
            if args[0] in {"docker", "nvidia-smi"}:
                raise InfrastructureError("command timed out")
            return command_result(args)

    result = run_doctor(TimeoutRunner({}))
    assert result.outcome == "infrastructure_invalid"
    assert doctor_exit_code(result) == 4
    assert {issue.code for issue in result.issues} == {
        "DOCKER_PROBE_TIMEOUT",
        "NVIDIA_GPU_PROBE_TIMEOUT",
    }
