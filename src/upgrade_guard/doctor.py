"""Host preflight that never imports an NVIDIA runtime."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any, Literal

from upgrade_guard.containers.commands import CommandResult, CommandRunner, Runner
from upgrade_guard.contracts.doctor import DoctorDocker, DoctorGpu, DoctorIssue, DoctorResult


def run_doctor(runner: Runner | None = None) -> DoctorResult:
    """Inspect the host and fail closed before any model work."""

    command_runner = runner or CommandRunner()
    issues: list[DoctorIssue] = []
    system = platform.system()
    architecture = platform.machine()

    if system != "Linux":
        issues.append(
            DoctorIssue(
                code="HOST_OS_UNSUPPORTED",
                category="unsupported",
                message="final qualification requires a Linux x86-64 host",
                evidence=system,
            )
        )
    if architecture not in {"x86_64", "amd64"}:
        issues.append(
            DoctorIssue(
                code="HOST_ARCHITECTURE_UNSUPPORTED",
                category="unsupported",
                message="final qualification requires x86-64 host architecture",
                evidence=architecture,
            )
        )

    docker, docker_issues = _probe_docker(command_runner)
    issues.extend(docker_issues)
    gpus, gpu_issues = _probe_gpus(command_runner)
    issues.extend(gpu_issues)

    if docker.available and docker.server_os != "linux":
        issues.append(
            DoctorIssue(
                code="DOCKER_SERVER_OS_UNSUPPORTED",
                category="unsupported",
                message="Docker server must run Linux workers",
                evidence=docker.server_os,
            )
        )
    if docker.available and docker.server_architecture not in {"x86_64", "amd64"}:
        issues.append(
            DoctorIssue(
                code="DOCKER_SERVER_ARCHITECTURE_UNSUPPORTED",
                category="unsupported",
                message="Docker server must run linux/amd64 workers",
                evidence=docker.server_architecture,
            )
        )
    if docker.available and "nvidia" not in docker.runtimes and system == "Linux":
        issues.append(
            DoctorIssue(
                code="NVIDIA_CONTAINER_RUNTIME_UNCONFIRMED",
                category="infrastructure",
                message=(
                    "Docker does not report an NVIDIA runtime; matrix lock must prove GPU injection"
                ),
                evidence=",".join(docker.runtimes),
            )
        )

    outcome = _outcome(issues)
    return DoctorResult(
        schema_version="upgradeguard.dev/doctor/v1",
        outcome=outcome,
        host_os=system,
        host_release=platform.release(),
        host_architecture=architecture,
        python_version=platform.python_version(),
        docker=docker,
        gpus=tuple(gpus),
        issues=tuple(issues),
    )


def _probe_docker(runner: Runner) -> tuple[DoctorDocker, list[DoctorIssue]]:
    issues: list[DoctorIssue] = []
    context_result = runner.run(("docker", "context", "show"), timeout_seconds=10)
    version_result = runner.run(("docker", "version", "--format", "{{json .}}"), timeout_seconds=15)
    info_result = runner.run(("docker", "info", "--format", "{{json .}}"), timeout_seconds=15)
    if version_result.returncode != 0 or info_result.returncode != 0:
        evidence = _bounded_error(version_result, info_result)
        issues.append(
            DoctorIssue(
                code="DOCKER_UNAVAILABLE",
                category="infrastructure",
                message="Docker client cannot inspect a running daemon",
                evidence=evidence,
            )
        )
        return DoctorDocker(available=False), issues

    try:
        version = json.loads(version_result.stdout)
        info = json.loads(info_result.stdout)
        client_version = _nested_string(version, "Client", "Version")
        server_version = _nested_string(version, "Server", "Version")
        runtimes_value = info.get("Runtimes", {})
        runtimes = tuple(sorted(runtimes_value)) if isinstance(runtimes_value, dict) else ()
        return (
            DoctorDocker(
                available=True,
                client_version=client_version,
                server_version=server_version,
                server_os=_optional_string(info.get("OSType")),
                server_architecture=_optional_string(info.get("Architecture")),
                runtimes=runtimes,
                context=context_result.stdout.strip() if context_result.returncode == 0 else None,
            ),
            issues,
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        issues.append(
            DoctorIssue(
                code="DOCKER_PROBE_INVALID",
                category="infrastructure",
                message="Docker returned an invalid version or info document",
                evidence=str(error),
            )
        )
        return DoctorDocker(available=False), issues


def _probe_gpus(runner: Runner) -> tuple[list[DoctorGpu], list[DoctorIssue]]:
    result = runner.run(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        timeout_seconds=15,
    )
    if result.returncode != 0:
        return [], [
            DoctorIssue(
                code="NVIDIA_GPU_UNAVAILABLE",
                category="unsupported",
                message="no NVIDIA GPU is visible through nvidia-smi",
                evidence=(result.stderr or result.stdout).strip()[:500] or None,
            )
        ]

    gpus: list[DoctorGpu] = []
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            name, uuid, compute_capability, memory, driver = [
                field.strip() for field in line.split(",", maxsplit=4)
            ]
            gpus.append(
                DoctorGpu(
                    name=name,
                    uuid=uuid,
                    compute_capability=compute_capability,
                    vram_mib=int(memory),
                    driver_version=driver,
                )
            )
        if not gpus:
            raise ValueError("nvidia-smi returned no GPUs")
    except (TypeError, ValueError) as error:
        return [], [
            DoctorIssue(
                code="NVIDIA_GPU_PROBE_INVALID",
                category="infrastructure",
                message="nvidia-smi returned an invalid GPU inventory",
                evidence=str(error),
            )
        ]
    return gpus, []


def _outcome(
    issues: list[DoctorIssue],
) -> Literal["supported", "unsupported", "infrastructure_invalid"]:
    if any(issue.category == "unsupported" for issue in issues):
        return "unsupported"
    if issues:
        return "infrastructure_invalid"
    return "supported"


def _bounded_error(*results: CommandResult) -> str | None:
    combined = "\n".join(
        value
        for result in results
        for value in (result.stderr.strip(), result.stdout.strip())
        if value
    )
    return combined[:500] or None


def _nested_string(value: dict[str, Any], *keys: str) -> str:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            raise TypeError(f"{'.'.join(keys)} is not an object path")
        current = current[key]
    if not isinstance(current, str) or not current:
        raise TypeError(f"{'.'.join(keys)} is not a nonempty string")
    return current


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def doctor_exit_code(result: DoctorResult) -> int:
    """Map doctor outcome to the stable public exit contract."""

    if result.outcome == "supported":
        return 0
    if result.outcome == "unsupported":
        return 3
    return 4


def doctor_json(result: DoctorResult) -> str:
    """Return stable indented JSON for CLI output and stored evidence."""

    return result.model_dump_json(indent=2)


if __name__ == "__main__":  # pragma: no cover
    sys.stdout.write(doctor_json(run_doctor()) + "\n")
