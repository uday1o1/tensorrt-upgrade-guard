"""Typed diagnosis for Docker GPU request failures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from upgrade_guard.containers.commands import Runner
from upgrade_guard.contracts.environment import (
    NvidiaContainerToolkitVersionAttempt,
    NvidiaContainerToolkitVersionObservation,
    NvidiaContainerToolkitVersionSource,
)
from upgrade_guard.errors import (
    InfrastructureError,
    UnsupportedEnvironmentError,
    UpgradeGuardError,
)

DockerGpuDiagnosisCode = Literal[
    "NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE",
    "NVIDIA_CONTAINER_RUNTIME_MISCONFIGURED",
    "DOCKER_GPU_REQUEST_FAILED",
]

_CDI_VENDOR_DISCOVERY_FAILURE = (
    "failed to discover gpu vendor from cdi",
    "no known gpu vendor found",
)
_NVIDIA_RUNTIME_FAILURE_MARKERS = (
    "could not select device driver",
    "unknown or invalid runtime name: nvidia",
    "unknown runtime specified nvidia",
    "unresolvable cdi devices",
    "nvidia-container-cli: initialization error",
    "nvidia-container-cli: mount error",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|bearer|token|password|passwd|secret|api[-_]?key)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@")
_TOOLKIT_VERSION_COMMANDS: tuple[
    tuple[NvidiaContainerToolkitVersionSource, tuple[str, ...]], ...
] = (
    ("nvidia-container-cli", ("nvidia-container-cli", "--version")),
    ("nvidia-ctk", ("nvidia-ctk", "--version")),
    ("nvidia-container-runtime", ("nvidia-container-runtime", "--version")),
    *(
        (
            "dpkg",
            (
                "dpkg-query",
                "--show",
                "--showformat=${binary:Package}=${Version}\\n",
                package,
            ),
        )
        for package in (
            "nvidia-container-toolkit",
            "nvidia-container-toolkit-base",
            "libnvidia-container1",
        )
    ),
    *(
        (
            "rpm",
            (
                "rpm",
                "--query",
                "--queryformat",
                "%{NAME}=%{VERSION}-%{RELEASE}\\n",
                package,
            ),
        )
        for package in (
            "nvidia-container-toolkit",
            "nvidia-container-toolkit-base",
            "libnvidia-container1",
        )
    ),
)


@dataclass(frozen=True, slots=True)
class DockerGpuDiagnosis:
    """Stable internal diagnosis layered under the public failure taxonomy."""

    code: DockerGpuDiagnosisCode
    outcome: Literal["unsupported", "infrastructure"]
    message: str


def observe_nvidia_container_toolkit_version(
    runner: Runner,
) -> NvidiaContainerToolkitVersionObservation:
    """Observe the host toolkit version without inferring it from Docker success."""

    attempts: list[NvidiaContainerToolkitVersionAttempt] = []
    for source, command in _TOOLKIT_VERSION_COMMANDS:
        try:
            result = runner.run(command, timeout_seconds=15)
        except InfrastructureError as error:
            attempts.append(
                NvidiaContainerToolkitVersionAttempt(
                    source=source,
                    command=command,
                    outcome="error",
                    detail=bounded_redacted_output(error.message, limit=256),
                )
            )
            continue
        output = (result.stdout.strip() or result.stderr.strip()).splitlines()
        if result.returncode == 0 and output and output[0].strip():
            version = bounded_redacted_output(output[0].strip(), limit=256)
            attempts.append(
                NvidiaContainerToolkitVersionAttempt(
                    source=source,
                    command=command,
                    outcome="observed",
                    returncode=0,
                    detail=version,
                )
            )
            return NvidiaContainerToolkitVersionObservation(
                status="observed",
                version=version,
                source=source,
                attempts=tuple(attempts),
            )
        detail = bounded_redacted_output(result.stderr, result.stdout, limit=256) or None
        attempts.append(
            NvidiaContainerToolkitVersionAttempt(
                source=source,
                command=command,
                outcome=(
                    "unavailable"
                    if _result_proves_source_unavailable(source, result.returncode, detail)
                    else "error"
                ),
                returncode=result.returncode,
                detail=detail or "version command failed without diagnostic output",
            )
        )
    return NvidiaContainerToolkitVersionObservation(
        status="unavailable",
        attempts=tuple(attempts),
    )


def diagnose_docker_gpu_failure(
    *,
    stdout: str,
    stderr: str,
    toolkit_observation: NvidiaContainerToolkitVersionObservation | None = None,
) -> DockerGpuDiagnosis:
    """Classify Docker GPU failures using output plus typed host evidence."""

    output = f"{stderr}\n{stdout}".casefold()
    cdi_vendor_failure = all(marker in output for marker in _CDI_VENDOR_DISCOVERY_FAILURE)
    runtime_failure = cdi_vendor_failure or any(
        marker in output for marker in _NVIDIA_RUNTIME_FAILURE_MARKERS
    )
    if cdi_vendor_failure and _all_toolkit_sources_unavailable(toolkit_observation):
        return DockerGpuDiagnosis(
            code="NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE",
            outcome="unsupported",
            message="NVIDIA Container Toolkit is unavailable on the Docker host",
        )
    if runtime_failure:
        return DockerGpuDiagnosis(
            code="NVIDIA_CONTAINER_RUNTIME_MISCONFIGURED",
            outcome="infrastructure",
            message="NVIDIA container runtime is misconfigured for Docker GPU requests",
        )
    return DockerGpuDiagnosis(
        code="DOCKER_GPU_REQUEST_FAILED",
        outcome="infrastructure",
        message="Docker GPU request failed",
    )


def docker_gpu_failure_error(
    fallback_message: str,
    *,
    stdout: str,
    stderr: str,
    toolkit_observation: NvidiaContainerToolkitVersionObservation | None = None,
    details: dict[str, Any] | None = None,
) -> UpgradeGuardError:
    """Build a typed public error without exposing unbounded process output."""

    diagnosis = diagnose_docker_gpu_failure(
        stdout=stdout,
        stderr=stderr,
        toolkit_observation=toolkit_observation,
    )
    error_details = dict(details or {})
    error_details.update(
        {
            "diagnosis_code": diagnosis.code,
            "error": bounded_redacted_output(stderr, stdout),
            "toolkit_evidence": _toolkit_evidence_status(toolkit_observation),
        }
    )
    if diagnosis.outcome == "unsupported":
        return UnsupportedEnvironmentError(diagnosis.message, details=error_details)
    message = fallback_message
    if diagnosis.code == "NVIDIA_CONTAINER_RUNTIME_MISCONFIGURED":
        message = diagnosis.message
    return InfrastructureError(message, details=error_details)


def bounded_redacted_output(*values: str, limit: int = 4000) -> str:
    """Return bounded diagnostic output with common credential forms redacted."""

    joined = "\n".join(value.strip() for value in values if value.strip())
    redacted = _URL_USERINFO.sub(r"\1<redacted>@", joined)
    redacted = _BEARER_CREDENTIAL.sub("<redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", redacted)
    return redacted[:limit]


def _all_toolkit_sources_unavailable(
    observation: NvidiaContainerToolkitVersionObservation | None,
) -> bool:
    return (
        observation is not None
        and observation.status == "unavailable"
        and bool(observation.attempts)
        and all(attempt.outcome == "unavailable" for attempt in observation.attempts)
        and tuple((attempt.source, attempt.command) for attempt in observation.attempts)
        == _TOOLKIT_VERSION_COMMANDS
    )


def _result_proves_source_unavailable(
    source: NvidiaContainerToolkitVersionSource,
    returncode: int,
    detail: str | None,
) -> bool:
    if returncode == 127:
        return True
    normalized = (detail or "").casefold()
    if source == "dpkg" and returncode == 1:
        return "no packages found matching" in normalized
    if source == "rpm" and returncode == 1:
        return "is not installed" in normalized or (
            "package " in normalized and "not installed" in normalized
        )
    return False


def _toolkit_evidence_status(
    observation: NvidiaContainerToolkitVersionObservation | None,
) -> Literal["unobserved", "observed", "unavailable", "inconclusive"]:
    if observation is None:
        return "unobserved"
    if observation.status == "observed":
        return "observed"
    if _all_toolkit_sources_unavailable(observation):
        return "unavailable"
    return "inconclusive"
