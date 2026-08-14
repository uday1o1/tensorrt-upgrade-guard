"""Transactional environment matrix locking."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

import yaml
from pydantic import ValidationError

from upgrade_guard.containers.commands import CommandRunner, Runner
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.doctor import DoctorResult
from upgrade_guard.contracts.environment import (
    EnvironmentLock,
    HostObservation,
    MatrixLock,
)
from upgrade_guard.contracts.matrix import MatrixSpec
from upgrade_guard.doctor import run_doctor
from upgrade_guard.errors import InfrastructureError, InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.matrix.compatibility import evaluate_compatibility
from upgrade_guard.matrix.digest import (
    RegistryClient,
    ResolvedArtifact,
    credentials_from_environment,
    parse_image_reference,
)
from upgrade_guard.matrix.probe import DockerWorkerProbe, ProbeExecution

WORKER_BASE_DIGEST_LABEL = "com.udayarora.upgradeguard.base.manifest.digest"

_TOOLKIT_VERSION_COMMANDS = (
    ("nvidia-container-cli", "--version"),
    ("nvidia-ctk", "--version"),
    ("nvidia-container-runtime", "--version"),
    *(
        (
            "dpkg-query",
            "--show",
            "--showformat=${binary:Package}=${Version}\\n",
            package,
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
            "--query",
            "--queryformat",
            "%{NAME}=%{VERSION}-%{RELEASE}\\n",
            package,
        )
        for package in (
            "nvidia-container-toolkit",
            "nvidia-container-toolkit-base",
            "libnvidia-container1",
        )
    ),
)


class ImageResolver(Protocol):
    """Immutable image resolver interface."""

    def resolve_linux_amd64(self, authored_reference: str) -> ResolvedArtifact: ...


class WorkerProber(Protocol):
    """Exact worker probe interface."""

    def run(self, image: object, gpu_uuid: str) -> ProbeExecution: ...


class EnvironmentResolver:
    """Create registry clients with credentials scoped to each authored registry."""

    def resolve_linux_amd64(self, authored_reference: str) -> ResolvedArtifact:
        parts = parse_image_reference(authored_reference)
        credentials = credentials_from_environment(parts.registry)
        return RegistryClient(credentials=credentials).resolve_linux_amd64(authored_reference)


class MatrixLocker:
    """Resolve, probe, validate, and atomically write a complete matrix lock."""

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        resolver: ImageResolver | None = None,
        prober: WorkerProber | None = None,
        doctor: Callable[[], DoctorResult] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.resolver = resolver or EnvironmentResolver()
        self.prober = prober or DockerWorkerProbe(self.runner)
        self.doctor = doctor or (lambda: run_doctor(self.runner))
        self.clock = clock or (lambda: datetime.now(UTC))

    def lock(self, matrix_path: Path, output_path: Path) -> MatrixLock:
        """Build a lock and publish it only after every gate succeeds."""

        if output_path.exists() or output_path.is_symlink():
            raise InvalidInputError(f"refusing to overwrite existing lock: {output_path}")
        source = _read_matrix(matrix_path)
        specification = _parse_matrix(source)
        lock = self.build(specification, source_sha256=sha256_bytes(source))
        _write_atomic(output_path, lock.model_dump_json(indent=2) + "\n")
        return lock

    def build(self, specification: MatrixSpec, *, source_sha256: str) -> MatrixLock:
        """Build an in-memory lock from real host and container observations."""

        doctor = self.doctor()
        if doctor.outcome == "unsupported":
            raise UnsupportedEnvironmentError(
                "host does not satisfy the qualification boundary",
                details={"issues": [issue.model_dump(mode="json") for issue in doctor.issues]},
            )
        if doctor.outcome != "supported":
            raise InfrastructureError(
                "host preflight is inconclusive",
                details={"issues": [issue.model_dump(mode="json") for issue in doctor.issues]},
            )
        selected = [gpu for gpu in doctor.gpus if gpu.uuid == specification.gpu_uuid]
        if len(selected) != 1:
            raise UnsupportedEnvironmentError(
                "selected GPU UUID is not uniquely visible on the host",
                details={"selected_gpu_uuid": specification.gpu_uuid},
            )

        host = _host_observation(doctor, self.runner)
        environments: list[EnvironmentLock] = []
        for request in specification.environments:
            base = self.resolver.resolve_linux_amd64(request.base_image)
            worker = self.resolver.resolve_linux_amd64(request.worker_image)
            declared_base = worker.label(WORKER_BASE_DIGEST_LABEL)
            if declared_base != base.image.manifest_digest:
                raise InvalidInputError(
                    "final worker image does not declare the resolved base manifest",
                    details={
                        "environment": request.id,
                        "label": WORKER_BASE_DIGEST_LABEL,
                        "expected": base.image.manifest_digest,
                        "observed": declared_base,
                    },
                )
            execution = self.prober.run(worker.image, specification.gpu_uuid)
            if execution.probe.observed_driver != selected[0].driver_version:
                raise InfrastructureError(
                    "worker and host observed different NVIDIA driver versions",
                    details={
                        "host": selected[0].driver_version,
                        "worker": execution.probe.observed_driver,
                    },
                )
            compatibility = evaluate_compatibility(
                execution.probe,
                request.capabilities,
                expected_gpu_uuid=specification.gpu_uuid,
            )
            if not compatibility.compatible:
                raise UnsupportedEnvironmentError(
                    f"environment {request.id} is incompatible",
                    details={"reasons": list(compatibility.reasons)},
                )
            environments.append(
                EnvironmentLock(
                    id=request.id,
                    base_image=base.image,
                    worker_image=worker.image,
                    declared_base_manifest_digest=declared_base,
                    probe=execution.probe,
                    host=host,
                    compatibility=compatibility,
                    probe_command_sha256=execution.command_sha256,
                    probe_output_sha256=execution.output_sha256,
                    probed_at=self.clock(),
                )
            )
        if len(environments) != 2:
            raise InvalidInputError("matrix lock requires exactly two environments")
        _validate_pair(environments)
        zero_checksum = "sha256:" + ("0" * 64)
        lock = MatrixLock(
            api_version="upgradeguard.dev/v1alpha1",
            kind="EnvironmentLock",
            source_matrix_sha256=source_sha256,
            gpu_uuid=specification.gpu_uuid,
            created_at=self.clock(),
            environments=(environments[0], environments[1]),
            lock_sha256=zero_checksum,
        )
        return lock.model_copy(update={"lock_sha256": lock.computed_sha256()})


def _read_matrix(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise InvalidInputError(f"could not read matrix: {path}") from error


def _parse_matrix(source: bytes) -> MatrixSpec:
    try:
        value = yaml.safe_load(source)
        return MatrixSpec.model_validate(value)
    except (yaml.YAMLError, UnicodeDecodeError, ValidationError) as error:
        raise InvalidInputError(
            "matrix does not satisfy the strict schema",
            details={"reason": str(error)},
        ) from error


def _host_observation(doctor: DoctorResult, runner: Runner) -> HostObservation:
    docker = doctor.docker
    if not docker.available or not docker.client_version or not docker.server_version:
        raise InfrastructureError("Docker version evidence is incomplete")
    toolkit_version = _nvidia_container_toolkit_version(runner)
    runtime = "nvidia" if "nvidia" in docker.runtimes else "cdi"
    architecture = cast(Literal["x86_64", "amd64"], doctor.host_architecture)
    return HostObservation(
        operating_system=_host_operating_system(),
        kernel=platform.release(),
        architecture=architecture,
        docker_client_version=docker.client_version,
        docker_server_version=docker.server_version,
        docker_runtime=runtime,
        nvidia_container_toolkit_version=toolkit_version,
    )


def _nvidia_container_toolkit_version(runner: Runner) -> str:
    attempts: list[dict[str, object]] = []
    for command in _TOOLKIT_VERSION_COMMANDS:
        try:
            result = runner.run(command, timeout_seconds=15)
        except InfrastructureError as error:
            attempts.append({"command": list(command), "error": error.message})
            continue
        output = (result.stdout.strip() or result.stderr.strip()).splitlines()
        if result.returncode == 0 and output and output[0].strip():
            return output[0].strip()[:256]
        attempts.append(
            {
                "command": list(command),
                "returncode": result.returncode,
                "error": (result.stderr or result.stdout).strip()[:256],
            }
        )
    raise InfrastructureError(
        "NVIDIA Container Toolkit version could not be observed",
        details={"attempts": attempts},
    )


def _host_operating_system() -> str:
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.partition("=")[2].strip().strip('"')
    return platform.system()


def _validate_pair(environments: list[EnvironmentLock]) -> None:
    first, second = environments
    if first.probe.gpu != second.probe.gpu:
        raise InfrastructureError(
            "baseline and candidate workers observed different GPU properties"
        )
    if first.probe.observed_driver != second.probe.observed_driver:
        raise InfrastructureError("baseline and candidate workers observed different host drivers")
    if first.host != second.host:
        raise InfrastructureError("baseline and candidate workers did not share one host lock")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise InfrastructureError(f"could not atomically write lock: {path}") from error


def lock_json(lock: MatrixLock) -> str:
    """Render the exact stored representation."""

    return json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=False)
