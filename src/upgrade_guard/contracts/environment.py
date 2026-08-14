"""Immutable environment and matrix lock contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.matrix import EnvironmentId, GpuUuid

Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class PlatformIdentity(StrictModel):
    """OCI platform selected for a worker."""

    os: Literal["linux"]
    architecture: Literal["amd64"]
    variant: str | None = None


class ResolvedImage(StrictModel):
    """Byte-verified identities for one authored OCI image reference."""

    authored_reference: str
    registry: str
    repository: str
    authored_tag: str | None
    requested_digest: Sha256Digest | None
    index_digest: Sha256Digest | None
    manifest_digest: Sha256Digest
    config_digest: Sha256Digest
    manifest_media_type: str
    config_media_type: str
    platform: PlatformIdentity

    @property
    def canonical_reference(self) -> str:
        """Return the exact selected manifest reference used for pull and run."""

        return f"{self.registry}/{self.repository}@{self.manifest_digest}"


class ToolObservation(StrictModel):
    """One observed executable and its version evidence."""

    available: bool
    path: str | None = None
    version: str | None = None
    sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def require_available_details(self) -> ToolObservation:
        if self.available and (self.path is None or self.version is None):
            raise ValueError("available tools require path and version")
        if not self.available and any((self.path, self.version, self.sha256)):
            raise ValueError("unavailable tools cannot contain observed details")
        return self


class TrtexecObservation(ToolObservation):
    """Observed trtexec executable and exact option inventory."""

    help_sha256: Sha256Digest | None = None
    options: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_help_evidence(self) -> TrtexecObservation:
        if self.available and (self.help_sha256 is None or not self.options):
            raise ValueError("available trtexec requires help hash and option inventory")
        return self


class GpuObservation(StrictModel):
    """GPU identity captured by a worker probe."""

    name: str
    uuid: GpuUuid
    compute_capability: str
    vram_mib: int = Field(gt=0)
    vbios_version: str | None
    power_limit_watts: float | None = Field(default=None, gt=0)


NvidiaContainerToolkitVersionSource = Literal[
    "nvidia-container-cli",
    "nvidia-ctk",
    "nvidia-container-runtime",
    "dpkg",
    "rpm",
]


class NvidiaContainerToolkitVersionAttempt(StrictModel):
    """One bounded host-side toolkit version observation attempt."""

    source: NvidiaContainerToolkitVersionSource
    command: tuple[str, ...]
    outcome: Literal["observed", "unavailable", "error"]
    returncode: int | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> NvidiaContainerToolkitVersionAttempt:
        if not self.command or any(not component for component in self.command):
            raise ValueError("toolkit version attempts require a complete command")
        if self.outcome == "observed" and not self.detail:
            raise ValueError("observed toolkit version attempts require evidence")
        if self.outcome == "error" and not self.detail:
            raise ValueError("failed toolkit version attempts require an error detail")
        return self


class NvidiaContainerToolkitVersionObservation(StrictModel):
    """Truthful host provenance for a toolkit version, when observable."""

    status: Literal["observed", "unavailable"]
    version: str | None = None
    source: NvidiaContainerToolkitVersionSource | None = None
    attempts: tuple[NvidiaContainerToolkitVersionAttempt, ...]

    @model_validator(mode="after")
    def validate_observation(self) -> NvidiaContainerToolkitVersionObservation:
        if not self.attempts:
            raise ValueError("toolkit version observation requires at least one attempt")
        observed = [attempt for attempt in self.attempts if attempt.outcome == "observed"]
        if self.status == "observed":
            if not self.version or self.source is None:
                raise ValueError("observed toolkit versions require a version and source")
            if len(observed) != 1 or observed[0].source != self.source:
                raise ValueError("observed toolkit version must match exactly one attempt")
        elif self.version is not None or self.source is not None or observed:
            raise ValueError("unavailable toolkit versions cannot contain observed details")
        return self


class HostObservation(StrictModel):
    """Host evidence that is shared by the locked worker pair."""

    operating_system: str
    kernel: str
    architecture: Literal["x86_64", "amd64"]
    docker_client_version: str
    docker_server_version: str
    docker_runtime_inventory: tuple[str, ...]
    gpu_injection_interface: Literal["docker-gpus"]
    gpu_injection_verified: Literal[True]
    nvidia_container_toolkit_version: NvidiaContainerToolkitVersionObservation


class WorkerProbe(StrictModel):
    """Version inventory emitted from one exact worker container."""

    schema_version: Literal["upgradeguard.dev/worker-probe/v1"]
    image_manifest_digest: Sha256Digest
    gpu: GpuObservation
    observed_driver: str
    cuda_runtime: str
    cuda_toolkit: str
    tensorrt: str
    python: str
    polygraphy: str
    onnx: str
    onnxruntime: str
    operating_system: str
    kernel: str
    trtexec: TrtexecObservation
    compute_sanitizer: ToolObservation
    nsight_systems: ToolObservation
    nsight_compute: ToolObservation
    c_compiler: ToolObservation
    cxx_compiler: ToolObservation
    cuda_compiler: ToolObservation
    cmake: ToolObservation
    ninja: ToolObservation
    cuda_headers: tuple[str, ...]
    tensorrt_headers: tuple[str, ...]


class CompatibilityEvidence(StrictModel):
    """Versioned policy result for one environment."""

    policy_version: str
    source_urls: tuple[str, ...]
    checked_at: AwareDatetime
    minimum_driver: str
    minimum_compute_capability: str
    compatible: bool
    reasons: tuple[str, ...]


class EnvironmentLock(StrictModel):
    """Complete immutable identity and observations for one worker."""

    id: EnvironmentId
    base_image: ResolvedImage
    worker_image: ResolvedImage
    declared_base_manifest_digest: Sha256Digest
    probe: WorkerProbe
    host: HostObservation
    compatibility: CompatibilityEvidence
    probe_command_sha256: Sha256Digest
    probe_output_sha256: Sha256Digest
    probed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_provenance(self) -> EnvironmentLock:
        if self.declared_base_manifest_digest != self.base_image.manifest_digest:
            raise ValueError("worker base provenance does not match the resolved base manifest")
        if self.probe.image_manifest_digest != self.worker_image.manifest_digest:
            raise ValueError("probe image identity does not match the resolved worker manifest")
        return self


class MatrixLock(StrictModel):
    """Atomic lock for one real host, GPU, baseline, and candidate pair."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["EnvironmentLock"]
    source_matrix_sha256: Sha256Digest
    gpu_uuid: GpuUuid
    created_at: AwareDatetime
    environments: tuple[EnvironmentLock, EnvironmentLock]
    lock_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_shared_gpu(self) -> MatrixLock:
        observed = {environment.probe.gpu.uuid for environment in self.environments}
        if observed != {self.gpu_uuid}:
            raise ValueError("both workers must observe the selected GPU UUID")
        return self

    def computed_sha256(self) -> str:
        """Hash every lock field except the checksum field itself."""

        return model_sha256(self, exclude={"lock_sha256"})
