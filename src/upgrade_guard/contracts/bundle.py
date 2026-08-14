"""Hash-verified reproduction-bundle contract."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.contracts.matrix import GpuUuid

ExactImageReference = Annotated[
    str,
    StringConstraints(pattern=r"^[^\s@]+@sha256:[0-9a-f]{64}$", max_length=512),
]


class WorkerBuildArgument(StrictModel):
    """One reviewed Docker build argument without shell interpretation."""

    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    value: str = Field(min_length=1, max_length=1024)


class ReplayRequirements(StrictModel):
    """Portable target requirements independent of the original GPU identity."""

    platform: Literal["linux/amd64"] = "linux/amd64"
    minimum_compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    minimum_driver: str = Field(min_length=1, max_length=128)
    minimum_vram_mib: int = Field(gt=0)


def canonical_cmake_cuda_architecture(compute_capability: str) -> str:
    """Convert one canonical CUDA compute capability to CMake's numeric form."""

    match = re.fullmatch(r"([1-9][0-9]*)\.([0-9])", compute_capability)
    if match is None:
        raise ValueError("compute capability must be canonical major.minor")
    return f"{int(match.group(1))}{match.group(2)}"


class CudaArchitectureBuild(StrictModel):
    """Original GPU architecture bound to replay-time CUDA compilation."""

    original_compute_capability: str = Field(min_length=3, max_length=8)
    cmake_cuda_architecture: str = Field(pattern=r"^[1-9][0-9]+$")

    @model_validator(mode="after")
    def validate_conversion(self) -> CudaArchitectureBuild:
        if self.cmake_cuda_architecture != canonical_cmake_cuda_architecture(
            self.original_compute_capability
        ):
            raise ValueError("CMake CUDA architecture differs from compute capability")
        return self


def is_cmake_configure_command(command: tuple[str, ...]) -> bool:
    """Return whether an argument array invokes CMake configure mode."""

    if not command or command[0].rsplit("/", 1)[-1] != "cmake":
        return False
    nonconfigure = {"--build", "--install", "--open", "--help", "--version"}
    return not any(argument in nonconfigure for argument in command[1:])


def validate_cmake_cuda_command(
    command: tuple[str, ...], architecture: CudaArchitectureBuild | None
) -> None:
    """Require exact architecture binding on each CMake configure command."""

    if not is_cmake_configure_command(command):
        return
    if architecture is None:
        raise ValueError("CMake CUDA configure requires locked architecture evidence")
    authored = tuple(
        argument.removeprefix("-DCMAKE_CUDA_ARCHITECTURES=")
        for argument in command
        if argument.startswith("-DCMAKE_CUDA_ARCHITECTURES=")
    )
    if authored != (architecture.cmake_cuda_architecture,):
        raise ValueError("CMake CUDA configure command has wrong or ambiguous architecture")


class LocalWorkerBuild(StrictModel):
    """Hash-bound recipe for rebuilding the worker without a published worker image."""

    base_image: ExactImageReference
    base_image_manifest_digest: Sha256Digest
    dockerfile: ArtifactReference
    worker_lock: ArtifactReference
    build_arguments: tuple[WorkerBuildArgument, ...]
    cuda_architecture: CudaArchitectureBuild | None = None

    @model_validator(mode="after")
    def validate_recipe(self) -> LocalWorkerBuild:
        if not self.base_image.endswith(f"@{self.base_image_manifest_digest}"):
            raise ValueError("worker rebuild base reference and manifest digest differ")
        arguments = {argument.name: argument.value for argument in self.build_arguments}
        if len(arguments) != len(self.build_arguments):
            raise ValueError("worker rebuild arguments must have unique names")
        required = {
            "BASE_IMAGE": self.base_image,
            "BASE_MANIFEST_DIGEST": self.base_image_manifest_digest,
        }
        if any(arguments.get(name) != value for name, value in required.items()):
            raise ValueError("worker rebuild arguments must bind the exact base image")
        return self

    def computed_sha256(self) -> str:
        """Hash the complete local worker rebuild recipe."""

        return model_sha256(self)


class SourceBuildRequest(StrictModel):
    """Source code that requires explicit operator trust before compilation."""

    sources: tuple[ArtifactReference, ...]
    original_worker_image_manifest_digest: Sha256Digest
    original_gpu_uuid: GpuUuid
    replay_requirements: ReplayRequirements
    cuda_architecture: CudaArchitectureBuild | None = None
    local_worker_build: LocalWorkerBuild
    command: tuple[str, ...]

    @model_validator(mode="after")
    def validate_source_build(self) -> SourceBuildRequest:
        if not self.sources:
            raise ValueError("source build requests require source files")
        if not self.command or any(not argument or "\x00" in argument for argument in self.command):
            raise ValueError("source build command must be a nonempty NUL-free argument array")
        validate_cmake_cuda_command(self.command, self.cuda_architecture)
        if self.local_worker_build.cuda_architecture != self.cuda_architecture:
            raise ValueError("source and worker rebuild CUDA architectures differ")
        return self


class BundleManifest(StrictModel):
    """Typed reproduction contents, limits, and expected predicate."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["ReproductionBundle"]
    id: str
    created_at: AwareDatetime
    files: tuple[ArtifactReference, ...]
    baseline_environment: ArtifactReference
    candidate_environment: ArtifactReference
    qualification: ArtifactReference
    expected_failure: FailureRecord
    model: ArtifactReference
    inputs: tuple[ArtifactReference, ...]
    source_build: SourceBuildRequest | None
    included_engine: ArtifactReference | None
    file_count_limit: int = Field(default=512, ge=1, le=4096)
    expanded_size_limit_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=1,
        le=64 * 1024 * 1024 * 1024,
    )
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_file_inventory(self) -> BundleManifest:
        paths = [artifact.path for artifact in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle file inventory cannot contain duplicate paths")
        if len(paths) > self.file_count_limit:
            raise ValueError("bundle file inventory exceeds file_count_limit")
        if sum(artifact.bytes for artifact in self.files) > self.expanded_size_limit_bytes:
            raise ValueError("bundle file inventory exceeds expanded_size_limit_bytes")
        required_artifacts = [
            self.baseline_environment,
            self.candidate_environment,
            self.qualification,
            self.model,
            *self.inputs,
        ]
        if self.source_build is not None:
            required_artifacts.extend(self.source_build.sources)
            required_artifacts.append(self.source_build.local_worker_build.dockerfile)
            required_artifacts.append(self.source_build.local_worker_build.worker_lock)
        if self.included_engine is not None:
            required_artifacts.append(self.included_engine)
        inventory = {artifact.path: artifact for artifact in self.files}
        if any(inventory.get(artifact.path) != artifact for artifact in required_artifacts):
            raise ValueError("bundle required artifacts must appear in the complete file inventory")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"manifest_sha256"})
