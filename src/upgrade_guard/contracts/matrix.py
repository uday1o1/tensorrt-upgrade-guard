"""Authored environment-matrix contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from upgrade_guard.contracts.base import StrictModel

EnvironmentId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$", min_length=1, max_length=64),
]
ImageReference = Annotated[str, StringConstraints(min_length=1, max_length=512)]
GpuUuid = Annotated[str, StringConstraints(pattern=r"^GPU-[A-Fa-f0-9-]+$")]


class CapabilityPolicy(StrictModel):
    """Tools required to complete the full V1 on a locked worker pair."""

    trtexec: Literal[True] = True
    cxx_compiler: Literal[True] = True
    cmake: Literal[True] = True
    ninja: Literal[True] = True
    cuda_compiler: Literal[True] = True
    cuda_headers: Literal[True] = True
    tensorrt_headers: Literal[True] = True
    compute_sanitizer: Literal[True] = True
    nsight_systems: Literal[True] = True
    nsight_compute: Literal[True] = True


class EnvironmentRequest(StrictModel):
    """One authored environment before immutable resolution."""

    id: EnvironmentId
    base_image: ImageReference
    worker_image: ImageReference
    platform: Literal["linux/amd64"] = "linux/amd64"
    capabilities: CapabilityPolicy = Field(default_factory=CapabilityPolicy)

    @model_validator(mode="after")
    def require_separate_worker_identity(self) -> EnvironmentRequest:
        """A base image cannot masquerade as a derived final worker."""

        if self.base_image == self.worker_image:
            raise ValueError("worker_image must identify a separately derived final worker image")
        return self


class MatrixSpec(StrictModel):
    """Human-authored baseline and candidate matrix."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["EnvironmentMatrix"]
    gpu_uuid: GpuUuid
    environments: tuple[EnvironmentRequest, ...]

    @model_validator(mode="after")
    def validate_pair(self) -> MatrixSpec:
        """Require exactly one ordered baseline and candidate pair."""

        if len(self.environments) != 2:
            raise ValueError("exactly two ordered environments are required")
        identifiers = [environment.id for environment in self.environments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("environment IDs must be unique")
        return self
