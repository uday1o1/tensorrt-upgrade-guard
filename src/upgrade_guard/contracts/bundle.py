"""Hash-verified reproduction-bundle contract."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord
from upgrade_guard.contracts.environment import Sha256Digest


class SourceBuildRequest(StrictModel):
    """Source code that requires explicit operator trust before compilation."""

    sources: tuple[ArtifactReference, ...]
    worker_image_manifest_digest: Sha256Digest
    selected_gpu_uuid: str
    command: tuple[str, ...]


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
        required = {
            self.baseline_environment.path,
            self.candidate_environment.path,
            self.qualification.path,
            self.model.path,
            *(artifact.path for artifact in self.inputs),
        }
        if not required.issubset(paths):
            raise ValueError("bundle required artifacts must appear in the complete file inventory")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"manifest_sha256"})
