"""TensorRT engine-build manifest."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord, ResultStatus
from upgrade_guard.contracts.environment import Sha256Digest


class BuildManifest(StrictModel):
    """Exact engine and plugin build evidence."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["BuildManifest"]
    id: str
    case_manifest_sha256: Sha256Digest
    environment_lock_sha256: Sha256Digest
    command: tuple[str, ...]
    command_sha256: Sha256Digest
    parser_warnings: tuple[str, ...]
    parser_errors: tuple[str, ...]
    builder_configuration: dict[str, str]
    plugin_source_sha256: Sha256Digest | None
    plugin_binary: ArtifactReference | None
    plugin_compile_command: tuple[str, ...] | None
    plugin_build_log: ArtifactReference | None
    timing_cache_mode: Literal["disabled", "cold", "warm_environment_local"]
    timing_cache: ArtifactReference | None
    started_at: AwareDatetime
    ended_at: AwareDatetime
    duration_seconds: float = Field(ge=0)
    engine: ArtifactReference | None
    engine_inspector: ArtifactReference | None
    engine_device_memory_bytes: int | None = Field(default=None, ge=0)
    engine_bytes: int | None = Field(default=None, ge=0)
    status: ResultStatus
    failure: FailureRecord | None
    warnings: tuple[str, ...]
