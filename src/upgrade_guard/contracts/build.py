"""TensorRT engine-build manifest."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from upgrade_guard.classify import status_for_failure
from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord, ResultStatus
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode


class WorkerBuildMessage(StrictModel):
    """One bounded TensorRT logger observation."""

    severity: str
    message: str


class WorkerBuildArtifact(StrictModel):
    """One path and identity emitted inside the isolated worker."""

    path: str
    sha256: Sha256Digest
    bytes: int | None = Field(default=None, ge=0)


class WorkerEngineArtifact(WorkerBuildArtifact):
    """Serialized engine identity and gateable memory requirement."""

    bytes: int = Field(ge=0)
    device_memory_bytes: int = Field(ge=0)


class WorkerTimingCacheArtifact(StrictModel):
    """Environment-local timing-cache transition."""

    path: str
    input_sha256: Sha256Digest | None
    output_sha256: Sha256Digest
    bytes: int = Field(ge=0)


class WorkerBuildResult(StrictModel):
    """Strict production output of ``worker.build_engine``."""

    schema_version: Literal["upgradeguard.dev/worker-build/v1"]
    status: Literal["passed", "failed"]
    command: tuple[str, ...]
    command_sha256: Sha256Digest
    model: WorkerBuildArtifact | None = None
    engine: WorkerEngineArtifact | None = None
    memory_diagnostics: dict[str, Any] | None = None
    inspector: WorkerBuildArtifact | None = None
    timing_cache: WorkerTimingCacheArtifact | None = None
    parser_errors: tuple[str, ...] = ()
    builder_messages: tuple[WorkerBuildMessage, ...] = ()
    builder_warnings: tuple[WorkerBuildMessage, ...] = ()
    builder_configuration: dict[str, str] = Field(default_factory=dict)
    timing_cache_state: Literal["cold", "warm"] | None = None
    tensorrt_version: str | None = None
    started_unix_seconds: float | None = None
    ended_unix_seconds: float | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    strongly_typed: Literal[True] | None = None
    failure_code: FailureCode | None = None
    error_type: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> WorkerBuildResult:
        success = (
            self.model,
            self.engine,
            self.memory_diagnostics,
            self.inspector,
            self.timing_cache,
            self.timing_cache_state,
            self.tensorrt_version,
            self.started_unix_seconds,
            self.ended_unix_seconds,
            self.duration_seconds,
            self.strongly_typed,
        )
        if self.status == "passed" and (any(item is None for item in success) or self.failure_code):
            raise ValueError("passing worker builds require complete success evidence")
        if self.status == "failed" and (
            self.failure_code is None
            or self.error_type is None
            or self.message is None
            or self.started_unix_seconds is None
            or self.ended_unix_seconds is None
            or self.duration_seconds is None
        ):
            raise ValueError("failed worker builds require typed failure evidence")
        timestamps = (
            self.started_unix_seconds,
            self.ended_unix_seconds,
            self.duration_seconds,
        )
        if all(value is not None for value in timestamps):
            started, ended, duration = timestamps
            assert started is not None and ended is not None and duration is not None
            if (
                not all(math.isfinite(value) for value in (started, ended, duration))
                or ended < started
                or abs(duration - (ended - started)) > 1e-6
            ):
                raise ValueError("worker build timestamps and duration must agree")
        return self


class BuildManifestAdapterContext(StrictModel):
    """Host-owned identities needed to promote worker evidence."""

    id: str
    case_manifest_sha256: Sha256Digest
    environment_lock_sha256: Sha256Digest
    plugin_source_sha256: Sha256Digest | None = None
    plugin_binary: ArtifactReference | None = None
    plugin_compile_command: tuple[str, ...] | None = None
    plugin_build_log: ArtifactReference | None = None
    failure: FailureRecord | None = None


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

    @model_validator(mode="after")
    def require_failure_for_failed_status(self) -> BuildManifest:
        if (self.status is ResultStatus.PASSED) != (self.failure is None):
            raise ValueError("build status and failure record must agree")
        elapsed = (self.ended_at - self.started_at).total_seconds()
        if elapsed < 0 or abs(self.duration_seconds - elapsed) > 1e-5:
            raise ValueError("build manifest timestamps and duration must agree")
        return self


def adapt_worker_build(
    worker: WorkerBuildResult, context: BuildManifestAdapterContext
) -> BuildManifest:
    """Promote strict worker evidence into the stable build manifest."""

    if worker.status == "passed":
        assert worker.engine is not None
        assert worker.inspector is not None
        assert worker.timing_cache is not None
        assert worker.started_unix_seconds is not None
        assert worker.ended_unix_seconds is not None
        assert worker.duration_seconds is not None
        assert worker.timing_cache_state is not None
    status = status_for_failure(worker.failure_code)
    if worker.status == "failed" and (
        context.failure is None or context.failure.code is not worker.failure_code
    ):
        raise ValueError("worker and host failure classification must agree")
    return BuildManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="BuildManifest",
        id=context.id,
        case_manifest_sha256=context.case_manifest_sha256,
        environment_lock_sha256=context.environment_lock_sha256,
        command=worker.command,
        command_sha256=worker.command_sha256,
        parser_warnings=tuple(
            item.message for item in worker.builder_warnings if "parser" in item.message.casefold()
        ),
        parser_errors=worker.parser_errors,
        builder_configuration=worker.builder_configuration,
        plugin_source_sha256=context.plugin_source_sha256,
        plugin_binary=context.plugin_binary,
        plugin_compile_command=context.plugin_compile_command,
        plugin_build_log=context.plugin_build_log,
        timing_cache_mode=(
            "cold"
            if worker.timing_cache_state == "cold"
            else "warm_environment_local"
            if worker.timing_cache_state == "warm"
            else "disabled"
        ),
        timing_cache=(
            _artifact(
                worker.timing_cache.path,
                worker.timing_cache.output_sha256,
                worker.timing_cache.bytes,
            )
            if worker.timing_cache is not None
            else None
        ),
        started_at=_datetime(worker.started_unix_seconds),
        ended_at=_datetime(worker.ended_unix_seconds),
        duration_seconds=worker.duration_seconds or 0.0,
        engine=(
            _artifact(worker.engine.path, worker.engine.sha256, worker.engine.bytes)
            if worker.engine is not None
            else None
        ),
        engine_inspector=(
            _artifact(worker.inspector.path, worker.inspector.sha256, worker.inspector.bytes)
            if worker.inspector is not None
            else None
        ),
        engine_device_memory_bytes=(
            worker.engine.device_memory_bytes if worker.engine is not None else None
        ),
        engine_bytes=worker.engine.bytes if worker.engine is not None else None,
        status=status,
        failure=context.failure,
        warnings=tuple(item.message for item in worker.builder_warnings),
    )


def _artifact(path: str, sha256: str, byte_count: int | None) -> ArtifactReference:
    worker_path = PurePosixPath(path)
    output = PurePosixPath("/output")
    if not worker_path.is_relative_to(output) or ".." in worker_path.parts:
        raise ValueError("worker artifact path must be below /output")
    suffix = worker_path.suffix
    media_type = {
        ".json": "application/json",
        ".plan": "application/octet-stream",
    }.get(suffix, "application/octet-stream")
    return ArtifactReference(
        path=worker_path.relative_to(output).as_posix(),
        sha256=sha256,
        bytes=byte_count or 0,
        media_type=media_type,
    )


def _datetime(value: float | None) -> datetime:
    return datetime.fromtimestamp(value or 0.0, tz=UTC)
