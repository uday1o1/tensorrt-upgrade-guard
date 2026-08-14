"""Worker and cross-environment run-result contracts."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from upgrade_guard.classify import status_for_failure
from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import (
    ArtifactReference,
    FailureRecord,
    ResultStatus,
    TensorContract,
)
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode


class WorkerTensorArtifact(StrictModel):
    """One named tensor artifact emitted by a worker."""

    name: str
    path: str
    sha256: Sha256Digest
    bytes: int = Field(ge=0)
    dtype: Literal["float16", "float32", "float64", "int8", "int32", "int64", "bool"]
    shape: tuple[int, ...]


class WorkerInputIntegrity(StrictModel):
    """Per-named-input host and device identity for one repetition."""

    name: str
    source_sha256: Sha256Digest
    host_value_sha256: Sha256Digest
    device_value_sha256: Sha256Digest
    stable: bool


class WorkerRepetition(StrictModel):
    """One execution repetition with input-integrity and output evidence."""

    index: int = Field(ge=0)
    inputs: tuple[WorkerInputIntegrity, ...] = ()
    outputs: tuple[WorkerTensorArtifact, ...]


class WorkerTacticDiagnostic(StrictModel):
    """Runtime-selected plugin tactic bound to an engine and shape."""

    path: str
    sha256: Sha256Digest
    bytes: int = Field(ge=0)
    engine_sha256: Sha256Digest
    selected_tactic: Literal["kSCALAR_REFERENCE", "kVECTORIZED_WARP"]
    rows: int = Field(gt=0)
    hidden: int = Field(gt=0)
    enqueue_count: int = Field(gt=0)


class WorkerCorrectnessResult(StrictModel):
    """Strict production output of ``worker.run_correctness``."""

    schema_version: Literal["upgradeguard.dev/worker-correctness/v1"]
    status: Literal["passed", "failed"]
    command: tuple[str, ...]
    command_sha256: Sha256Digest
    engine_sha256: Sha256Digest | None = None
    input_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    repetitions: tuple[WorkerRepetition, ...] = ()
    input_integrity_stable: bool | None = None
    tactic_diagnostic: WorkerTacticDiagnostic | None = None
    memory_diagnostics: dict[str, Any] | None = None
    tensorrt_version: str | None = None
    started_unix_seconds: float
    ended_unix_seconds: float
    duration_seconds: float = Field(ge=0)
    failure_code: FailureCode | None = None
    error_type: str | None = None
    message: str | None = None
    input_integrity_evidence: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> WorkerCorrectnessResult:
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.started_unix_seconds,
                    self.ended_unix_seconds,
                    self.duration_seconds,
                )
            )
            or self.ended_unix_seconds < self.started_unix_seconds
            or abs(self.duration_seconds - (self.ended_unix_seconds - self.started_unix_seconds))
            > 1e-6
        ):
            raise ValueError("worker run timestamps and duration must agree")
        if self.status == "passed":
            if (
                self.engine_sha256 is None
                or not self.input_sha256
                or not self.repetitions
                or self.input_integrity_stable is not True
                or self.memory_diagnostics is None
                or self.tensorrt_version is None
                or self.failure_code is not None
            ):
                raise ValueError("passing worker runs require complete typed evidence")
            expected_inputs = set(self.input_sha256)
            for expected_index, repetition in enumerate(self.repetitions):
                if repetition.index != expected_index:
                    raise ValueError("worker repetition indexes must be contiguous")
                observed_inputs = {item.name for item in repetition.inputs}
                if (
                    len(repetition.inputs) != len(expected_inputs)
                    or observed_inputs != expected_inputs
                    or not all(
                        item.stable and item.host_value_sha256 == item.device_value_sha256
                        for item in repetition.inputs
                    )
                ):
                    raise ValueError("worker repetition input integrity failed")
                if any(
                    item.source_sha256 != self.input_sha256[item.name] for item in repetition.inputs
                ):
                    raise ValueError("worker repetition source input identity changed")
                output_names = [item.name for item in repetition.outputs]
                if not output_names or len(output_names) != len(set(output_names)):
                    raise ValueError("worker repetition outputs must have unique names")
            if (
                self.tactic_diagnostic is not None
                and self.tactic_diagnostic.engine_sha256 != self.engine_sha256
            ):
                raise ValueError("worker tactic diagnostic belongs to a different engine")
        elif self.failure_code is None or self.error_type is None or self.message is None:
            raise ValueError("failed worker runs require typed failure evidence")
        return self


class NumericalSummary(StrictModel):
    """Bounded numerical metrics for one output comparison."""

    output_name: str
    element_count: int = Field(ge=0)
    maximum_absolute_error: float = Field(ge=0)
    mean_absolute_error: float = Field(ge=0)
    median_absolute_error: float = Field(ge=0)
    p99_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    p99_relative_error: float = Field(ge=0)
    cosine_similarity: float = Field(ge=-1, le=1)
    l2_error: float = Field(ge=0)
    reference_nonfinite_count: int = Field(ge=0)
    candidate_nonfinite_count: int = Field(ge=0)
    failed_element_count: int = Field(ge=0)
    failed_element_indexes: tuple[int, ...]
    elementwise_passed: bool
    top1_agreement: bool | None
    top5_agreement: bool | None


class DeterminismSummary(StrictModel):
    """Bitwise and tolerance repetition evidence."""

    repetitions: int = Field(ge=1)
    unique_output_hashes: tuple[Sha256Digest, ...]
    bitwise_stable: bool
    tolerance_stable: bool
    input_hashes_stable: bool
    nonfinite_observed: bool


class TimingBlock(StrictModel):
    """One primary unprofiled benchmark block."""

    pair_index: int = Field(ge=0)
    environment_id: str
    order_in_pair: Literal[0, 1]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    iteration_count: int = Field(ge=0)
    median_milliseconds: float | None = Field(default=None, gt=0)
    mean_milliseconds: float | None = Field(default=None, gt=0)
    coefficient_of_variation: float | None = Field(default=None, ge=0)
    raw_timings: ArtifactReference | None
    temperature_celsius: float | None
    graphics_clock_mhz: int | None
    memory_clock_mhz: int | None
    power_watts: float | None
    utilization_percent: float | None
    competing_compute_processes: tuple[str, ...]
    profiled: Literal[False] = False

    @model_validator(mode="after")
    def accepted_blocks_require_measurements(self) -> TimingBlock:
        if self.accepted and (
            self.median_milliseconds is None
            or self.mean_milliseconds is None
            or self.raw_timings is None
        ):
            raise ValueError("accepted timing blocks require raw and aggregate timings")
        if not self.accepted and not self.rejection_reasons:
            raise ValueError("rejected timing blocks require a reason")
        return self


class MemoryObservation(StrictModel):
    """Separate gateable and diagnostic memory measurements."""

    serialized_engine_bytes: int = Field(ge=0)
    engine_device_memory_bytes: int = Field(ge=0)
    execution_context_bytes: int | None = Field(default=None, ge=0)
    builder_peak_bytes: int | None = Field(default=None, ge=0)
    coarse_process_gpu_bytes: int | None = Field(default=None, ge=0)
    host_peak_bytes: int | None = Field(default=None, ge=0)


class HardwareObservation(StrictModel):
    """Observed identity and block-validity facts."""

    gpu_uuid: str
    driver: str
    environment_lock_sha256: Sha256Digest
    valid: bool
    invalid_reasons: tuple[str, ...]


class RunResultAdapterContext(StrictModel):
    """Host-owned identities needed to promote worker execution evidence."""

    id: str
    case_manifest_sha256: Sha256Digest
    build_manifest_sha256: Sha256Digest
    environment_lock_sha256: Sha256Digest
    hardware_sha256: Sha256Digest
    hardware: HardwareObservation
    started_at: AwareDatetime
    ended_at: AwareDatetime
    serialized_engine_bytes: int = Field(ge=0)
    engine_device_memory_bytes: int = Field(ge=0)
    determinism_tolerance_stable: bool
    numerical: tuple[NumericalSummary, ...] = ()
    failure: FailureRecord | None = None


class RunResult(StrictModel):
    """One case execution and all retained evidence."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["RunResult"]
    id: str
    case_manifest_sha256: Sha256Digest
    build_manifest_sha256: Sha256Digest
    environment_lock_sha256: Sha256Digest
    hardware_sha256: Sha256Digest
    command: tuple[str, ...]
    command_sha256: Sha256Digest
    output_schema: tuple[TensorContract, ...]
    output_artifacts: tuple[ArtifactReference, ...]
    numerical: tuple[NumericalSummary, ...]
    determinism: DeterminismSummary | None
    timing_blocks: tuple[TimingBlock, ...]
    memory: MemoryObservation | None
    hardware: HardwareObservation
    started_at: AwareDatetime
    ended_at: AwareDatetime
    status: ResultStatus
    failure: FailureRecord | None
    logs: tuple[ArtifactReference, ...]
    warnings: tuple[str, ...]
    diagnostics: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def require_failure_for_failed_status(self) -> RunResult:
        if (self.status is ResultStatus.PASSED) != (self.failure is None):
            raise ValueError("run status and failure record must agree")
        if self.hardware.environment_lock_sha256 != self.environment_lock_sha256:
            raise ValueError("run and hardware environment identities must agree")
        if self.ended_at < self.started_at:
            raise ValueError("run result timestamps must be chronological")
        return self


def adapt_worker_run(
    worker: WorkerCorrectnessResult, context: RunResultAdapterContext
) -> RunResult:
    """Promote strict worker evidence into the stable run manifest."""

    status = status_for_failure(worker.failure_code)
    if worker.status == "failed" and (
        context.failure is None or context.failure.code is not worker.failure_code
    ):
        raise ValueError("worker and host failure classification must agree")
    output_schema: tuple[TensorContract, ...] = ()
    output_artifacts: list[ArtifactReference] = []
    if worker.repetitions:
        output_schema = tuple(
            TensorContract(name=item.name, dtype=item.dtype, shape=item.shape)
            for item in worker.repetitions[0].outputs
        )
        for repetition in worker.repetitions:
            if tuple((item.name, item.dtype, item.shape) for item in repetition.outputs) != tuple(
                (item.name, item.dtype, item.shape) for item in worker.repetitions[0].outputs
            ):
                raise ValueError("worker output schema changed between repetitions")
            output_artifacts.extend(_worker_artifact(item) for item in repetition.outputs)
    unique_hashes = tuple(sorted({item.sha256 for item in output_artifacts}))
    hashes_by_output = {
        name: {
            item.sha256
            for repetition in worker.repetitions
            for item in repetition.outputs
            if item.name == name
        }
        for name in (item.name for item in output_schema)
    }
    determinism = (
        DeterminismSummary(
            repetitions=len(worker.repetitions),
            unique_output_hashes=unique_hashes,
            bitwise_stable=all(len(hashes) == 1 for hashes in hashes_by_output.values()),
            tolerance_stable=context.determinism_tolerance_stable,
            input_hashes_stable=worker.input_integrity_stable is True,
            nonfinite_observed=False,
        )
        if worker.repetitions
        else None
    )
    execution_context_bytes = None
    if worker.memory_diagnostics is not None:
        observed = worker.memory_diagnostics.get("execution_context_device_memory_bytes")
        execution_context_bytes = int(observed) if isinstance(observed, int) else None
    diagnostics = (
        (
            _path_artifact(
                worker.tactic_diagnostic.path,
                worker.tactic_diagnostic.sha256,
                worker.tactic_diagnostic.bytes,
                "application/x-ndjson",
            ),
        )
        if worker.tactic_diagnostic is not None
        else ()
    )
    return RunResult(
        api_version="upgradeguard.dev/v1alpha1",
        kind="RunResult",
        id=context.id,
        case_manifest_sha256=context.case_manifest_sha256,
        build_manifest_sha256=context.build_manifest_sha256,
        environment_lock_sha256=context.environment_lock_sha256,
        hardware_sha256=context.hardware_sha256,
        command=worker.command,
        command_sha256=worker.command_sha256,
        output_schema=output_schema,
        output_artifacts=tuple(output_artifacts),
        numerical=context.numerical,
        determinism=determinism,
        timing_blocks=(),
        memory=(
            MemoryObservation(
                serialized_engine_bytes=context.serialized_engine_bytes,
                engine_device_memory_bytes=context.engine_device_memory_bytes,
                execution_context_bytes=execution_context_bytes,
                builder_peak_bytes=None,
                coarse_process_gpu_bytes=None,
                host_peak_bytes=None,
            )
            if worker.status == "passed"
            else None
        ),
        hardware=context.hardware,
        started_at=context.started_at,
        ended_at=context.ended_at,
        status=status,
        failure=context.failure,
        logs=(),
        warnings=(),
        diagnostics=diagnostics,
    )


def _worker_artifact(item: WorkerTensorArtifact) -> ArtifactReference:
    return _path_artifact(item.path, item.sha256, item.bytes, "application/x-npy")


def _path_artifact(path: str, sha256: str, byte_count: int, media_type: str) -> ArtifactReference:
    worker_path = PurePosixPath(path)
    output = PurePosixPath("/output")
    if not worker_path.is_relative_to(output) or ".." in worker_path.parts:
        raise ValueError("worker artifact path must be below /output")
    return ArtifactReference(
        path=worker_path.relative_to(output).as_posix(),
        sha256=sha256,
        bytes=byte_count,
        media_type=media_type,
    )
