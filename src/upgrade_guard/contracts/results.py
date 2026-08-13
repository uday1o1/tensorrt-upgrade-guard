"""Worker and cross-environment run-result contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import (
    ArtifactReference,
    FailureRecord,
    ResultStatus,
    TensorContract,
)
from upgrade_guard.contracts.environment import Sha256Digest


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
