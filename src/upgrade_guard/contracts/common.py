"""Shared artifact, tensor, policy, and failure contracts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode

RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class Phase(StrEnum):
    """Stable qualification phases."""

    PREFLIGHT = "preflight"
    MATERIALIZE = "materialize"
    BUILD = "build"
    REFERENCE = "reference"
    CORRECTNESS = "correctness"
    DETERMINISM = "determinism"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    REDUCTION = "reduction"
    SANITIZER = "sanitizer"
    PROFILING = "profiling"
    REPRODUCTION = "reproduction"
    REPORT = "report"


class ResultStatus(StrEnum):
    """Overall result state independent of a specific failure code."""

    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    INCONCLUSIVE = "inconclusive"


class PrecisionMode(StrEnum):
    """Frozen precision artifact identities."""

    FP32 = "fp32"
    EXPLICIT_FP16 = "explicit_fp16"
    QDQ = "qdq"


class ArtifactReference(StrictModel):
    """One relative artifact with verified content identity."""

    path: RelativePath
    sha256: Sha256Digest
    bytes: int = Field(ge=0)
    media_type: str

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("artifact paths must be normalized relative POSIX paths")
        if "\\" in value or "\x00" in value:
            raise ValueError("artifact paths cannot contain backslashes or NUL bytes")
        return value


class TensorContract(StrictModel):
    """Expected named tensor dtype and concrete shape."""

    name: str
    dtype: Literal["float16", "float32", "float64", "int8", "int32", "int64", "bool"]
    shape: tuple[int, ...]

    @field_validator("shape")
    @classmethod
    def require_positive_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(dimension <= 0 for dimension in value):
            raise ValueError("tensor shapes must contain only positive dimensions")
        return value


class NumericalTolerance(StrictModel):
    """Immutable elementwise comparison tolerance."""

    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)


class NumericalPolicy(StrictModel):
    """Three independent numerical gates."""

    baseline_to_reference: NumericalTolerance
    candidate_to_reference: NumericalTolerance
    candidate_to_baseline: NumericalTolerance
    relative_error_guard: float = Field(default=1e-12, gt=0)
    require_top1_agreement: bool = False
    require_top5_agreement: bool = False


class DeterminismPolicy(StrictModel):
    """Repeated-execution policy."""

    repetitions: int = Field(default=20, ge=20)
    require_bitwise: bool
    tolerance: NumericalTolerance


class PerformancePolicy(StrictModel):
    """Unprofiled paired-block performance policy."""

    warmup_milliseconds: int = Field(ge=0)
    measurement_milliseconds: int = Field(gt=0)
    minimum_accepted_pairs: int = Field(default=20, ge=20)
    bootstrap_replicates: int = Field(default=5000, ge=1000)
    bootstrap_seed: int
    practical_allowance: float = Field(ge=0)
    shape_allowances: dict[str, float]
    shape_weights: dict[str, float]
    workload_provenance: str
    one_inference_stream: Literal[True] = True
    cuda_graph: Literal[False] = False

    @model_validator(mode="after")
    def validate_weights_and_allowances(self) -> PerformancePolicy:
        if not self.shape_weights:
            raise ValueError("at least one workload shape weight is required")
        if set(self.shape_allowances) != set(self.shape_weights):
            raise ValueError("shape allowances and weights must have identical shape IDs")
        if any(weight <= 0 for weight in self.shape_weights.values()):
            raise ValueError("shape weights must be positive")
        if abs(sum(self.shape_weights.values()) - 1.0) > 1e-9:
            raise ValueError("shape weights must sum to one")
        if any(allowance < 0 for allowance in self.shape_allowances.values()):
            raise ValueError("shape allowances cannot be negative")
        return self


class MemoryPolicy(StrictModel):
    """Gateable engine-size and device-memory allowances."""

    engine_bytes_absolute: int = Field(default=1024 * 1024, ge=0)
    engine_bytes_relative: float = Field(default=0.05, ge=0)
    device_memory_absolute: int = Field(default=8 * 1024 * 1024, ge=0)
    device_memory_relative: float = Field(default=0.05, ge=0)
    confirmation_builds: Literal[3] = 3


class HardwareValidityPolicy(StrictModel):
    """Locked environmental conditions for accepting a benchmark block."""

    selected_gpu_uuid: str
    maximum_temperature_celsius: float = Field(gt=0)
    maximum_clock_variation_ratio: float = Field(ge=0)
    maximum_power_variation_ratio: float = Field(ge=0)
    maximum_gpu_utilization_before_block: float = Field(ge=0, le=100)
    reject_competing_compute_processes: bool = True
    require_stable_power_limit: bool = True


class FailureRecord(StrictModel):
    """Complete stable failure predicate and evidence."""

    code: FailureCode
    phase: Phase
    environment_id: str | None
    model_id: str | None
    precision: PrecisionMode | None
    shape_id: str | None
    input_fixture_id: str | None
    output_name: str | None
    gate: str
    observed: str
    threshold: str | None
    evidence: tuple[ArtifactReference, ...]
    signature_sha256: Sha256Digest
