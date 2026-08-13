"""Top-level qualification specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import (
    DeterminismPolicy,
    HardwareValidityPolicy,
    MemoryPolicy,
    NumericalPolicy,
    PerformancePolicy,
    PrecisionMode,
)


class ShapeRange(StrictModel):
    """Minimum, optimum, and maximum optimization-profile shape."""

    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    @model_validator(mode="after")
    def validate_rank_and_order(self) -> ShapeRange:
        if not (len(self.minimum) == len(self.optimum) == len(self.maximum)):
            raise ValueError("profile shapes must have identical rank")
        if not self.minimum:
            raise ValueError("profile shape rank cannot be zero")
        for minimum, optimum, maximum in zip(self.minimum, self.optimum, self.maximum, strict=True):
            if minimum <= 0 or not minimum <= optimum <= maximum:
                raise ValueError("profile dimensions must satisfy 0 < min <= opt <= max")
        return self


class OptimizationProfile(StrictModel):
    """Named profile with one range per input."""

    id: str
    inputs: dict[str, ShapeRange]


class ConcreteShape(StrictModel):
    """One concrete input-shape case."""

    id: str
    inputs: dict[str, tuple[int, ...]]


class BuilderPolicy(StrictModel):
    """Strongly typed builder options."""

    strongly_typed: Literal[True] = True
    timing_cache: Literal["disabled", "environment_local"]
    workspace_limit_bytes: int = Field(gt=0)
    optimization_level: int = Field(ge=0, le=5)
    extra_options: tuple[str, ...] = ()


class ReductionBudget(StrictModel):
    """Bounded failure-reduction budget."""

    maximum_trials: int = Field(gt=0)
    maximum_seconds: int = Field(gt=0)
    confirmation_count: int = Field(ge=2)


class ArtifactRetentionPolicy(StrictModel):
    """Retained raw and aggregate artifact policy."""

    retain_raw_timings: bool = True
    retain_engines: bool = False
    retain_timing_caches: bool = False
    retain_diagnostic_reports: bool = True


class QualificationSpec(StrictModel):
    """Complete authored qualification request."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["Qualification"]
    baseline_environment_id: str
    candidate_environment_id: str
    environment_lock: str
    corpus_lock_id: str
    required_cases: tuple[str, ...]
    precision_modes: tuple[PrecisionMode, ...]
    optimization_profiles: tuple[OptimizationProfile, ...]
    concrete_shapes: tuple[ConcreteShape, ...]
    input_fixture_ids: tuple[str, ...]
    builder: BuilderPolicy
    numerical: NumericalPolicy
    determinism: DeterminismPolicy
    performance: PerformancePolicy
    memory: MemoryPolicy
    hardware_validity: HardwareValidityPolicy
    required_confirmations: int = Field(ge=1)
    reduction_budget: ReductionBudget
    retention: ArtifactRetentionPolicy

    @model_validator(mode="after")
    def validate_environment_and_case_sets(self) -> QualificationSpec:
        if self.baseline_environment_id == self.candidate_environment_id:
            raise ValueError("baseline and candidate environment IDs must differ")
        for name, values in (
            ("required_cases", self.required_cases),
            ("precision_modes", self.precision_modes),
            ("optimization_profiles", self.optimization_profiles),
            ("concrete_shapes", self.concrete_shapes),
            ("input_fixture_ids", self.input_fixture_ids),
        ):
            if not values:
                raise ValueError(f"{name} cannot be empty")
        return self
