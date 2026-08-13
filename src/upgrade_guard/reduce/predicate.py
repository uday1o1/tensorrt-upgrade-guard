"""Strict reduction request and stable predicate contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode


class NumericalPredicate(StrictModel):
    """Elementwise failure relation that reduction must preserve."""

    kind: Literal["numerical"]
    output_name: str
    reference_path: str
    candidate_path: str
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)


class ProfilePredicate(StrictModel):
    """Concrete shape outside a locked optimization profile."""

    kind: Literal["profile"]
    input_name: str
    observed_shape: tuple[int, ...]
    minimum_shape: tuple[int, ...]
    maximum_shape: tuple[int, ...]

    @model_validator(mode="after")
    def validate_shapes(self) -> ProfilePredicate:
        if not (len(self.observed_shape) == len(self.minimum_shape) == len(self.maximum_shape)):
            raise ValueError("profile predicate shapes require identical rank")
        if all(
            minimum <= observed <= maximum
            for observed, minimum, maximum in zip(
                self.observed_shape,
                self.minimum_shape,
                self.maximum_shape,
                strict=True,
            )
        ):
            raise ValueError("profile predicate observed shape does not violate the profile")
        return self


class PerformancePredicate(StrictModel):
    """Paired timing evidence and locked repeated-confidence policy."""

    kind: Literal["performance"]
    baseline_path: str
    candidate_path: str
    allowance: float = Field(ge=0)
    bootstrap_seed: int
    bootstrap_replicates: int = Field(ge=1000)
    minimum_pairs: int = Field(ge=20)


class ReductionRequest(StrictModel):
    """Bounded reduction session input."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["ReductionRequest"]
    failure_code: FailureCode
    signature_sha256: Sha256Digest
    confirmation_count: int = Field(ge=2)
    maximum_trials: int = Field(gt=0)
    maximum_seconds: int = Field(gt=0)
    predicate: NumericalPredicate | ProfilePredicate | PerformancePredicate = Field(
        discriminator="kind"
    )

    @model_validator(mode="after")
    def validate_code_matches_predicate(self) -> ReductionRequest:
        expected = {
            "numerical": FailureCode.NUMERICAL_REGRESSION,
            "profile": FailureCode.PROFILE_REJECTED,
            "performance": FailureCode.PERFORMANCE_REGRESSION,
        }[self.predicate.kind]
        if self.failure_code is not expected:
            raise ValueError("reduction failure code and predicate kind differ")
        return self
