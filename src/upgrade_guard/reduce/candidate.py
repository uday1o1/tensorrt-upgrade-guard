"""Typed candidates used by the real GPU failure predicates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256, sha256_file
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode, InvalidInputError


class LockedEnvironmentBoundary(StrictModel):
    """Observed adjacent passing and failing locked environments."""

    last_passing: str = Field(min_length=1, max_length=64)
    first_failing: str = Field(min_length=1, max_length=64)
    passing_evidence_sha256: tuple[Sha256Digest, ...] = Field(min_length=1)
    failing_evidence_sha256: tuple[Sha256Digest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_environments(self) -> LockedEnvironmentBoundary:
        if self.last_passing == self.first_failing:
            raise ValueError("environment boundary identities must be distinct")
        return self


class EnvironmentHistoryObservation(StrictModel):
    """One confirmed same-candidate result under a locked environment."""

    environment_id: str = Field(min_length=1, max_length=64)
    outcome: Literal["reproduced"]
    failure_code: FailureCode
    predicate_signature_sha256: Sha256Digest
    confirmation_count: int = Field(ge=2)
    trial_evidence_sha256: tuple[tuple[Sha256Digest, ...], ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_trial_evidence(self) -> EnvironmentHistoryObservation:
        if len(self.trial_evidence_sha256) != self.confirmation_count or any(
            not evidence for evidence in self.trial_evidence_sha256
        ):
            raise ValueError("environment observation requires evidence for every confirmation")
        return self


class EnvironmentHistoryNotApplicable(StrictModel):
    """Typed proof that a source-induced seed has no passing predecessor."""

    status: Literal["not_applicable"] = "not_applicable"
    reason: Literal["source_induced_seed_reproduces_in_all_locked_environments"] = (
        "source_induced_seed_reproduces_in_all_locked_environments"
    )
    observations: tuple[EnvironmentHistoryObservation, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_observations(self) -> EnvironmentHistoryNotApplicable:
        identities = tuple(item.environment_id for item in self.observations)
        if len(identities) != len(set(identities)):
            raise ValueError("not-applicable environment observations must be unique")
        failure_codes = {item.failure_code for item in self.observations}
        signatures = {item.predicate_signature_sha256 for item in self.observations}
        if len(failure_codes) != 1 or len(signatures) != 1:
            raise ValueError("not-applicable observations must preserve one stable predicate")
        return self


class G2ReductionCandidate(StrictModel):
    """Material parameters consumed by the quarantined numerical seed."""

    schema_version: Literal["upgradeguard.dev/g2-reduction-candidate/v1"] = (
        "upgradeguard.dev/g2-reduction-candidate/v1"
    )
    outputs: tuple[Literal["G2", "G3", "G5"], ...] = ("G2", "G3", "G5")
    rows: int = Field(ge=1, le=4096)
    hidden: int = Field(ge=1, le=65536)
    x_value: float = Field(allow_inf_nan=False)
    residual_value: float = Field(allow_inf_nan=False)
    gamma_value: float = Field(allow_inf_nan=False)
    environment_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$",
    )
    environment_history: tuple[str, ...] = ()
    environment_boundary: LockedEnvironmentBoundary | None = None
    environment_history_not_applicable: EnvironmentHistoryNotApplicable | None = None

    @model_validator(mode="after")
    def validate_outputs(self) -> G2ReductionCandidate:
        if self.outputs not in {("G2", "G3", "G5"), ("G2",)}:
            raise ValueError("G2 candidate outputs must be the original set or isolated G2")
        _validate_environment_history(
            self.environment_id,
            self.environment_history,
            self.environment_boundary,
            self.environment_history_not_applicable,
        )
        return self

    def candidate_sha256(self) -> str:
        """Hash every execution-driving candidate field."""

        return model_sha256(self)


class G7ReductionCandidate(StrictModel):
    """Model, profile, inputs, and builder options for one G7 predicate trial."""

    schema_version: Literal["upgradeguard.dev/g7-reduction-candidate/v1"] = (
        "upgradeguard.dev/g7-reduction-candidate/v1"
    )
    model_path: Path
    model_sha256: Sha256Digest
    output_names: tuple[str, ...] = ("output",)
    batch: int = Field(ge=1, le=4096)
    sequence: int = Field(ge=1, le=65536)
    hidden: int = Field(ge=1, le=65536)
    profile_min_batch: int = Field(ge=1)
    profile_opt_batch: int = Field(ge=1)
    profile_max_batch: int = Field(ge=1)
    profile_min_sequence: int = Field(ge=1)
    profile_opt_sequence: int = Field(ge=1)
    profile_max_sequence: int = Field(ge=1)
    input_mode: Literal["original", "zeros", "ones"] = "original"
    tokens_path: Path | None = None
    tokens_sha256: Sha256Digest | None = None
    mask_path: Path | None = None
    mask_sha256: Sha256Digest | None = None
    workspace_bytes: int = Field(gt=0)
    optimization_level: int = Field(ge=0, le=5)
    environment_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$",
    )
    graph_history_sha256: tuple[Sha256Digest, ...] = ()
    environment_history: tuple[str, ...] = ()
    environment_boundary: LockedEnvironmentBoundary | None = None
    environment_history_not_applicable: EnvironmentHistoryNotApplicable | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> G7ReductionCandidate:
        if not self.output_names or len(set(self.output_names)) != len(self.output_names):
            raise ValueError("G7 output names must be nonempty and unique")
        if not (
            self.profile_min_batch <= self.profile_opt_batch <= self.profile_max_batch
            and self.profile_min_sequence <= self.profile_opt_sequence <= self.profile_max_sequence
        ):
            raise ValueError("G7 optimization profile ordering is invalid")
        sources = (
            self.tokens_path,
            self.tokens_sha256,
            self.mask_path,
            self.mask_sha256,
        )
        if self.input_mode == "original" and any(item is None for item in sources):
            raise ValueError("original G7 inputs require both paths and hashes")
        _validate_environment_history(
            self.environment_id,
            self.environment_history,
            self.environment_boundary,
            self.environment_history_not_applicable,
        )
        return self

    def verify_artifacts(self) -> None:
        """Reject missing, replaced, or symlinked execution artifacts."""

        _verify(self.model_path, self.model_sha256, "G7 model")
        if self.input_mode == "original":
            assert self.tokens_path is not None
            assert self.tokens_sha256 is not None
            assert self.mask_path is not None
            assert self.mask_sha256 is not None
            _verify(self.tokens_path, self.tokens_sha256, "G7 tokens")
            _verify(self.mask_path, self.mask_sha256, "G7 mask")

    def candidate_sha256(self) -> str:
        """Hash portable identities and execution options, excluding host paths."""

        return model_sha256(
            self,
            exclude={"model_path", "tokens_path", "mask_path"},
        )


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise InvalidInputError(f"{label} must be a regular file")
    if sha256_file(path) != expected:
        raise InvalidInputError(f"{label} hash does not match its candidate contract")


def _validate_environment_history(
    environment_id: str,
    history: tuple[str, ...],
    boundary: LockedEnvironmentBoundary | None,
    not_applicable: EnvironmentHistoryNotApplicable | None,
) -> None:
    if history and (len(history) < 2 or len(history) != len(set(history))):
        raise ValueError("candidate environment history must contain unique ordered entries")
    if history and environment_id not in history:
        raise ValueError("candidate execution environment is absent from its history")
    if boundary is not None and not_applicable is not None:
        raise ValueError("environment history cannot have two terminal dispositions")
    if not_applicable is not None:
        observed = tuple(item.environment_id for item in not_applicable.observations)
        if history != observed:
            raise ValueError("not-applicable observations must cover the ordered history")
        if environment_id != history[-1]:
            raise ValueError("not-applicable seed must retain the final locked environment")
        return
    if boundary is None:
        return
    if history != (boundary.last_passing, boundary.first_failing):
        raise ValueError("reduced environment history must equal its retained boundary")
    if environment_id != boundary.first_failing:
        raise ValueError("candidate must execute in the first failing environment")
