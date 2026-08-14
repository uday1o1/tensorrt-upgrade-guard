"""Typed bounded reduction orchestration with hash-chained predicate attempts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode, InvalidInputError
from upgrade_guard.reduce.general import ReductionLimits

CandidateT = TypeVar("CandidateT")


class ReductionStage(StrEnum):
    """Required V1 reduction order plus confirmation and clean replay boundaries."""

    CONFIRM_ORIGINAL = "confirm_original"
    OUTPUTS = "outputs"
    CONCRETE_SHAPE = "concrete_shape"
    OPTIMIZATION_PROFILE = "optimization_profile"
    INPUTS = "inputs"
    BUILDER_OPTIONS = "builder_options"
    FREEZE_DYNAMIC_SHAPE = "freeze_dynamic_shape"
    FOLD_SHAPE_OPERATIONS = "fold_shape_operations"
    POLYGRAPHY_BISECT = "polygraphy_bisect"
    POLYGRAPHY_LINEAR = "polygraphy_linear"
    ENVIRONMENT_HISTORY = "environment_history"
    FINAL_EMPTY_REPLAY = "final_empty_replay"


REDUCTION_STAGES = tuple(
    stage
    for stage in ReductionStage
    if stage not in {ReductionStage.CONFIRM_ORIGINAL, ReductionStage.FINAL_EMPTY_REPLAY}
)


class PredicateOutcome(StrEnum):
    """Behavioral outcomes with infrastructure invalidity kept distinct."""

    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"


class ReductionStatus(StrEnum):
    """Terminal bounded-session status."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"


class PredicateObservation(StrictModel):
    """One predicate execution and its retained evidence identities."""

    outcome: PredicateOutcome
    failure_code: FailureCode | None = None
    predicate_signature_sha256: Sha256Digest | None = None
    evidence_sha256: tuple[Sha256Digest, ...] = ()
    detail: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def validate_outcome(self) -> PredicateObservation:
        if self.outcome is PredicateOutcome.REPRODUCED and (
            self.failure_code is None or self.predicate_signature_sha256 is None
        ):
            raise ValueError("reproduced trials require failure code and predicate signature")
        if self.outcome is PredicateOutcome.INFRASTRUCTURE_INVALID and not self.detail:
            raise ValueError("infrastructure-invalid trials require a retained detail")
        return self


class ReductionAttempt(StrictModel):
    """One hash-chained predicate attempt retained in execution order."""

    sequence: int = Field(ge=1)
    stage: ReductionStage
    candidate_sha256: Sha256Digest
    outcome: PredicateOutcome
    failure_code: FailureCode | None
    predicate_signature_sha256: Sha256Digest | None
    evidence_sha256: tuple[Sha256Digest, ...]
    detail: str | None
    duration_seconds: float = Field(ge=0)
    previous_attempt_sha256: Sha256Digest | None
    attempt_sha256: Sha256Digest

    def computed_sha256(self) -> str:
        """Hash the complete attempt except its self-hash."""

        return model_sha256(self, exclude={"attempt_sha256"})


class ReductionEnvironmentIdentity(StrictModel):
    """One ordered locked worker identity used by the predicate."""

    id: str = Field(min_length=1, max_length=64)
    worker_manifest_sha256: Sha256Digest


class ReductionShapeIdentity(StrictModel):
    """One named concrete shape retained by the original predicate."""

    input_name: str = Field(min_length=1, max_length=256)
    dimensions: tuple[int, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ReductionShapeIdentity:
        if any(dimension <= 0 for dimension in self.dimensions):
            raise ValueError("reduction predicate dimensions must be positive")
        return self


class ReductionPredicateContract(StrictModel):
    """Complete original failure relation preserved by every reduction trial."""

    schema_version: Literal["upgradeguard.dev/reduction-predicate/v1"] = (
        "upgradeguard.dev/reduction-predicate/v1"
    )
    failure_code: FailureCode
    predicate_signature_sha256: Sha256Digest
    environments: tuple[ReductionEnvironmentIdentity, ReductionEnvironmentIdentity]
    model_sha256: Sha256Digest
    executor_sha256: Sha256Digest | None = None
    output_name: str | None = Field(default=None, max_length=256)
    concrete_shapes: tuple[ReductionShapeIdentity, ...]
    input_sha256: tuple[Sha256Digest, ...]
    threshold_relationship: str = Field(min_length=1, max_length=4096)
    confirmation_count: int = Field(ge=2)
    infrastructure_satisfies_predicate: Literal[False] = False
    predicate_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_predicate(self) -> ReductionPredicateContract:
        if self.environments[0].id == self.environments[1].id:
            raise ValueError("reduction predicate environment pair must be ordered and distinct")
        if not self.concrete_shapes:
            raise ValueError("reduction predicate requires at least one concrete shape")
        if not self.input_sha256:
            raise ValueError("reduction predicate requires input identities")
        if self.predicate_sha256 != self.computed_sha256():
            raise ValueError("reduction predicate self-hash is invalid")
        return self

    def computed_sha256(self) -> str:
        """Hash every predicate field except its checksum."""

        return model_sha256(self, exclude={"predicate_sha256"})


class ReductionSessionManifest(StrictModel):
    """Portable, bounded, hash-chained reduction execution evidence."""

    schema_version: Literal["upgradeguard.dev/reduction-session/v1"] = (
        "upgradeguard.dev/reduction-session/v1"
    )
    predicate: ReductionPredicateContract
    status: ReductionStatus
    expected_failure_code: FailureCode
    predicate_signature_sha256: Sha256Digest
    original_candidate_sha256: Sha256Digest
    final_candidate_sha256: Sha256Digest
    completed_stages: tuple[ReductionStage, ...]
    attempts: tuple[ReductionAttempt, ...]
    maximum_trials: int = Field(gt=0)
    maximum_seconds: float = Field(gt=0)
    confirmation_count: int = Field(ge=2)
    elapsed_seconds: float = Field(ge=0)
    session_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_attempt_history(self) -> ReductionSessionManifest:
        if self.predicate.failure_code is not self.expected_failure_code:
            raise ValueError("session failure code differs from its predicate")
        if self.predicate.predicate_signature_sha256 != self.predicate_signature_sha256:
            raise ValueError("session signature differs from its predicate")
        if self.predicate.confirmation_count != self.confirmation_count:
            raise ValueError("session confirmation count differs from its predicate")
        previous: str | None = None
        for sequence, attempt in enumerate(self.attempts, start=1):
            if attempt.sequence != sequence:
                raise ValueError("reduction attempt sequence is not contiguous")
            if attempt.previous_attempt_sha256 != previous:
                raise ValueError("reduction attempt hash chain is invalid")
            if attempt.attempt_sha256 != attempt.computed_sha256():
                raise ValueError("reduction attempt self-hash is invalid")
            previous = attempt.attempt_sha256
        if self.session_sha256 != self.computed_sha256():
            raise ValueError("reduction session self-hash is invalid")
        return self

    def computed_sha256(self) -> str:
        """Hash every session field except the checksum itself."""

        return model_sha256(self, exclude={"session_sha256"})


@dataclass(frozen=True, slots=True)
class ReductionWorkflowResult[CandidateT]:
    """Final candidate paired with its typed execution manifest."""

    candidate: CandidateT
    manifest: ReductionSessionManifest


Reducer = Callable[[CandidateT], Sequence[CandidateT]]
Predicate = Callable[[CandidateT], PredicateObservation]
CleanReplay = Callable[[CandidateT, Path], PredicateObservation]
CandidateHasher = Callable[[CandidateT], str]


class ReductionStateMachine[CandidateT]:
    """Run every required reducer while re-confirming one stable predicate."""

    def __init__(
        self,
        *,
        expected_failure_code: FailureCode,
        predicate_signature_sha256: str,
        predicate_contract: ReductionPredicateContract,
        predicate: Predicate[CandidateT],
        reducers: Mapping[ReductionStage, Reducer[CandidateT]],
        clean_replay: CleanReplay[CandidateT],
        candidate_sha256: CandidateHasher[CandidateT],
        limits: ReductionLimits,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if set(reducers) != set(REDUCTION_STAGES):
            missing = sorted(stage.value for stage in set(REDUCTION_STAGES) - set(reducers))
            extra = sorted(stage.value for stage in set(reducers) - set(REDUCTION_STAGES))
            raise InvalidInputError(
                "reduction state machine requires every V1 reducer exactly once",
                details={"missing": missing, "extra": extra},
            )
        self._expected_failure_code = expected_failure_code
        self._predicate_signature = _digest(predicate_signature_sha256)
        if (
            predicate_contract.failure_code is not expected_failure_code
            or predicate_contract.predicate_signature_sha256 != self._predicate_signature
            or predicate_contract.confirmation_count != limits.confirmation_count
        ):
            raise InvalidInputError(
                "state-machine configuration differs from its predicate contract"
            )
        self._predicate_contract = predicate_contract
        self._predicate = predicate
        self._reducers = dict(reducers)
        self._clean_replay = clean_replay
        self._candidate_sha256 = candidate_sha256
        self._limits = limits
        self._clock = clock
        self._started = clock()
        self._attempts: list[ReductionAttempt] = []
        self._completed: list[ReductionStage] = []

    def run(
        self, original: CandidateT, *, replay_parent: Path
    ) -> ReductionWorkflowResult[CandidateT]:
        """Confirm, reduce in required order, and replay from fresh empty directories."""

        replay_parent.mkdir(parents=True, exist_ok=True)
        original_hash = self._hash_candidate(original)
        current = original
        outcome = self._confirm(current, ReductionStage.CONFIRM_ORIGINAL)
        if outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
            return self._result(current, original_hash, ReductionStatus.INFRASTRUCTURE_INVALID)
        if outcome is not PredicateOutcome.REPRODUCED:
            if self._budget_exhausted():
                return self._result(current, original_hash, ReductionStatus.BUDGET_EXHAUSTED)
            raise InvalidInputError("original reduction candidate is not a confirmed failure")
        self._completed.append(ReductionStage.CONFIRM_ORIGINAL)

        for stage in REDUCTION_STAGES:
            if self._budget_exhausted():
                return self._result(current, original_hash, ReductionStatus.BUDGET_EXHAUSTED)
            candidates = tuple(self._reducers[stage](current))
            attempted_candidate = False
            accepted_candidate = False
            for candidate in candidates:
                if self._hash_candidate(candidate) == self._hash_candidate(current):
                    continue
                attempted_candidate = True
                outcome = self._confirm(candidate, stage)
                if outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
                    return self._result(
                        current, original_hash, ReductionStatus.INFRASTRUCTURE_INVALID
                    )
                if outcome is PredicateOutcome.REPRODUCED:
                    current = candidate
                    accepted_candidate = True
                    break
                if self._budget_exhausted():
                    return self._result(current, original_hash, ReductionStatus.BUDGET_EXHAUSTED)
            if stage is ReductionStage.ENVIRONMENT_HISTORY and not attempted_candidate:
                raise InvalidInputError(
                    "environment history reducer did not produce a changed boundary"
                )
            if not attempted_candidate:
                outcome = self._confirm(current, stage)
                if outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
                    return self._result(
                        current, original_hash, ReductionStatus.INFRASTRUCTURE_INVALID
                    )
                if outcome is not PredicateOutcome.REPRODUCED:
                    if self._budget_exhausted():
                        return self._result(
                            current, original_hash, ReductionStatus.BUDGET_EXHAUSTED
                        )
                    raise InvalidInputError(
                        f"stable predicate no longer reproduces at stage {stage.value}"
                    )
            if stage is ReductionStage.ENVIRONMENT_HISTORY and not accepted_candidate:
                raise InvalidInputError(
                    "environment history reducer did not retain its changed boundary"
                )
            self._completed.append(stage)

        final_outcome = self._confirm_clean_replay(current, replay_parent)
        if final_outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
            return self._result(current, original_hash, ReductionStatus.INFRASTRUCTURE_INVALID)
        if final_outcome is not PredicateOutcome.REPRODUCED:
            if self._budget_exhausted():
                return self._result(current, original_hash, ReductionStatus.BUDGET_EXHAUSTED)
            raise InvalidInputError(
                "final reduced candidate did not replay from an empty directory"
            )
        self._completed.append(ReductionStage.FINAL_EMPTY_REPLAY)
        return self._result(current, original_hash, ReductionStatus.COMPLETED)

    def _confirm(self, candidate: CandidateT, stage: ReductionStage) -> PredicateOutcome:
        for _ in range(self._limits.confirmation_count):
            if self._budget_exhausted():
                return PredicateOutcome.NOT_REPRODUCED
            started = self._clock()
            observation = self._predicate(candidate)
            self._record(stage, candidate, observation, self._clock() - started)
            if observation.outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
                return observation.outcome
            if observation.outcome is not PredicateOutcome.REPRODUCED:
                return PredicateOutcome.NOT_REPRODUCED
            self._validate_predicate_identity(observation)
        return PredicateOutcome.REPRODUCED

    def _confirm_clean_replay(self, candidate: CandidateT, replay_parent: Path) -> PredicateOutcome:
        for _ in range(self._limits.confirmation_count):
            if self._budget_exhausted():
                return PredicateOutcome.NOT_REPRODUCED
            started = self._clock()
            with tempfile.TemporaryDirectory(
                prefix="upgrade-guard-replay-", dir=replay_parent
            ) as name:
                clean = Path(name)
                if any(clean.iterdir()):
                    raise InvalidInputError("clean replay directory was not empty")
                observation = self._clean_replay(candidate, clean)
            self._record(
                ReductionStage.FINAL_EMPTY_REPLAY,
                candidate,
                observation,
                self._clock() - started,
            )
            if observation.outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
                return observation.outcome
            if observation.outcome is not PredicateOutcome.REPRODUCED:
                return PredicateOutcome.NOT_REPRODUCED
            self._validate_predicate_identity(observation)
        return PredicateOutcome.REPRODUCED

    def _record(
        self,
        stage: ReductionStage,
        candidate: CandidateT,
        observation: PredicateObservation,
        duration_seconds: float,
    ) -> None:
        previous = self._attempts[-1].attempt_sha256 if self._attempts else None
        attempt = ReductionAttempt(
            sequence=len(self._attempts) + 1,
            stage=stage,
            candidate_sha256=self._hash_candidate(candidate),
            outcome=observation.outcome,
            failure_code=observation.failure_code,
            predicate_signature_sha256=observation.predicate_signature_sha256,
            evidence_sha256=observation.evidence_sha256,
            detail=observation.detail,
            duration_seconds=max(0.0, duration_seconds),
            previous_attempt_sha256=previous,
            attempt_sha256="sha256:" + "0" * 64,
        )
        attempt = attempt.model_copy(update={"attempt_sha256": attempt.computed_sha256()})
        self._attempts.append(attempt)

    def _validate_predicate_identity(self, observation: PredicateObservation) -> None:
        if (
            observation.failure_code is not self._expected_failure_code
            or observation.predicate_signature_sha256 != self._predicate_signature
        ):
            raise InvalidInputError("reducer reproduced a different failure predicate")

    def _budget_exhausted(self) -> bool:
        return len(self._attempts) >= self._limits.maximum_trials or (
            self._clock() - self._started >= self._limits.maximum_seconds
        )

    def _hash_candidate(self, candidate: CandidateT) -> str:
        return _digest(self._candidate_sha256(candidate))

    def _result(
        self,
        candidate: CandidateT,
        original_hash: str,
        status: ReductionStatus,
    ) -> ReductionWorkflowResult[CandidateT]:
        manifest = ReductionSessionManifest.model_construct(
            predicate=self._predicate_contract,
            status=status,
            expected_failure_code=self._expected_failure_code,
            predicate_signature_sha256=self._predicate_signature,
            original_candidate_sha256=original_hash,
            final_candidate_sha256=self._hash_candidate(candidate),
            completed_stages=tuple(self._completed),
            attempts=tuple(self._attempts),
            maximum_trials=self._limits.maximum_trials,
            maximum_seconds=self._limits.maximum_seconds,
            confirmation_count=self._limits.confirmation_count,
            elapsed_seconds=max(0.0, self._clock() - self._started),
            session_sha256="sha256:" + "0" * 64,
        )
        manifest = ReductionSessionManifest.model_validate(
            manifest.model_copy(update={"session_sha256": manifest.computed_sha256()})
        )
        return ReductionWorkflowResult(candidate, manifest)


def write_session_manifest(path: Path, manifest: ReductionSessionManifest) -> None:
    """Atomically retain a validated reduction manifest."""

    if manifest.session_sha256 != manifest.computed_sha256():
        raise InvalidInputError("reduction session self-hash is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest.model_dump(mode="json"), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _digest(value: str) -> str:
    if (
        not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise InvalidInputError("reduction identity must be a lowercase SHA-256 digest")
    return value
