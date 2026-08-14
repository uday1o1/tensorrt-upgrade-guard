"""Candidate-aware reduction for genuine stored numerical qualification failures."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from upgrade_guard.compare.numerical import ThreeWayPrecedenceError, decide_three_way
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from upgrade_guard.contracts.build import WorkerBuildResult
from upgrade_guard.contracts.common import DeterminismPolicy, FailureRecord, NumericalPolicy
from upgrade_guard.contracts.environment import EnvironmentLock, MatrixLock, Sha256Digest
from upgrade_guard.contracts.results import WorkerCorrectnessResult
from upgrade_guard.errors import FailureCode, InfrastructureError, InvalidInputError
from upgrade_guard.reduce.candidate import LockedEnvironmentBoundary
from upgrade_guard.reduce.general import ReductionLimits
from upgrade_guard.reduce.workflow import (
    REDUCTION_STAGES,
    PredicateObservation,
    PredicateOutcome,
    ReductionEnvironmentIdentity,
    ReductionPredicateContract,
    ReductionSessionManifest,
    ReductionShapeIdentity,
    ReductionStage,
    ReductionStateMachine,
    ReductionStatus,
    write_session_manifest,
)
from upgrade_guard.worker.common import write_json_atomic


class DomainInput(StrictModel):
    """One exact named input consumed by the failing case."""

    name: str = Field(min_length=1, max_length=256)
    path: Path
    sha256: Sha256Digest
    shape: tuple[int, ...] = Field(min_length=1)


class DomainPlugin(StrictModel):
    """One environment-specific plugin binary used by the original execution."""

    environment_id: str = Field(min_length=1, max_length=64)
    path: Path
    sha256: Sha256Digest


class DomainNumericalCandidate(StrictModel):
    """All execution-driving inputs for one genuine three-way numerical predicate."""

    schema_version: str = "upgradeguard.dev/domain-numerical-candidate/v1"
    model_path: Path
    model_sha256: Sha256Digest
    profile_path: Path
    profile_sha256: Sha256Digest
    inputs: tuple[DomainInput, ...] = Field(min_length=1)
    reference_path: Path
    reference_sha256: Sha256Digest
    output_name: str = Field(min_length=1, max_length=256)
    semantics: Literal["tensor", "classification"] = "tensor"
    policy: NumericalPolicy
    determinism: DeterminismPolicy
    workspace_bytes: int = Field(gt=0)
    optimization_level: int = Field(ge=0, le=5)
    environment_history: tuple[str, str]
    plugins: tuple[DomainPlugin, ...] = ()
    comparison_flat_index: int | None = Field(default=None, ge=0)
    classification_indexes: tuple[int, ...] = ()
    environment_boundary: LockedEnvironmentBoundary | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> DomainNumericalCandidate:
        if self.environment_history[0] == self.environment_history[1]:
            raise ValueError("domain candidate requires two distinct locked environments")
        names = tuple(item.name for item in self.inputs)
        if len(names) != len(set(names)):
            raise ValueError("domain candidate input names must be unique")
        plugin_ids = tuple(item.environment_id for item in self.plugins)
        if len(plugin_ids) != len(set(plugin_ids)) or any(
            item not in self.environment_history for item in plugin_ids
        ):
            raise ValueError("domain candidate plugin identities are invalid")
        if self.plugins and set(plugin_ids) != set(self.environment_history):
            raise ValueError("plugin failures require one binary for each locked environment")
        if (
            self.environment_boundary is not None
            and (
                self.environment_boundary.last_passing,
                self.environment_boundary.first_failing,
            )
            != self.environment_history
        ):
            raise ValueError("domain environment boundary differs from locked history")
        if self.comparison_flat_index is not None and self.classification_indexes:
            raise ValueError("domain output reduction cannot use two index modes")
        if self.semantics == "tensor" and self.classification_indexes:
            raise ValueError("tensor candidate cannot retain classification indexes")
        if self.semantics == "classification" and self.comparison_flat_index is not None:
            raise ValueError("classification candidate cannot retain one scalar index")
        if self.classification_indexes and (
            tuple(sorted(set(self.classification_indexes))) != self.classification_indexes
        ):
            raise ValueError("classification indexes must be unique and sorted")
        return self

    def verify_artifacts(self) -> None:
        """Reject missing, replaced, or symlinked predicate inputs."""

        _verify(self.model_path, self.model_sha256, "model")
        _verify(self.profile_path, self.profile_sha256, "profile")
        _verify(self.reference_path, self.reference_sha256, "reference")
        for input_item in self.inputs:
            _verify(input_item.path, input_item.sha256, f"input {input_item.name}")
        for plugin_item in self.plugins:
            _verify(
                plugin_item.path,
                plugin_item.sha256,
                f"plugin {plugin_item.environment_id}",
            )

    def candidate_sha256(self) -> str:
        """Hash portable execution identity while excluding host-only source paths."""

        value = self.model_dump(mode="python")
        value["model_path"] = None
        value["profile_path"] = None
        value["reference_path"] = None
        value["inputs"] = [{**item.model_dump(mode="python"), "path": None} for item in self.inputs]
        value["plugins"] = [
            {**item.model_dump(mode="python"), "path": None} for item in self.plugins
        ]
        return sha256_bytes(canonical_json_bytes(value))


class DomainNumericalGpuPredicate:
    """Rebuild and run the same case under both exact locked workers."""

    def __init__(
        self,
        *,
        project: Path,
        matrix: MatrixLock,
        failure: FailureRecord,
        evidence_root: Path,
        worker: DockerGpuWorker | None = None,
    ) -> None:
        if failure.code is not FailureCode.NUMERICAL_REGRESSION:
            raise InvalidInputError("domain numerical predicate requires a numerical failure")
        self.project = project.resolve(strict=True)
        self.matrix = matrix
        self.failure = failure
        self.evidence_root = evidence_root
        self.worker = worker or DockerGpuWorker()

    def evaluate(
        self,
        candidate: DomainNumericalCandidate,
        output: Path | None = None,
    ) -> PredicateObservation:  # pragma: no cover - requires exact locked GPU workers
        """Execute two workers and classify their new outputs with the locked policy."""

        try:
            candidate.verify_artifacts()
            trial = output or self.evidence_root / f"trial-{uuid4().hex}"
            if trial.exists():
                if trial.is_symlink() or any(trial.iterdir()):
                    raise InfrastructureError("domain predicate trial directory is not empty")
            else:
                trial.mkdir(parents=True)
            corpus = trial / "corpus"
            runs = trial / "runs"
            corpus.mkdir()
            runs.mkdir()
            shutil.copyfile(candidate.model_path, corpus / "model.onnx")
            shutil.copyfile(candidate.profile_path, corpus / "profile.json")
            shutil.copyfile(candidate.reference_path, corpus / "reference.npy")
            for index, item in enumerate(candidate.inputs):
                shutil.copyfile(item.path, corpus / f"input-{index:03d}.npy")
            plugins = {item.environment_id: item for item in candidate.plugins}
            for plugin_environment_id in candidate.environment_history:
                plugin = plugins.get(plugin_environment_id)
                if plugin is not None:
                    shutil.copyfile(
                        plugin.path,
                        corpus / f"plugin-{plugin_environment_id}.so",
                    )
            outputs: dict[str, np.ndarray[Any, Any]] = {}
            evidence: list[str] = []
            for environment_id in candidate.environment_history:
                environment = self._environment(environment_id)
                environment_output = runs / environment_id
                environment_output.mkdir()
                plugin_argument = (
                    ("--plugin", f"/corpus/plugin-{environment_id}.so")
                    if environment_id in plugins
                    else ()
                )
                build_command = (
                    "python3",
                    "-m",
                    "upgrade_guard.worker.build_engine",
                    "--model",
                    "/corpus/model.onnx",
                    "--profile",
                    "/corpus/profile.json",
                    "--engine",
                    f"/output/{environment_id}/engine.plan",
                    "--inspector",
                    f"/output/{environment_id}/inspector.json",
                    "--timing-cache",
                    f"/output/{environment_id}/timing.cache",
                    "--result",
                    f"/output/{environment_id}/build.json",
                    "--workspace-bytes",
                    str(candidate.workspace_bytes),
                    "--optimization-level",
                    str(candidate.optimization_level),
                    *plugin_argument,
                )
                build_process = self.worker.run(
                    image=environment.worker_image.canonical_reference,
                    gpu_uuid=self.matrix.gpu_uuid,
                    mounts=WorkerMounts(self.project, corpus, runs),
                    command=build_command,
                    timeout_seconds=1800,
                    accepted_returncodes=(0, 1),
                )
                build = _load_build(environment_output / "build.json", build_command)
                evidence.append(sha256_file(environment_output / "build.json"))
                if build_process.returncode != 0 or build.status != "passed":
                    return self._not_reproduced(evidence, "candidate engine build did not pass")
                correctness_command = [
                    "python3",
                    "-m",
                    "upgrade_guard.worker.run_correctness",
                    "--engine",
                    f"/output/{environment_id}/engine.plan",
                ]
                for index, item in enumerate(candidate.inputs):
                    correctness_command.extend(
                        ("--input", f"{item.name}=/corpus/input-{index:03d}.npy")
                    )
                correctness_command.extend(
                    (
                        "--output",
                        f"/output/{environment_id}/outputs",
                        "--result",
                        f"/output/{environment_id}/correctness.json",
                        "--repetitions",
                        str(candidate.determinism.repetitions),
                        *plugin_argument,
                    )
                )
                command = tuple(correctness_command)
                process = self.worker.run(
                    image=environment.worker_image.canonical_reference,
                    gpu_uuid=self.matrix.gpu_uuid,
                    mounts=WorkerMounts(self.project, corpus, runs),
                    command=command,
                    timeout_seconds=900,
                    accepted_returncodes=(0, 1),
                )
                result = _load_correctness(environment_output / "correctness.json", command)
                evidence.append(sha256_file(environment_output / "correctness.json"))
                if process.returncode != 0 or result.status != "passed":
                    return self._not_reproduced(evidence, "candidate execution did not pass")
                values = _worker_outputs(
                    result,
                    root=runs,
                    output_name=candidate.output_name,
                )
                if len(values) != candidate.determinism.repetitions:
                    raise InfrastructureError("domain predicate repetition count differs")
                if not all(np.all(np.isfinite(value)) for value in values):
                    return self._not_reproduced(evidence, "domain output became nonfinite")
                if not _repetitions_are_stable(values, candidate.determinism):
                    return self._not_reproduced(evidence, "domain output became nondeterministic")
                outputs[environment_id] = values[0]
                first_outputs = tuple(
                    item
                    for item in result.repetitions[0].outputs
                    if item.name == candidate.output_name
                )
                if len(first_outputs) != 1:
                    raise InfrastructureError("domain first output evidence differs")
                evidence.append(
                    sha256_file(runs / Path(first_outputs[0].path).relative_to("/output"))
                )
            reference = np.load(corpus / "reference.npy", allow_pickle=False)
            baseline = outputs[candidate.environment_history[0]]
            current = outputs[candidate.environment_history[1]]
            if candidate.comparison_flat_index is not None:
                index = candidate.comparison_flat_index
                if (
                    index >= reference.size
                    or baseline.size != reference.size
                    or current.size != reference.size
                ):
                    return self._not_reproduced(evidence, "reduced output element no longer exists")
                reference = reference.reshape(-1)[index : index + 1]
                baseline = baseline.reshape(-1)[index : index + 1]
                current = current.reshape(-1)[index : index + 1]
            elif candidate.classification_indexes:
                indexes = np.asarray(candidate.classification_indexes, dtype=np.int64)
                if any(
                    value.ndim != 2
                    or value.shape[0] != reference.shape[0]
                    or value.shape[1] <= int(indexes[-1])
                    for value in (reference, baseline, current)
                ):
                    return self._not_reproduced(
                        evidence,
                        "reduced classification outputs no longer exist",
                    )
                reference = reference[:, indexes]
                baseline = baseline[:, indexes]
                current = current[:, indexes]
            try:
                decision = decide_three_way(
                    candidate.output_name,
                    reference,
                    baseline,
                    current,
                    policy=candidate.policy,
                    semantics=(
                        "classification" if candidate.semantics == "classification" else None
                    ),
                )
            except ThreeWayPrecedenceError:
                return self._not_reproduced(evidence, "three-way precedence changed")
            if decision.failure_code is not FailureCode.NUMERICAL_REGRESSION:
                return self._not_reproduced(evidence, "numerical regression did not reproduce")
            return PredicateObservation(
                outcome=PredicateOutcome.REPRODUCED,
                failure_code=FailureCode.NUMERICAL_REGRESSION,
                predicate_signature_sha256=self.failure.signature_sha256,
                evidence_sha256=tuple(evidence),
            )
        except (
            InfrastructureError,
            InvalidInputError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            return PredicateObservation(
                outcome=PredicateOutcome.INFRASTRUCTURE_INVALID,
                detail=f"{type(error).__name__}: {error}",
            )

    def _environment(self, environment_id: str) -> EnvironmentLock:
        matches = tuple(item for item in self.matrix.environments if item.id == environment_id)
        if len(matches) != 1:
            raise InfrastructureError("domain predicate environment is absent from matrix")
        return matches[0]

    def _not_reproduced(self, evidence: Sequence[str], detail: str) -> PredicateObservation:
        return PredicateObservation(
            outcome=PredicateOutcome.NOT_REPRODUCED,
            evidence_sha256=tuple(evidence),
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class DomainReductionResult:
    """Reduced candidate and complete hash-chained session."""

    candidate: DomainNumericalCandidate
    session: ReductionSessionManifest


def run_domain_numerical_reduction(
    *,
    project: Path,
    matrix: MatrixLock,
    failure: FailureRecord,
    original: DomainNumericalCandidate,
    retained_baseline_output: Path,
    retained_candidate_output: Path,
    output: Path,
    limits: ReductionLimits,
    predicate: DomainNumericalGpuPredicate | None = None,
) -> DomainReductionResult:
    """Confirm, reduce, and clean-replay one real numerical qualification failure."""

    original.verify_artifacts()
    _verify(
        retained_baseline_output,
        sha256_file(retained_baseline_output),
        "retained baseline",
    )
    _verify(retained_candidate_output, sha256_file(retained_candidate_output), "retained output")
    execution = predicate or DomainNumericalGpuPredicate(
        project=project,
        matrix=matrix,
        failure=failure,
        evidence_root=output / "attempts",
    )
    environment_identities = tuple(
        ReductionEnvironmentIdentity(
            id=item.id,
            worker_manifest_sha256=item.worker_image.manifest_digest,
        )
        for item in matrix.environments
    )
    assert len(environment_identities) == 2
    unvalidated_contract = ReductionPredicateContract.model_construct(
        failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=failure.signature_sha256,
        environments=(environment_identities[0], environment_identities[1]),
        model_sha256=original.model_sha256,
        executor_sha256=None,
        output_name=original.output_name,
        concrete_shapes=tuple(
            ReductionShapeIdentity(input_name=item.name, dimensions=item.shape)
            for item in original.inputs
        ),
        input_sha256=tuple(item.sha256 for item in original.inputs),
        threshold_relationship=failure.threshold or "the locked numerical policy must pass",
        confirmation_count=limits.confirmation_count,
        predicate_sha256="sha256:" + "0" * 64,
    )
    contract = ReductionPredicateContract.model_validate(
        unvalidated_contract.model_copy(
            update={"predicate_sha256": unvalidated_contract.computed_sha256()}
        )
    )
    reducers: dict[
        ReductionStage,
        Callable[[DomainNumericalCandidate], Sequence[DomainNumericalCandidate]],
    ] = dict.fromkeys(REDUCTION_STAGES, _no_candidates)

    reference = np.load(original.reference_path, allow_pickle=False)
    baseline_value = np.load(retained_baseline_output, allow_pickle=False)
    candidate_value = np.load(retained_candidate_output, allow_pickle=False)
    if reference.shape != candidate_value.shape or baseline_value.shape != reference.shape:
        raise InvalidInputError("retained numerical arrays have different shapes")
    if original.semantics == "classification":
        retained_indexes: set[int] = set()
        if reference.ndim != 2:
            raise InvalidInputError("classification reduction requires batched class scores")
        class_count = reference.shape[1]
        for value in (reference, baseline_value, candidate_value):
            count = min(5, class_count)
            for row in value:
                retained_indexes.update(int(item) for item in np.argpartition(row, -count)[-count:])
        reference_tolerance = original.policy.candidate_to_reference
        drift_tolerance = original.policy.candidate_to_baseline
        for mask in (
            np.abs(candidate_value - reference)
            > reference_tolerance.atol + reference_tolerance.rtol * np.abs(reference),
            np.abs(candidate_value - baseline_value)
            > drift_tolerance.atol + drift_tolerance.rtol * np.abs(baseline_value),
        ):
            retained_indexes.update(int(item) for item in np.nonzero(mask)[1])
        reduced_indexes = tuple(sorted(retained_indexes))
        if len(reduced_indexes) >= class_count:
            raise InvalidInputError("classification failure has no smaller output subset")
        reducers[ReductionStage.OUTPUTS] = lambda candidate: (
            candidate.model_copy(update={"classification_indexes": reduced_indexes}),
        )
    else:
        reference_tolerance = original.policy.candidate_to_reference
        drift_tolerance = original.policy.candidate_to_baseline
        failing = np.union1d(
            np.flatnonzero(
                np.abs(candidate_value - reference)
                > reference_tolerance.atol + reference_tolerance.rtol * np.abs(reference)
            ),
            np.flatnonzero(
                np.abs(candidate_value - baseline_value)
                > drift_tolerance.atol + drift_tolerance.rtol * np.abs(baseline_value)
            ),
        )
        if failing.size == 0:
            raise InvalidInputError("retained candidate output does not preserve the failure")
        reduced_index = int(failing[0])
        reducers[ReductionStage.OUTPUTS] = lambda candidate: (
            candidate.model_copy(update={"comparison_flat_index": reduced_index}),
        )
    reducers[ReductionStage.BUILDER_OPTIONS] = lambda candidate: (
        candidate.model_copy(
            update={
                "workspace_bytes": min(candidate.workspace_bytes, 256 * 1024**2),
                "optimization_level": 0,
            }
        ),
    )
    reducers[ReductionStage.ENVIRONMENT_HISTORY] = lambda candidate: (
        candidate.model_copy(
            update={
                "environment_boundary": LockedEnvironmentBoundary(
                    last_passing=candidate.environment_history[0],
                    first_failing=candidate.environment_history[1],
                    passing_evidence_sha256=(sha256_file(retained_baseline_output),),
                    failing_evidence_sha256=(sha256_file(retained_candidate_output),),
                )
            }
        ),
    )
    machine = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=failure.signature_sha256,
        predicate_contract=contract,
        predicate=execution.evaluate,
        reducers=reducers,
        clean_replay=lambda candidate, clean: execution.evaluate(candidate, clean),
        candidate_sha256=DomainNumericalCandidate.candidate_sha256,
        limits=limits,
    )
    output.mkdir(parents=True, exist_ok=True)
    replay_parent = output / "clean-replay"
    replay_parent.mkdir(exist_ok=True)
    result = machine.run(original, replay_parent=replay_parent)
    if result.manifest.status is not ReductionStatus.COMPLETED:
        if result.manifest.status is ReductionStatus.INFRASTRUCTURE_INVALID:
            raise InfrastructureError("domain numerical reduction infrastructure was invalid")
        raise InvalidInputError("domain numerical reduction exhausted its locked budget")
    write_session_manifest(output / "session.json", result.manifest)
    write_json_atomic(output / "candidate.json", result.candidate.model_dump(mode="json"))
    return DomainReductionResult(result.candidate, result.manifest)


def _no_candidates(candidate: DomainNumericalCandidate) -> tuple[()]:
    del candidate
    return ()


def _verify(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise InvalidInputError(f"domain numerical {label} identity differs")


def _load_build(path: Path, command: tuple[str, ...]) -> WorkerBuildResult:
    try:
        value = WorkerBuildResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InfrastructureError("domain build result is malformed") from error
    if value.command != command or value.command_sha256 != command_sha256(command):
        raise InfrastructureError("domain build command identity differs")
    return value


def _load_correctness(path: Path, command: tuple[str, ...]) -> WorkerCorrectnessResult:
    try:
        value = WorkerCorrectnessResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InfrastructureError("domain correctness result is malformed") from error
    if value.command != command or value.command_sha256 != command_sha256(command):
        raise InfrastructureError("domain correctness command identity differs")
    return value


def _worker_outputs(
    result: WorkerCorrectnessResult,
    *,
    root: Path,
    output_name: str,
) -> tuple[np.ndarray[Any, Any], ...]:
    values = []
    resolved_root = root.resolve(strict=True)
    for repetition in result.repetitions:
        matches = tuple(item for item in repetition.outputs if item.name == output_name)
        if len(matches) != 1:
            raise InfrastructureError("domain worker output schema differs")
        artifact = matches[0]
        relative = Path(artifact.path).relative_to("/output")
        path = root / relative
        resolved = path.resolve(strict=True)
        if (
            path.is_symlink()
            or not resolved.is_relative_to(resolved_root)
            or sha256_file(resolved) != artifact.sha256
        ):
            raise InfrastructureError("domain worker output identity differs")
        values.append(np.load(resolved, allow_pickle=False))
    return tuple(values)


def _repetitions_are_stable(
    values: Sequence[np.ndarray[Any, Any]],
    policy: DeterminismPolicy,
) -> bool:
    """Apply the exact authored determinism policy to repeated worker outputs."""

    if not values:
        return False
    reference = values[0]
    if policy.require_bitwise:
        return all(np.array_equal(reference, value, equal_nan=False) for value in values[1:])
    return all(
        np.allclose(
            reference,
            value,
            atol=policy.tolerance.atol,
            rtol=policy.tolerance.rtol,
            equal_nan=False,
        )
        for value in values[1:]
    )
