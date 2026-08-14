"""End-to-end candidate-aware reduction for the remote seeded GPU gates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from upgrade_guard.contracts.base import canonical_json_bytes, sha256_bytes, sha256_file
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode, InfrastructureError, InvalidInputError
from upgrade_guard.reduce.candidate import (
    EnvironmentHistoryNotApplicable,
    EnvironmentHistoryObservation,
    G2ReductionCandidate,
    G7ReductionCandidate,
    LockedEnvironmentBoundary,
)
from upgrade_guard.reduce.general import ReductionLimits, TrialOutcome, reduce_environment_history
from upgrade_guard.reduce.gpu import CandidateGpuPredicate
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


@dataclass(frozen=True, slots=True)
class RemoteReductionResult:
    """Final execution-driving candidates and their complete session evidence."""

    g2: G2ReductionCandidate
    g2_session: ReductionSessionManifest
    g7: G7ReductionCandidate
    g7_session: ReductionSessionManifest


def run_remote_reductions(
    *,
    project: Path,
    state: Path,
    core_corpus: Path,
    plugin_corpus: Path,
    matrix: MatrixLock,
    signature_sha256: str,
    output: Path,
    predicate: CandidateGpuPredicate | None = None,
    limits: ReductionLimits | None = None,
) -> RemoteReductionResult:
    """Reduce G2 and G7 only through re-executed exact GPU predicates."""

    output.mkdir(parents=True, exist_ok=True)
    execution = predicate or CandidateGpuPredicate(
        project=project,
        state=state,
        matrix=matrix,
        signature_sha256=signature_sha256,
        evidence_root=output / "attempts" / f"invocation-{uuid4().hex}",
    )
    bounded = limits or ReductionLimits(
        maximum_trials=200,
        maximum_seconds=7200,
        confirmation_count=2,
    )
    candidate_environment = matrix.environments[1].id
    environment_history = tuple(environment.id for environment in matrix.environments)
    environment_pair = tuple(
        ReductionEnvironmentIdentity(
            id=environment.id,
            worker_manifest_sha256=environment.worker_image.manifest_digest,
        )
        for environment in matrix.environments
    )
    if len(environment_pair) != 2:
        raise InfrastructureError("remote reduction requires exactly two locked environments")
    g2_model = plugin_corpus / "residual-rmsnorm-fp32.onnx"
    g2_executor = state / "plugin-build" / "candidate" / "build" / "upgrade_guard_gpu_faults"
    for artifact in (g2_model, g2_executor):
        if not artifact.is_file() or artifact.is_symlink():
            raise InfrastructureError(f"remote reduction artifact is unavailable: {artifact.name}")
    g2_original = G2ReductionCandidate(
        rows=1,
        hidden=259,
        x_value=0.5,
        residual_value=0.25,
        gamma_value=1.0,
        environment_id=candidate_environment,
        environment_history=environment_history,
    )
    g2_contract = _predicate_contract(
        failure_code=FailureCode.NUMERICAL_REGRESSION,
        signature_sha256=signature_sha256,
        environments=environment_pair,
        model_sha256=sha256_file(g2_model),
        executor_sha256=sha256_file(g2_executor),
        output_name="residual_rmsnorm",
        shapes=(ReductionShapeIdentity(input_name="x", dimensions=(1, 1, 259)),),
        input_sha256=(sha256_bytes(canonical_json_bytes([0.5, 0.25, 1.0])),),
        threshold_relationship="absolute error exceeds 0.1 while the clean control passes",
        confirmation_count=bounded.confirmation_count,
    )
    resumed_g2 = _load_completed_g2(output, g2_contract)
    if resumed_g2 is None:
        g2_machine = ReductionStateMachine(
            expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature_sha256,
            predicate_contract=g2_contract,
            predicate=execution.evaluate_g2,
            reducers=_g2_reducers(execution, bounded, signature_sha256),
            clean_replay=lambda candidate, clean: execution.evaluate_g2(candidate, clean),
            candidate_sha256=G2ReductionCandidate.candidate_sha256,
            limits=bounded,
        )
        g2 = g2_machine.run(g2_original, replay_parent=output / "clean-replay")
        _require_completed("G2", g2.manifest)
        write_session_manifest(output / "G2-session.json", g2.manifest)
        _write_candidate(output / "G2-candidate.json", g2.candidate)
        g2_candidate = g2.candidate
        g2_manifest = g2.manifest
    else:
        g2_candidate, g2_manifest = resumed_g2

    tokens = state / "fault-inputs" / "g7" / "tokens.npy"
    mask = state / "fault-inputs" / "g7" / "mask.npy"
    model = core_corpus / "models" / "tiny-transformer-fp32.onnx"
    for artifact in (tokens, mask, model):
        if not artifact.is_file() or artifact.is_symlink():
            raise InfrastructureError(f"remote reduction artifact is unavailable: {artifact.name}")
    g7_original = G7ReductionCandidate(
        model_path=model,
        model_sha256=sha256_file(model),
        batch=9,
        sequence=8,
        hidden=256,
        profile_min_batch=1,
        profile_opt_batch=4,
        profile_max_batch=8,
        profile_min_sequence=8,
        profile_opt_sequence=128,
        profile_max_sequence=512,
        input_mode="original",
        tokens_path=tokens,
        tokens_sha256=sha256_file(tokens),
        mask_path=mask,
        mask_sha256=sha256_file(mask),
        workspace_bytes=2 * 1024**3,
        optimization_level=3,
        environment_id=candidate_environment,
        environment_history=environment_history,
    )
    g7_machine = ReductionStateMachine(
        expected_failure_code=FailureCode.PROFILE_REJECTED,
        predicate_signature_sha256=signature_sha256,
        predicate_contract=_predicate_contract(
            failure_code=FailureCode.PROFILE_REJECTED,
            signature_sha256=signature_sha256,
            environments=environment_pair,
            model_sha256=sha256_file(model),
            executor_sha256=None,
            output_name=None,
            shapes=(
                ReductionShapeIdentity(input_name="tokens", dimensions=(9, 8, 256)),
                ReductionShapeIdentity(input_name="mask", dimensions=(9, 1, 1, 8)),
            ),
            input_sha256=(sha256_file(tokens), sha256_file(mask)),
            threshold_relationship=(
                "tokens batch 9 exceeds profile maximum batch 8 while an in-profile control passes"
            ),
            confirmation_count=bounded.confirmation_count,
        ),
        predicate=execution.evaluate_g7,
        reducers=_g7_reducers(execution, bounded, signature_sha256),
        clean_replay=lambda candidate, clean: execution.evaluate_g7(candidate, clean),
        candidate_sha256=G7ReductionCandidate.candidate_sha256,
        limits=bounded,
    )
    g7 = g7_machine.run(g7_original, replay_parent=output / "clean-replay")
    _require_completed("G7", g7.manifest)
    write_session_manifest(output / "G7-session.json", g7.manifest)
    portable_g7 = g7.candidate.model_copy(
        update={
            "model_path": Path("model.onnx"),
            "tokens_path": Path("inputs/tokens.npy")
            if g7.candidate.input_mode == "original"
            else None,
            "mask_path": Path("inputs/mask.npy") if g7.candidate.input_mode == "original" else None,
        }
    )
    (output / "G7-candidate.json").write_text(
        portable_g7.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return RemoteReductionResult(g2_candidate, g2_manifest, g7.candidate, g7.manifest)


G2Reducer = Callable[[G2ReductionCandidate], Sequence[G2ReductionCandidate]]
G7Reducer = Callable[[G7ReductionCandidate], Sequence[G7ReductionCandidate]]


def _g2_reducers(
    execution: CandidateGpuPredicate,
    limits: ReductionLimits,
    predicate_signature_sha256: str,
) -> dict[ReductionStage, G2Reducer]:
    def empty(candidate: G2ReductionCandidate) -> tuple[G2ReductionCandidate, ...]:
        del candidate
        return ()

    def outputs(candidate: G2ReductionCandidate) -> tuple[G2ReductionCandidate, ...]:
        return (candidate.model_copy(update={"outputs": ("G2",)}),)

    def inputs(candidate: G2ReductionCandidate) -> tuple[G2ReductionCandidate, ...]:
        return (
            candidate.model_copy(
                update={"x_value": 0.0, "residual_value": 1.0, "gamma_value": 1.0}
            ),
            candidate.model_copy(
                update={"x_value": 1.0, "residual_value": 1.0, "gamma_value": 1.0}
            ),
        )

    reducers: dict[ReductionStage, G2Reducer] = dict.fromkeys(REDUCTION_STAGES, empty)
    reducers[ReductionStage.OUTPUTS] = outputs
    reducers[ReductionStage.INPUTS] = inputs
    reducers[ReductionStage.ENVIRONMENT_HISTORY] = lambda candidate: (
        _reduce_g2_environment_history(candidate, execution, limits, predicate_signature_sha256),
    )
    return reducers


def _g7_reducers(
    execution: CandidateGpuPredicate,
    limits: ReductionLimits,
    predicate_signature_sha256: str,
) -> dict[ReductionStage, G7Reducer]:
    def empty(candidate: G7ReductionCandidate) -> tuple[G7ReductionCandidate, ...]:
        del candidate
        return ()

    def profile(candidate: G7ReductionCandidate) -> tuple[G7ReductionCandidate, ...]:
        return (
            candidate.model_copy(
                update={
                    "profile_opt_sequence": candidate.profile_min_sequence,
                    "profile_max_sequence": candidate.profile_min_sequence,
                }
            ),
        )

    def inputs(candidate: G7ReductionCandidate) -> tuple[G7ReductionCandidate, ...]:
        common = {
            "tokens_path": None,
            "tokens_sha256": None,
            "mask_path": None,
            "mask_sha256": None,
        }
        return (
            candidate.model_copy(update={**common, "input_mode": "zeros"}),
            candidate.model_copy(update={**common, "input_mode": "ones"}),
        )

    def options(candidate: G7ReductionCandidate) -> tuple[G7ReductionCandidate, ...]:
        return (
            candidate.model_copy(
                update={"workspace_bytes": 256 * 1024**2, "optimization_level": 0}
            ),
        )

    def transform(operation: str, maximum_seconds: float) -> G7Reducer:
        def apply(candidate: G7ReductionCandidate) -> tuple[G7ReductionCandidate, ...]:
            return (
                execution.transform_g7(
                    candidate,
                    operation,
                    maximum_seconds=maximum_seconds,
                ),
            )

        return apply

    reducers: dict[ReductionStage, G7Reducer] = dict.fromkeys(REDUCTION_STAGES, empty)
    reducers[ReductionStage.OPTIMIZATION_PROFILE] = profile
    reducers[ReductionStage.INPUTS] = inputs
    reducers[ReductionStage.BUILDER_OPTIONS] = options
    reducers[ReductionStage.FREEZE_DYNAMIC_SHAPE] = transform("freeze", 300)
    reducers[ReductionStage.FOLD_SHAPE_OPERATIONS] = transform("fold", 300)
    reducers[ReductionStage.POLYGRAPHY_BISECT] = transform("bisect", 900)
    reducers[ReductionStage.POLYGRAPHY_LINEAR] = transform("linear", 900)
    reducers[ReductionStage.ENVIRONMENT_HISTORY] = lambda candidate: (
        _reduce_g7_environment_history(candidate, execution, limits, predicate_signature_sha256),
    )
    return reducers


def _reduce_g2_environment_history(
    candidate: G2ReductionCandidate,
    execution: CandidateGpuPredicate,
    limits: ReductionLimits,
    predicate_signature_sha256: str,
) -> G2ReductionCandidate:
    observations: dict[str, list[PredicateObservation]] = {}

    def trial(environment_id: str) -> TrialOutcome:
        observation = execution.evaluate_g2_environment(candidate, environment_id)
        observations.setdefault(environment_id, []).append(observation)
        return _trial_outcome(
            observation,
            FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256,
        )

    try:
        boundary = reduce_environment_history(candidate.environment_history, trial, limits)
    except InvalidInputError:
        not_applicable = _not_applicable(
            candidate.environment_history,
            observations,
            FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256,
            limits.confirmation_count,
        )
        if not_applicable is None:
            raise
        reduced = candidate.model_dump(mode="python")
        reduced.update(
            environment_boundary=None,
            environment_history_not_applicable=not_applicable,
        )
        return G2ReductionCandidate.model_validate(reduced)
    reduced = candidate.model_dump(mode="python")
    reduced.update(
        environment_id=boundary.first_failing,
        environment_history=(boundary.last_passing, boundary.first_failing),
        environment_history_not_applicable=None,
        environment_boundary=LockedEnvironmentBoundary(
            last_passing=boundary.last_passing,
            first_failing=boundary.first_failing,
            passing_evidence_sha256=_evidence_for(observations, boundary.last_passing),
            failing_evidence_sha256=_evidence_for(observations, boundary.first_failing),
        ),
    )
    return G2ReductionCandidate.model_validate(reduced)


def _reduce_g7_environment_history(
    candidate: G7ReductionCandidate,
    execution: CandidateGpuPredicate,
    limits: ReductionLimits,
    predicate_signature_sha256: str,
) -> G7ReductionCandidate:
    observations: dict[str, list[PredicateObservation]] = {}

    def trial(environment_id: str) -> TrialOutcome:
        observation = execution.evaluate_g7_environment(candidate, environment_id)
        observations.setdefault(environment_id, []).append(observation)
        return _trial_outcome(
            observation,
            FailureCode.PROFILE_REJECTED,
            predicate_signature_sha256,
        )

    try:
        boundary = reduce_environment_history(candidate.environment_history, trial, limits)
    except InvalidInputError:
        not_applicable = _not_applicable(
            candidate.environment_history,
            observations,
            FailureCode.PROFILE_REJECTED,
            predicate_signature_sha256,
            limits.confirmation_count,
        )
        if not_applicable is None:
            raise
        reduced = candidate.model_dump(mode="python")
        reduced.update(
            environment_boundary=None,
            environment_history_not_applicable=not_applicable,
        )
        return G7ReductionCandidate.model_validate(reduced)
    reduced = candidate.model_dump(mode="python")
    reduced.update(
        environment_id=boundary.first_failing,
        environment_history=(boundary.last_passing, boundary.first_failing),
        environment_history_not_applicable=None,
        environment_boundary=LockedEnvironmentBoundary(
            last_passing=boundary.last_passing,
            first_failing=boundary.first_failing,
            passing_evidence_sha256=_evidence_for(observations, boundary.last_passing),
            failing_evidence_sha256=_evidence_for(observations, boundary.first_failing),
        ),
    )
    return G7ReductionCandidate.model_validate(reduced)


def _trial_outcome(
    observation: PredicateObservation,
    expected_failure_code: FailureCode,
    expected_signature_sha256: str,
) -> TrialOutcome:
    if observation.outcome is PredicateOutcome.REPRODUCED:
        if (
            observation.failure_code is not expected_failure_code
            or observation.predicate_signature_sha256 != expected_signature_sha256
        ):
            raise InvalidInputError("environment-history observation changed the stable predicate")
        return TrialOutcome.REPRODUCED
    if observation.outcome is PredicateOutcome.NOT_REPRODUCED:
        return TrialOutcome.NOT_REPRODUCED
    raise InfrastructureError(
        "environment-history predicate was infrastructure-invalid",
        details={"detail": observation.detail},
    )


def _evidence_for(
    observations: dict[str, list[PredicateObservation]], environment_id: str
) -> tuple[str, ...]:
    retained = tuple(
        dict.fromkeys(
            digest
            for observation in observations.get(environment_id, ())
            for digest in observation.evidence_sha256
        )
    )
    if not retained:
        raise InfrastructureError(f"environment boundary retained no evidence for {environment_id}")
    return retained


def _not_applicable(
    environment_history: tuple[str, ...],
    observations: dict[str, list[PredicateObservation]],
    expected_failure_code: FailureCode,
    expected_signature_sha256: str,
    confirmation_count: int,
) -> EnvironmentHistoryNotApplicable | None:
    retained: list[EnvironmentHistoryObservation] = []
    for environment_id in environment_history:
        trials = observations.get(environment_id, ())
        if len(trials) != confirmation_count or any(
            trial.outcome is not PredicateOutcome.REPRODUCED
            or trial.failure_code is not expected_failure_code
            or trial.predicate_signature_sha256 != expected_signature_sha256
            or not trial.evidence_sha256
            for trial in trials
        ):
            return None
        retained.append(
            EnvironmentHistoryObservation(
                environment_id=environment_id,
                outcome="reproduced",
                failure_code=expected_failure_code,
                predicate_signature_sha256=expected_signature_sha256,
                confirmation_count=confirmation_count,
                trial_evidence_sha256=tuple(trial.evidence_sha256 for trial in trials),
            )
        )
    return EnvironmentHistoryNotApplicable(observations=tuple(retained))


def _require_completed(seed: str, manifest: ReductionSessionManifest) -> None:
    if manifest.status is not ReductionStatus.COMPLETED:
        detail = {
            "seed": seed,
            "status": manifest.status.value,
            "attempts": len(manifest.attempts),
        }
        if (
            manifest.attempts
            and manifest.attempts[-1].outcome is PredicateOutcome.INFRASTRUCTURE_INVALID
        ):
            detail["last_error"] = manifest.attempts[-1].detail
        raise InfrastructureError(
            f"{seed} candidate-aware reduction did not complete",
            details=detail,
        )


def _load_completed_g2(
    output: Path,
    predicate: ReductionPredicateContract,
) -> tuple[G2ReductionCandidate, ReductionSessionManifest] | None:
    candidate_path = output / "G2-candidate.json"
    session_path = output / "G2-session.json"
    if not candidate_path.exists() and not session_path.exists():
        return None
    try:
        if candidate_path.is_symlink() or session_path.is_symlink():
            raise ValueError("resumed G2 artifacts cannot be symlinks")
        candidate = G2ReductionCandidate.model_validate_json(
            candidate_path.read_text(encoding="utf-8")
        )
        session = ReductionSessionManifest.model_validate_json(
            session_path.read_text(encoding="utf-8")
        )
        if (
            session.status is not ReductionStatus.COMPLETED
            or session.predicate != predicate
            or session.final_candidate_sha256 != candidate.candidate_sha256()
            or (
                candidate.environment_boundary is None
                and candidate.environment_history_not_applicable is None
            )
        ):
            raise ValueError("resumed G2 reduction identity does not match")
    except (OSError, UnicodeDecodeError, ValueError):
        stale = output / "stale" / f"G2-{uuid4().hex}"
        stale.mkdir(parents=True)
        for path in (candidate_path, session_path):
            if path.exists() or path.is_symlink():
                path.replace(stale / path.name)
        return None
    return candidate, session


def _write_candidate(path: Path, candidate: G2ReductionCandidate) -> None:
    write_json_atomic(path, candidate.model_dump(mode="json"))


def _predicate_contract(
    *,
    failure_code: FailureCode,
    signature_sha256: str,
    environments: tuple[ReductionEnvironmentIdentity, ReductionEnvironmentIdentity],
    model_sha256: str,
    executor_sha256: str | None,
    output_name: str | None,
    shapes: tuple[ReductionShapeIdentity, ...],
    input_sha256: tuple[str, ...],
    threshold_relationship: str,
    confirmation_count: int,
) -> ReductionPredicateContract:
    predicate = ReductionPredicateContract.model_construct(
        failure_code=failure_code,
        predicate_signature_sha256=signature_sha256,
        environments=environments,
        model_sha256=model_sha256,
        executor_sha256=executor_sha256,
        output_name=output_name,
        concrete_shapes=shapes,
        input_sha256=input_sha256,
        threshold_relationship=threshold_relationship,
        confirmation_count=confirmation_count,
        infrastructure_satisfies_predicate=False,
        predicate_sha256="sha256:" + "0" * 64,
    )
    return ReductionPredicateContract.model_validate(
        predicate.model_copy(update={"predicate_sha256": predicate.computed_sha256()})
    )
