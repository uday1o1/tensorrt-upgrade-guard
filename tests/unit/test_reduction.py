"""Stable numerical and profile reduction tests."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.runtime import WorkerMounts
from upgrade_guard.contracts.base import canonical_json_bytes, sha256_bytes, sha256_file
from upgrade_guard.errors import FailureCode, InvalidInputError
from upgrade_guard.reduce.candidate import G2ReductionCandidate, G7ReductionCandidate
from upgrade_guard.reduce.general import (
    ConfirmedEvaluator,
    ReductionLimits,
    TrialOutcome,
    reduce_environment_history,
    reduce_sequence,
    simplify_finite_input,
)
from upgrade_guard.reduce.gpu import CandidateGpuPredicate
from upgrade_guard.reduce.inputs import reduce_numerical_failure
from upgrade_guard.reduce.performance import reduce_performance_failure
from upgrade_guard.reduce.polygraphy import reduction_commands, run_polygraphy_reduction
from upgrade_guard.reduce.predicate import ProfilePredicate
from upgrade_guard.reduce.profile_graph import _reduction_command
from upgrade_guard.reduce.remote import run_remote_reductions
from upgrade_guard.reduce.session import reduce_failure_directory
from upgrade_guard.reduce.shapes import reduce_profile_failure
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


def _predicate_contract(
    signature: str,
    code: FailureCode = FailureCode.NUMERICAL_REGRESSION,
) -> ReductionPredicateContract:
    predicate = ReductionPredicateContract.model_construct(
        failure_code=code,
        predicate_signature_sha256=signature,
        environments=(
            ReductionEnvironmentIdentity(
                id="baseline", worker_manifest_sha256="sha256:" + "1" * 64
            ),
            ReductionEnvironmentIdentity(
                id="candidate", worker_manifest_sha256="sha256:" + "2" * 64
            ),
        ),
        model_sha256="sha256:" + "3" * 64,
        output_name="output",
        concrete_shapes=(ReductionShapeIdentity(input_name="input", dimensions=(1,)),),
        input_sha256=("sha256:" + "4" * 64,),
        threshold_relationship="candidate exceeds the authored threshold",
        confirmation_count=2,
        infrastructure_satisfies_predicate=False,
        predicate_sha256="sha256:" + "0" * 64,
    )
    return ReductionPredicateContract.model_validate(
        predicate.model_copy(update={"predicate_sha256": predicate.computed_sha256()})
    )


def test_numerical_reducer_retains_strongest_threshold_violation() -> None:
    reference = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 1] += 0.1
    candidate[1, 1] += 0.5
    reduced = reduce_numerical_failure(reference, candidate, atol=1e-5, rtol=1e-4)
    assert reduced.original_shape == (2, 2)
    assert reduced.multidimensional_index == (1, 1)
    assert reduced.reference.shape == (1,)
    assert reduced.absolute_error > reduced.threshold


def test_numerical_reducer_rejects_nonfailing_evidence() -> None:
    array = np.ones(4, dtype=np.float32)
    with pytest.raises(InvalidInputError, match="do not satisfy"):
        reduce_numerical_failure(array, array.copy(), atol=1e-5, rtol=1e-4)


def test_profile_reducer_keeps_one_minimal_violation() -> None:
    reduced = reduce_profile_failure(
        ProfilePredicate(
            kind="profile",
            input_name="tokens",
            observed_shape=(9, 513, 256),
            minimum_shape=(1, 8, 256),
            maximum_shape=(8, 512, 256),
        )
    )
    assert reduced.observed_shape == (9, 8, 256)
    assert reduced.maximum_shape == (8, 8, 256)
    assert reduced.violating_dimension == 0


def test_public_reduction_writes_hash_addressed_smaller_arrays(tmp_path: Path) -> None:
    source = tmp_path / "failure"
    source.mkdir()
    reference = np.arange(8, dtype=np.float32)
    candidate = reference.copy()
    candidate[5] += 1
    np.save(source / "reference.npy", reference, allow_pickle=False)
    np.save(source / "candidate.npy", candidate, allow_pickle=False)
    (source / "reduction-request.json").write_text(
        json.dumps(
            {
                "api_version": "upgradeguard.dev/v1alpha1",
                "kind": "ReductionRequest",
                "failure_code": "NUMERICAL_REGRESSION",
                "signature_sha256": "sha256:" + "1" * 64,
                "confirmation_count": 2,
                "maximum_trials": 20,
                "maximum_seconds": 60,
                "predicate": {
                    "kind": "numerical",
                    "output_name": "output",
                    "reference_path": "reference.npy",
                    "candidate_path": "candidate.npy",
                    "atol": 1e-5,
                    "rtol": 1e-4,
                },
            }
        ),
        encoding="utf-8",
    )
    result = reduce_failure_directory(source, tmp_path / "reduced")
    assert result["failure_code"] == "NUMERICAL_REGRESSION"
    assert np.load(tmp_path / "reduced" / "candidate.npy").shape == (1,)


def test_polygraphy_uses_bisect_then_linear_with_argument_arrays(tmp_path: Path) -> None:
    commands = reduction_commands(
        model=tmp_path / "model.onnx",
        output=tmp_path / "reduced.onnx",
        predicate_command=("upgrade-guard", "dev", "predicate", "predicate.json"),
    )
    assert commands[0].index("--mode=bisect") < commands[0].index("--check")
    assert commands[1].index("--mode=linear") < commands[1].index("--check")
    assert commands[0][commands[0].index("--fail-code") + 1] == "86"
    assert "polygraphy" == commands[0][0]
    assert str(tmp_path / "reduced.bisect.onnx") in commands[0]
    assert str(tmp_path / "reduced.bisect.onnx") in commands[1]
    assert str(tmp_path / "model.onnx") not in commands[1]


def test_polygraphy_profile_checker_uses_fixed_artifact_after_all_reduce_options() -> None:
    arguments = Namespace(
        model=Path("/corpus/model.onnx"),
        output=Path("/output/reduced.onnx"),
        operation="bisect",
        profile=Path("/corpus/profile.json"),
        control_tokens=Path("/corpus/control-tokens.npy"),
        control_mask=Path("/corpus/control-mask.npy"),
        failure_tokens=Path("/corpus/failure-tokens.npy"),
        failure_mask=Path("/corpus/failure-mask.npy"),
        workspace_bytes=1024,
        optimization_level=0,
        signature="sha256:" + "a" * 64,
    )
    command = _reduction_command(arguments, Path("/output/checks.jsonl"))
    check_index = command.index("--check")
    assert command.index("--mode=bisect") < check_index
    assert command.index("--fail-code") < check_index
    assert command[check_index + 1 : check_index + 4] == (
        "python3",
        "-m",
        "upgrade_guard.reduce.profile_check",
    )


def test_gpu_predicates_materially_pass_candidate_fields_to_locked_worker(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    evidence = tmp_path / "evidence"
    project.mkdir()
    state.mkdir()
    signature = "sha256:" + "a" * 64
    matrix = SimpleNamespace(
        gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        environments=(
            SimpleNamespace(
                id="baseline",
                worker_image=SimpleNamespace(
                    canonical_reference="registry/worker@sha256:" + "1" * 64
                ),
            ),
            SimpleNamespace(
                id="candidate",
                worker_image=SimpleNamespace(
                    canonical_reference="registry/worker@sha256:" + "2" * 64
                ),
            ),
        ),
    )

    class FakeWorker:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        def run(self, **kwargs: object) -> CommandResult:
            command: tuple[str, ...] = tuple(cast(Sequence[str], kwargs["command"]))
            mounts = cast(WorkerMounts, kwargs["mounts"])
            assert mounts.state == state
            self.commands.append(command)
            if command[0].endswith("upgrade_guard_gpu_faults"):
                rows = int(command[command.index("--rows") + 1])
                hidden = int(command[command.index("--hidden") + 1])
                payload = {
                    "G2": {
                        "detected": True,
                        "control": "passed",
                        "rows": rows,
                        "hidden": hidden,
                    }
                }
                return CommandResult(command, 0, json.dumps(payload), "", 0.1)
            output = mounts.output
            output.mkdir(parents=True, exist_ok=True)
            if "upgrade_guard.worker.build_engine" in command:
                (output / "build.json").write_text('{"status":"passed"}\n', encoding="utf-8")
                return CommandResult(command, 0, "", "", 0.1)
            result_name = command[command.index("--result") + 1]
            result_path = output / Path(result_name).name
            if result_path.name == "control.json":
                result_path.write_text('{"status":"passed"}\n', encoding="utf-8")
                return CommandResult(command, 0, "", "", 0.1)
            result_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failure_code": "PROFILE_REJECTED",
                        "message": "input shape was rejected for tokens",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return CommandResult(command, 1, "", "", 0.1)

    worker = FakeWorker()
    predicate = CandidateGpuPredicate(
        project=project,
        state=state,
        matrix=matrix,  # type: ignore[arg-type]
        signature_sha256=signature,
        evidence_root=evidence,
        worker=worker,  # type: ignore[arg-type]
    )
    g2 = G2ReductionCandidate(
        outputs=("G2",),
        rows=3,
        hidden=259,
        x_value=0.0,
        residual_value=1.0,
        gamma_value=1.0,
        environment_id="candidate",
    )
    assert predicate.evaluate_g2(g2).outcome is PredicateOutcome.REPRODUCED
    g2_command = worker.commands[-1]
    assert g2_command[g2_command.index("--rows") + 1] == "3"
    assert g2_command[g2_command.index("--x-value") + 1] == "0"
    assert "--only-g2" in g2_command

    model = tmp_path / "model.onnx"
    tokens = tmp_path / "tokens.npy"
    mask = tmp_path / "mask.npy"
    model.write_bytes(b"model")
    np.save(tokens, np.zeros((9, 8, 256), np.float32), allow_pickle=False)
    np.save(mask, np.zeros((9, 1, 1, 8), np.float32), allow_pickle=False)
    g7 = G7ReductionCandidate(
        model_path=model,
        model_sha256=sha256_bytes(model.read_bytes()),
        batch=9,
        sequence=8,
        hidden=256,
        profile_min_batch=1,
        profile_opt_batch=1,
        profile_max_batch=8,
        profile_min_sequence=8,
        profile_opt_sequence=8,
        profile_max_sequence=8,
        input_mode="zeros",
        workspace_bytes=268435456,
        optimization_level=0,
        environment_id="candidate",
    )
    assert predicate.evaluate_g7(g7).outcome is PredicateOutcome.REPRODUCED
    build_command = next(
        command for command in worker.commands if "upgrade_guard.worker.build_engine" in command
    )
    assert build_command[build_command.index("--workspace-bytes") + 1] == "268435456"
    assert build_command[build_command.index("--optimization-level") + 1] == "0"
    trial_corpora = sorted(evidence.glob("*-G7/corpus"))
    assert trial_corpora
    assert np.load(trial_corpora[-1] / "failure-tokens.npy").shape == (9, 8, 256)
    assert not np.any(np.load(trial_corpora[-1] / "failure-tokens.npy"))


def test_remote_state_machine_reexecutes_every_stage_and_uses_final_candidates(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    core = tmp_path / "core"
    plugin = tmp_path / "plugin"
    project.mkdir()
    (state / "fault-inputs" / "g7").mkdir(parents=True)
    (state / "plugin-build" / "candidate" / "build").mkdir(parents=True)
    (core / "models").mkdir(parents=True)
    plugin.mkdir()
    model = core / "models" / "tiny-transformer-fp32.onnx"
    model.write_bytes(b"original")
    (plugin / "residual-rmsnorm-fp32.onnx").write_bytes(b"plugin-model")
    (state / "plugin-build" / "candidate" / "build" / "upgrade_guard_gpu_faults").write_bytes(
        b"executor"
    )
    np.save(state / "fault-inputs" / "g7" / "tokens.npy", np.zeros((9, 8, 256), np.float32))
    np.save(state / "fault-inputs" / "g7" / "mask.npy", np.zeros((9, 1, 1, 8), np.float32))
    signature = "sha256:" + "b" * 64

    class FakePredicate:
        def __init__(self) -> None:
            self.g2: list[G2ReductionCandidate] = []
            self.g7: list[G7ReductionCandidate] = []
            self.transforms: list[tuple[str, bytes]] = []

        def _result(self, code: FailureCode) -> PredicateObservation:
            return PredicateObservation(
                outcome=PredicateOutcome.REPRODUCED,
                failure_code=code,
                predicate_signature_sha256=signature,
                evidence_sha256=(signature,),
            )

        def evaluate_g2(
            self, candidate: G2ReductionCandidate, output: Path | None = None
        ) -> PredicateObservation:
            self.g2.append(candidate)
            if output is not None:
                (output / "evidence.json").write_text("{}\n", encoding="utf-8")
            return self._result(FailureCode.NUMERICAL_REGRESSION)

        def evaluate_g7(
            self, candidate: G7ReductionCandidate, output: Path | None = None
        ) -> PredicateObservation:
            self.g7.append(candidate)
            if output is not None:
                (output / "evidence.json").write_text("{}\n", encoding="utf-8")
            if candidate.model_path.read_bytes() == b"freeze":
                return PredicateObservation(outcome=PredicateOutcome.NOT_REPRODUCED)
            return self._result(FailureCode.PROFILE_REJECTED)

        def evaluate_g2_environment(
            self, candidate: G2ReductionCandidate, environment_id: str
        ) -> PredicateObservation:
            if environment_id == candidate.environment_id:
                return self.evaluate_g2(candidate)
            return PredicateObservation(
                outcome=PredicateOutcome.NOT_REPRODUCED,
                evidence_sha256=("sha256:" + "1" * 64,),
            )

        def evaluate_g7_environment(
            self, candidate: G7ReductionCandidate, environment_id: str
        ) -> PredicateObservation:
            if environment_id == candidate.environment_id:
                return self.evaluate_g7(candidate)
            return PredicateObservation(
                outcome=PredicateOutcome.NOT_REPRODUCED,
                evidence_sha256=("sha256:" + "2" * 64,),
            )

        def transform_g7(
            self,
            candidate: G7ReductionCandidate,
            operation: str,
            *,
            maximum_seconds: float,
        ) -> G7ReductionCandidate:
            assert maximum_seconds > 0
            self.transforms.append((operation, candidate.model_path.read_bytes()))
            transformed = tmp_path / f"{len(self.transforms)}-{operation}.onnx"
            transformed.write_bytes(operation.encode())
            return candidate.model_copy(
                update={
                    "model_path": transformed,
                    "model_sha256": sha256_bytes(operation.encode()),
                    "graph_history_sha256": (*candidate.graph_history_sha256, signature),
                }
            )

    fake = FakePredicate()
    matrix = SimpleNamespace(
        environments=(
            SimpleNamespace(
                id="baseline",
                worker_image=SimpleNamespace(manifest_digest="sha256:" + "1" * 64),
            ),
            SimpleNamespace(
                id="candidate",
                worker_image=SimpleNamespace(manifest_digest="sha256:" + "2" * 64),
            ),
        )
    )
    result = run_remote_reductions(
        project=project,
        state=state,
        core_corpus=core,
        plugin_corpus=plugin,
        matrix=matrix,  # type: ignore[arg-type]
        signature_sha256=signature,
        output=tmp_path / "reductions",
        predicate=fake,  # type: ignore[arg-type]
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=60, confirmation_count=2),
    )
    assert result.g2.outputs == ("G2",)
    assert (result.g2.x_value, result.g2.residual_value) == (0.0, 1.0)
    assert result.g7.profile_max_sequence == 8
    assert result.g7.input_mode == "zeros"
    assert result.g7.workspace_bytes == 256 * 1024**2
    assert result.g7.optimization_level == 0
    assert result.g7.model_path.read_bytes() == b"linear"
    assert result.g2.environment_history == ("baseline", "candidate")
    assert result.g2.environment_boundary is not None
    assert result.g7.environment_boundary is not None
    assert fake.transforms == [
        ("freeze", b"original"),
        ("fold", b"original"),
        ("bisect", b"fold"),
        ("linear", b"bisect"),
    ]
    assert set(result.g2_session.completed_stages) == {
        ReductionStage.CONFIRM_ORIGINAL,
        *REDUCTION_STAGES,
        ReductionStage.FINAL_EMPTY_REPLAY,
    }
    assert set(result.g7_session.completed_stages) == set(result.g2_session.completed_stages)
    for stage in REDUCTION_STAGES:
        assert any(attempt.stage is stage for attempt in result.g2_session.attempts)
        assert any(attempt.stage is stage for attempt in result.g7_session.attempts)
    g2_executions = len(fake.g2)
    resumed = run_remote_reductions(
        project=project,
        state=state,
        core_corpus=core,
        plugin_corpus=plugin,
        matrix=matrix,  # type: ignore[arg-type]
        signature_sha256=signature,
        output=tmp_path / "reductions",
        predicate=fake,  # type: ignore[arg-type]
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=60, confirmation_count=2),
    )
    assert len(fake.g2) == g2_executions
    assert resumed.g2 == result.g2


def test_gpu_fault_fixture_exposes_bounded_g2_candidate_parameters() -> None:
    source = Path("cpp/faults/gpu_faults.cu").read_text(encoding="utf-8")
    for option in (
        "--only-g2",
        "--rows",
        "--hidden",
        "--x-value",
        "--residual-value",
        "--gamma-value",
    ):
        assert option in source
    assert "numericalSeed(rows, hidden, xValue, residualValue, gammaValue)" in source


def test_polygraphy_executes_bisect_then_linear_under_one_budget(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"large-model")
    output = tmp_path / "reduced.onnx"

    class FakeRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(
            self,
            args: tuple[str, ...],
            *,
            timeout_seconds: float = 30.0,
            cwd: Path | None = None,
            env: object = None,
        ) -> CommandResult:
            del cwd, env
            assert 0 < timeout_seconds <= 10
            command = tuple(args)
            self.calls.append(command)
            destination = Path(command[command.index("--output") + 1])
            destination.write_bytes(b"bisect" if len(self.calls) == 1 else b"linear")
            return CommandResult(command, 0, "", "", 0.1)

    runner = FakeRunner()
    result = run_polygraphy_reduction(
        model=model,
        output=output,
        predicate_command=("upgrade-guard", "dev", "predicate", "predicate.json"),
        maximum_seconds=10,
        runner=runner,  # type: ignore[arg-type]
    )

    assert len(runner.calls) == 2
    assert str(result.bisect_model) in runner.calls[1]
    assert result.final_model.read_bytes() == b"linear"
    assert result.bisect_model_sha256 != result.final_model_sha256


def test_typed_state_machine_runs_every_reducer_and_clean_replay(tmp_path: Path) -> None:
    signature = "sha256:" + "a" * 64
    original = ("failure", *(stage.value for stage in REDUCTION_STAGES))
    reducer_calls: list[ReductionStage] = []
    clean_directories: list[Path] = []

    def candidate_hash(candidate: tuple[str, ...]) -> str:
        return sha256_bytes(canonical_json_bytes(candidate))

    def predicate(candidate: tuple[str, ...]) -> PredicateObservation:
        assert "failure" in candidate
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
            evidence_sha256=(candidate_hash(candidate),),
        )

    reducers = {}
    for expected_stage in REDUCTION_STAGES:

        def reduce_stage(
            candidate: tuple[str, ...], stage: ReductionStage = expected_stage
        ) -> tuple[tuple[str, ...], ...]:
            reducer_calls.append(stage)
            return (tuple(item for item in candidate if item != stage.value),)

        reducers[expected_stage] = reduce_stage

    def clean_replay(candidate: tuple[str, ...], directory: Path) -> PredicateObservation:
        assert not tuple(directory.iterdir())
        clean_directories.append(directory)
        (directory / "observed.json").write_text("{}\n", encoding="utf-8")
        return predicate(candidate)

    machine = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=signature,
        predicate_contract=_predicate_contract(signature),
        predicate=predicate,
        reducers=reducers,
        clean_replay=clean_replay,
        candidate_sha256=candidate_hash,
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
    )
    result = machine.run(original, replay_parent=tmp_path)

    assert result.candidate == ("failure",)
    assert result.manifest.status is ReductionStatus.COMPLETED
    assert reducer_calls == list(REDUCTION_STAGES)
    assert len(clean_directories) == 2
    assert all(not directory.exists() for directory in clean_directories)
    assert result.manifest.completed_stages == (
        ReductionStage.CONFIRM_ORIGINAL,
        *REDUCTION_STAGES,
        ReductionStage.FINAL_EMPTY_REPLAY,
    )
    assert result.manifest.session_sha256 == result.manifest.computed_sha256()
    for index, attempt in enumerate(result.manifest.attempts):
        assert attempt.attempt_sha256 == attempt.computed_sha256()
        if index:
            assert (
                attempt.previous_attempt_sha256
                == result.manifest.attempts[index - 1].attempt_sha256
            )
    manifest_path = tmp_path / "reduction-session.json"
    write_session_manifest(manifest_path, result.manifest)
    loaded = ReductionSessionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert loaded == result.manifest


def test_reduction_predicate_and_candidate_artifacts_fail_closed_on_tamper(
    tmp_path: Path,
) -> None:
    signature = "sha256:" + "c" * 64
    predicate = _predicate_contract(signature)
    with pytest.raises(ValueError, match="self-hash"):
        ReductionPredicateContract.model_validate(
            predicate.model_copy(update={"threshold_relationship": "weakened"})
        )

    model = tmp_path / "model.onnx"
    tokens = tmp_path / "tokens.npy"
    mask = tmp_path / "mask.npy"
    model.write_bytes(b"model")
    np.save(tokens, np.zeros((9, 8, 256), np.float32), allow_pickle=False)
    np.save(mask, np.zeros((9, 1, 1, 8), np.float32), allow_pickle=False)
    candidate = G7ReductionCandidate(
        model_path=model,
        model_sha256=sha256_bytes(model.read_bytes()),
        batch=9,
        sequence=8,
        hidden=256,
        profile_min_batch=1,
        profile_opt_batch=4,
        profile_max_batch=8,
        profile_min_sequence=8,
        profile_opt_sequence=128,
        profile_max_sequence=512,
        tokens_path=tokens,
        tokens_sha256=sha256_bytes(tokens.read_bytes()),
        mask_path=mask,
        mask_sha256=sha256_bytes(mask.read_bytes()),
        workspace_bytes=1024,
        optimization_level=3,
        environment_id="candidate",
    )
    candidate.verify_artifacts()
    model.write_bytes(b"tampered")
    with pytest.raises(InvalidInputError, match="hash"):
        candidate.verify_artifacts()


def test_state_machine_stops_on_infrastructure_invalid_trial(tmp_path: Path) -> None:
    signature = "sha256:" + "b" * 64
    original = ("failure", "inputs")

    def candidate_hash(candidate: tuple[str, ...]) -> str:
        return sha256_bytes(canonical_json_bytes(candidate))

    calls = 0

    def predicate(candidate: tuple[str, ...]) -> PredicateObservation:
        nonlocal calls
        calls += 1
        if candidate == ("failure",):
            return PredicateObservation(
                outcome=PredicateOutcome.INFRASTRUCTURE_INVALID,
                detail="GPU worker timed out",
            )
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
        )

    reducers = {
        stage: (
            (lambda candidate: (("failure",),))
            if stage is ReductionStage.INPUTS
            else (lambda candidate: ())
        )
        for stage in REDUCTION_STAGES
    }
    machine = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=signature,
        predicate_contract=_predicate_contract(signature),
        predicate=predicate,
        reducers=reducers,
        clean_replay=lambda candidate, directory: predicate(candidate),
        candidate_sha256=candidate_hash,
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
    )

    result = machine.run(original, replay_parent=tmp_path)

    assert result.manifest.status is ReductionStatus.INFRASTRUCTURE_INVALID
    assert result.candidate == original
    assert result.manifest.attempts[-1].outcome is PredicateOutcome.INFRASTRUCTURE_INVALID
    assert calls == 9


def test_reduction_contract_validators_reject_incomplete_identity() -> None:
    signature = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="reproduced trials require"):
        PredicateObservation(outcome=PredicateOutcome.REPRODUCED)
    with pytest.raises(ValueError, match="require a retained detail"):
        PredicateObservation(outcome=PredicateOutcome.INFRASTRUCTURE_INVALID)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        ReductionShapeIdentity(input_name="tokens", dimensions=(1, 0))

    predicate = _predicate_contract(signature)
    same_environment = predicate.model_copy(
        update={"environments": (predicate.environments[0], predicate.environments[0])}
    )
    with pytest.raises(ValueError, match="ordered and distinct"):
        ReductionPredicateContract.model_validate(same_environment)
    with pytest.raises(ValueError, match="at least one concrete shape"):
        ReductionPredicateContract.model_validate(
            predicate.model_copy(update={"concrete_shapes": ()})
        )
    with pytest.raises(ValueError, match="input identities"):
        ReductionPredicateContract.model_validate(predicate.model_copy(update={"input_sha256": ()}))


def test_state_machine_rejects_incomplete_or_inconsistent_configuration() -> None:
    signature = "sha256:" + "a" * 64
    predicate = _predicate_contract(signature)

    def hasher(candidate: tuple[str, ...]) -> str:
        return sha256_bytes(canonical_json_bytes(candidate))

    def observed(candidate: tuple[str, ...]) -> PredicateObservation:
        del candidate
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
        )

    limits = ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2)
    reducers = {stage: (lambda candidate: ()) for stage in REDUCTION_STAGES}

    with pytest.raises(InvalidInputError, match="every V1 reducer"):
        ReductionStateMachine(
            expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
            predicate_contract=predicate,
            predicate=observed,
            reducers={},
            clean_replay=lambda candidate, directory: observed(candidate),
            candidate_sha256=hasher,
            limits=limits,
        )
    with pytest.raises(InvalidInputError, match="configuration differs"):
        ReductionStateMachine(
            expected_failure_code=FailureCode.PROFILE_REJECTED,
            predicate_signature_sha256=signature,
            predicate_contract=predicate,
            predicate=observed,
            reducers=reducers,
            clean_replay=lambda candidate, directory: observed(candidate),
            candidate_sha256=hasher,
            limits=limits,
        )
    with pytest.raises(InvalidInputError, match="lowercase SHA-256"):
        ReductionStateMachine(
            expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256="not-a-digest",
            predicate_contract=predicate,
            predicate=observed,
            reducers=reducers,
            clean_replay=lambda candidate, directory: observed(candidate),
            candidate_sha256=hasher,
            limits=limits,
        )


def test_state_machine_distinguishes_budget_and_nonreproduction(tmp_path: Path) -> None:
    signature = "sha256:" + "a" * 64

    def candidate_hash(candidate: tuple[str, ...]) -> str:
        return sha256_bytes(canonical_json_bytes(candidate))

    def reproduced(candidate: tuple[str, ...]) -> PredicateObservation:
        del candidate
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
        )

    reducers = {stage: (lambda candidate: ()) for stage in REDUCTION_STAGES}
    budgeted = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=signature,
        predicate_contract=_predicate_contract(signature),
        predicate=reproduced,
        reducers=reducers,
        clean_replay=lambda candidate, directory: reproduced(candidate),
        candidate_sha256=candidate_hash,
        limits=ReductionLimits(maximum_trials=2, maximum_seconds=30, confirmation_count=2),
    )
    result = budgeted.run(("failure",), replay_parent=tmp_path / "budget")
    assert result.manifest.status is ReductionStatus.BUDGET_EXHAUSTED

    not_reproduced = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=signature,
        predicate_contract=_predicate_contract(signature),
        predicate=lambda candidate: PredicateObservation(outcome=PredicateOutcome.NOT_REPRODUCED),
        reducers=reducers,
        clean_replay=lambda candidate, directory: reproduced(candidate),
        candidate_sha256=candidate_hash,
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
    )
    with pytest.raises(InvalidInputError, match="original reduction candidate"):
        not_reproduced.run(("failure",), replay_parent=tmp_path / "nonreproduced")


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (PredicateOutcome.NOT_REPRODUCED, InvalidInputError),
        (PredicateOutcome.INFRASTRUCTURE_INVALID, None),
    ],
)
def test_state_machine_final_replay_fails_closed(
    tmp_path: Path,
    outcome: PredicateOutcome,
    expected: type[Exception] | None,
) -> None:
    signature = "sha256:" + "a" * 64

    def candidate_hash(candidate: tuple[str, ...]) -> str:
        return sha256_bytes(canonical_json_bytes(candidate))

    def reproduced(candidate: tuple[str, ...]) -> PredicateObservation:
        del candidate
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
        )

    replay_observation = (
        PredicateObservation(outcome=outcome, detail="GPU timeout")
        if outcome is PredicateOutcome.INFRASTRUCTURE_INVALID
        else PredicateObservation(outcome=outcome)
    )
    machine = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=signature,
        predicate_contract=_predicate_contract(signature),
        predicate=reproduced,
        reducers={
            stage: (
                (lambda candidate: ((*candidate, "boundary"),))
                if stage is ReductionStage.ENVIRONMENT_HISTORY
                else (lambda candidate: ())
            )
            for stage in REDUCTION_STAGES
        },
        clean_replay=lambda candidate, directory: replay_observation,
        candidate_sha256=candidate_hash,
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
    )
    if expected is not None:
        with pytest.raises(expected, match="final reduced candidate"):
            machine.run(("failure",), replay_parent=tmp_path)
    else:
        result = machine.run(("failure",), replay_parent=tmp_path)
        assert result.manifest.status is ReductionStatus.INFRASTRUCTURE_INVALID


def test_state_machine_rejects_noop_environment_history_stage(tmp_path: Path) -> None:
    signature = "sha256:" + "a" * 64

    def reproduced(candidate: tuple[str, ...]) -> PredicateObservation:
        del candidate
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=signature,
        )

    machine = ReductionStateMachine(
        expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
        predicate_signature_sha256=signature,
        predicate_contract=_predicate_contract(signature),
        predicate=reproduced,
        reducers={stage: (lambda candidate: ()) for stage in REDUCTION_STAGES},
        clean_replay=lambda candidate, directory: reproduced(candidate),
        candidate_sha256=lambda candidate: sha256_bytes(canonical_json_bytes(candidate)),
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
    )

    with pytest.raises(InvalidInputError, match="environment history reducer"):
        machine.run(("failure",), replay_parent=tmp_path)


def test_confirmed_sequence_reduction_removes_unrelated_outputs_and_options() -> None:
    def predicate(items: tuple[str, ...]) -> TrialOutcome:
        return (
            TrialOutcome.REPRODUCED
            if "failing-output" in items and "required-option" in items
            else TrialOutcome.NOT_REPRODUCED
        )

    reduced = reduce_sequence(
        ("unused-output", "failing-output", "unused-option", "required-option"),
        predicate,
        ReductionLimits(maximum_trials=100, maximum_seconds=10, confirmation_count=2),
    )
    assert set(reduced.items) == {"failing-output", "required-option"}
    assert reduced.trace.reduced_items == 2
    assert reduced.trace.inconclusive_trials == 0


def test_sequence_reducer_fails_closed_on_unstable_original() -> None:
    outcomes = iter((TrialOutcome.REPRODUCED, TrialOutcome.INCONCLUSIVE))
    with pytest.raises(InvalidInputError, match="stable confirmed"):
        reduce_sequence(
            ("output",),
            lambda _: next(outcomes),
            ReductionLimits(maximum_trials=4, maximum_seconds=10, confirmation_count=2),
        )


def test_finite_input_reduction_retains_failure_and_simplifies_values() -> None:
    values = np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    def predicate(candidate: np.ndarray[tuple[int], np.dtype[np.float32]]) -> TrialOutcome:
        return (
            TrialOutcome.REPRODUCED
            if float(np.sum(candidate)) != 4.0
            else TrialOutcome.NOT_REPRODUCED
        )

    reduced = simplify_finite_input(
        values,
        predicate,
        ReductionLimits(maximum_trials=50, maximum_seconds=10, confirmation_count=2),
    )
    assert np.array_equal(reduced.values, np.zeros_like(values))
    assert reduced.changed_elements == 4


def test_environment_reducer_returns_first_adjacent_transition() -> None:
    failing = {"11.1", "11.2"}
    boundary = reduce_environment_history(
        ("10.13", "11.0", "11.1", "11.2"),
        lambda environment: (
            TrialOutcome.REPRODUCED if environment in failing else TrialOutcome.NOT_REPRODUCED
        ),
        ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2),
    )
    assert boundary.last_passing == "11.0"
    assert boundary.first_failing == "11.1"


def test_performance_reducer_never_uses_fewer_than_twenty_pairs() -> None:
    from upgrade_guard.compare.performance import AcceptedPair

    pairs = tuple(AcceptedPair(1.0, 1.2) for _ in range(30))
    reduced = reduce_performance_failure(
        pairs,
        allowance=0.10,
        seed=7,
        replicates=1000,
    )
    assert len(reduced.pairs) == 20
    assert reduced.estimate.one_sided_lower > 1.10
    with pytest.raises(InvalidInputError, match="does not satisfy"):
        reduce_performance_failure(
            (AcceptedPair(1.0, 1.2),),
            allowance=0.10,
            seed=7,
            replicates=1000,
        )


def test_public_performance_reduction_requires_repeated_pairs(tmp_path: Path) -> None:
    source = tmp_path / "performance-failure"
    source.mkdir()
    (source / "baseline.json").write_text(json.dumps([1.0] * 25), encoding="utf-8")
    (source / "candidate.json").write_text(json.dumps([1.2] * 25), encoding="utf-8")
    (source / "reduction-request.json").write_text(
        json.dumps(
            {
                "api_version": "upgradeguard.dev/v1alpha1",
                "kind": "ReductionRequest",
                "failure_code": "PERFORMANCE_REGRESSION",
                "signature_sha256": "sha256:" + "2" * 64,
                "confirmation_count": 2,
                "maximum_trials": 100,
                "maximum_seconds": 60,
                "predicate": {
                    "kind": "performance",
                    "baseline_path": "baseline.json",
                    "candidate_path": "candidate.json",
                    "allowance": 0.10,
                    "bootstrap_seed": 7,
                    "bootstrap_replicates": 1000,
                    "minimum_pairs": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    result = reduce_failure_directory(source, tmp_path / "performance-reduced")
    assert result["failure_code"] == "PERFORMANCE_REGRESSION"
    assert result["reduced_pairs"] == 20
    pairs = tmp_path / "performance-reduced" / "reduced-pairs.json"
    assert result["reduced_pairs_sha256"] == sha256_file(pairs)
    assert result["reduced_pairs_bytes"] == pairs.stat().st_size
    assert len(json.loads(pairs.read_text(encoding="utf-8"))) == 20


def test_reduction_limits_and_evaluator_enforce_both_budgets() -> None:
    with pytest.raises(InvalidInputError, match="cannot satisfy"):
        ReductionLimits(maximum_trials=1, maximum_seconds=1, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="at least two"):
        ReductionLimits(maximum_trials=2, maximum_seconds=1, confirmation_count=1)

    times = iter((0.0, 2.0))
    evaluator = ConfirmedEvaluator(
        lambda _: TrialOutcome.REPRODUCED,
        ReductionLimits(maximum_trials=2, maximum_seconds=1, confirmation_count=2),
        clock=lambda: next(times),
    )
    assert not evaluator.confirms("candidate")
    assert evaluator.exhausted


def test_sequence_reduction_validates_minimum_and_records_exhaustion() -> None:
    limits = ReductionLimits(maximum_trials=2, maximum_seconds=10, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="minimum"):
        reduce_sequence(("a",), lambda _: TrialOutcome.REPRODUCED, limits, minimum_items=2)
    reduced = reduce_sequence(
        ("a", "b"),
        lambda _: TrialOutcome.REPRODUCED,
        limits,
    )
    assert reduced.items == ("a", "b")
    assert reduced.trace.budget_exhausted


def test_input_reducer_validates_values_and_uses_region_simplification() -> None:
    limits = ReductionLimits(maximum_trials=100, maximum_seconds=10, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="nonempty numeric"):
        simplify_finite_input(
            np.asarray([], dtype=np.float32), lambda _: TrialOutcome.REPRODUCED, limits
        )
    with pytest.raises(InvalidInputError, match="finite"):
        simplify_finite_input(
            np.asarray([np.inf], dtype=np.float32),
            lambda _: TrialOutcome.REPRODUCED,
            limits,
        )
    with pytest.raises(InvalidInputError, match="stable confirmed"):
        simplify_finite_input(
            np.asarray([1.0], dtype=np.float32),
            lambda _: TrialOutcome.NOT_REPRODUCED,
            limits,
        )

    original = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    def regional(candidate: np.ndarray[tuple[int], np.dtype[np.float32]]) -> TrialOutcome:
        reproduced = np.array_equal(candidate, original) or (
            np.all(candidate[:2] == 0) and np.array_equal(candidate[2:], original[2:])
        )
        return TrialOutcome.REPRODUCED if reproduced else TrialOutcome.NOT_REPRODUCED

    reduced = simplify_finite_input(original, regional, limits)
    assert np.array_equal(reduced.values, np.asarray([0.0, 0.0, 4.0, 5.0]))
    assert reduced.changed_elements == 2


def test_environment_history_rejects_invalid_noisy_and_missing_boundaries() -> None:
    limits = ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2)
    with pytest.raises(InvalidInputError, match="unique ordered"):
        reduce_environment_history(("same", "same"), lambda _: TrialOutcome.REPRODUCED, limits)
    with pytest.raises(InvalidInputError, match="inconclusive"):
        reduce_environment_history(
            ("first", "second"),
            lambda _: TrialOutcome.INCONCLUSIVE,
            limits,
        )
    with pytest.raises(InvalidInputError, match="no adjacent"):
        reduce_environment_history(
            ("first", "second"),
            lambda _: TrialOutcome.NOT_REPRODUCED,
            limits,
        )
    with pytest.raises(InvalidInputError, match="exhausted"):
        reduce_environment_history(
            ("first", "second"),
            lambda _: TrialOutcome.REPRODUCED,
            ReductionLimits(maximum_trials=2, maximum_seconds=10, confirmation_count=2),
        )


def test_performance_reducer_reports_and_validates_candidate_budget() -> None:
    from upgrade_guard.compare.performance import AcceptedPair

    pairs = tuple(AcceptedPair(1.0, 1.2) for _ in range(25))
    exhausted = reduce_performance_failure(
        pairs,
        allowance=0.1,
        seed=5,
        replicates=1000,
        maximum_candidates=1,
    )
    assert exhausted.budget_exhausted
    assert exhausted.pairs == pairs
    with pytest.raises(InvalidInputError, match="candidate budget"):
        reduce_performance_failure(
            pairs,
            allowance=0.1,
            seed=5,
            replicates=1000,
            maximum_candidates=0,
        )
    with pytest.raises(InvalidInputError, match="time budget"):
        reduce_performance_failure(
            pairs,
            allowance=0.1,
            seed=5,
            replicates=1000,
            maximum_seconds=0,
        )


def test_public_profile_reduction_and_malformed_timings_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "profile-failure"
    source.mkdir()
    request = {
        "api_version": "upgradeguard.dev/v1alpha1",
        "kind": "ReductionRequest",
        "failure_code": "PROFILE_REJECTED",
        "signature_sha256": "sha256:" + "3" * 64,
        "confirmation_count": 2,
        "maximum_trials": 20,
        "maximum_seconds": 60,
        "predicate": {
            "kind": "profile",
            "input_name": "tokens",
            "observed_shape": [9, 128, 256],
            "minimum_shape": [1, 8, 256],
            "maximum_shape": [8, 512, 256],
        },
    }
    (source / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    reduced = reduce_failure_directory(source, tmp_path / "profile-reduced")
    assert reduced["kind"] == "profile"

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "baseline.json").write_text("{}", encoding="utf-8")
    (malformed / "candidate.json").write_text("[]", encoding="utf-8")
    request["failure_code"] = "PERFORMANCE_REGRESSION"
    request["predicate"] = {
        "kind": "performance",
        "baseline_path": "baseline.json",
        "candidate_path": "candidate.json",
        "allowance": 0.1,
        "bootstrap_seed": 7,
        "bootstrap_replicates": 1000,
        "minimum_pairs": 20,
    }
    (malformed / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="timing array"):
        reduce_failure_directory(malformed, tmp_path / "malformed-reduced")
