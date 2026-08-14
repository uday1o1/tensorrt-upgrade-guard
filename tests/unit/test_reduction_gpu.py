"""Host-side failure-boundary tests for candidate-aware GPU reduction."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.runtime import WorkerMounts
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.errors import FailureCode, InfrastructureError, InvalidInputError
from upgrade_guard.reduce import g2_replay
from upgrade_guard.reduce.candidate import G2ReductionCandidate, G7ReductionCandidate
from upgrade_guard.reduce.general import ReductionLimits
from upgrade_guard.reduce.gpu import CandidateGpuPredicate
from upgrade_guard.reduce.polygraphy import run_polygraphy_reduction
from upgrade_guard.reduce.remote import (
    _load_completed_g2,
    _predicate_contract,
    _reduce_g2_environment_history,
    _write_candidate,
    run_remote_reductions,
)
from upgrade_guard.reduce.workflow import (
    PredicateObservation,
    PredicateOutcome,
    ReductionEnvironmentIdentity,
    ReductionSessionManifest,
    ReductionShapeIdentity,
)

SIGNATURE = "sha256:" + "a" * 64
GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"


def _matrix() -> SimpleNamespace:
    return SimpleNamespace(
        gpu_uuid=GPU_UUID,
        environments=(
            SimpleNamespace(
                id="baseline",
                worker_image=SimpleNamespace(
                    canonical_reference="registry/baseline@sha256:" + "1" * 64,
                    manifest_digest="sha256:" + "1" * 64,
                ),
            ),
            SimpleNamespace(
                id="candidate",
                worker_image=SimpleNamespace(
                    canonical_reference="registry/candidate@sha256:" + "2" * 64,
                    manifest_digest="sha256:" + "2" * 64,
                ),
            ),
        ),
    )


def _predicate(tmp_path: Path, worker: object) -> CandidateGpuPredicate:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    return CandidateGpuPredicate(
        project=project,
        state=state,
        matrix=_matrix(),  # type: ignore[arg-type]
        signature_sha256=SIGNATURE,
        evidence_root=tmp_path / "evidence",
        worker=worker,  # type: ignore[arg-type]
        timeout_seconds=17,
    )


def _g2() -> G2ReductionCandidate:
    return G2ReductionCandidate(
        outputs=("G2",),
        rows=1,
        hidden=259,
        x_value=0.0,
        residual_value=1.0,
        gamma_value=1.0,
        environment_id="candidate",
    )


def test_g2_replay_emits_code_only_for_observed_predicate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed = {"G2": {"detected": True, "control": "passed", "rows": 1, "hidden": 259}}
    monkeypatch.setattr(
        g2_replay.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(observed), ""),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "g2-replay",
            "--executable",
            "/output/g2",
            "--rows",
            "1",
            "--hidden",
            "259",
            "--x-value",
            "0",
            "--residual-value",
            "1",
            "--gamma-value",
            "1",
        ],
    )

    g2_replay.main()

    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "failed"
    assert value["failure_code"] == "NUMERICAL_REGRESSION"

    observed["G2"]["detected"] = False
    g2_replay.main()
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "passed"
    assert value["failure_code"] is None


def test_environment_execution_changes_only_locked_worker_identity(tmp_path: Path) -> None:
    class CapturingWorker:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []
            self.images: list[str] = []

        def run(self, **kwargs: object) -> CommandResult:
            command = tuple(cast(Sequence[str], kwargs["command"]))
            self.commands.append(command)
            self.images.append(cast(str, kwargs["image"]))
            payload = {
                "G2": {
                    "detected": True,
                    "control": "passed",
                    "rows": 1,
                    "hidden": 259,
                }
            }
            return CommandResult(command, 0, json.dumps(payload), "", 0.1)

    worker = CapturingWorker()
    predicate = _predicate(tmp_path, worker)
    candidate = _g2().model_copy(update={"environment_history": ("baseline", "candidate")})

    baseline = predicate.evaluate_g2_environment(candidate, "baseline")
    candidate_result = predicate.evaluate_g2_environment(candidate, "candidate")

    assert baseline.outcome is PredicateOutcome.REPRODUCED
    assert candidate_result.outcome is PredicateOutcome.REPRODUCED
    assert worker.images == [
        "registry/baseline@sha256:" + "1" * 64,
        "registry/candidate@sha256:" + "2" * 64,
    ]
    assert worker.commands[0][1:] == worker.commands[1][1:]
    assert "/baseline/" in worker.commands[0][0]
    assert "/candidate/" in worker.commands[1][0]
    assert candidate.environment_id == "candidate"


class G2Worker:
    def __init__(self, *, returncode: int = 0, stdout: str = "", error: Exception | None = None):
        self.returncode = returncode
        self.stdout = stdout
        self.error = error

    def run(self, **kwargs: object) -> CommandResult:
        if self.error is not None:
            raise self.error
        command = tuple(cast(Sequence[str], kwargs["command"]))
        return CommandResult(command, self.returncode, self.stdout, "", 0.1)


@pytest.mark.parametrize(
    ("worker", "expected"),
    (
        (
            G2Worker(error=InfrastructureError("docker failed")),
            PredicateOutcome.INFRASTRUCTURE_INVALID,
        ),
        (G2Worker(returncode=2, stdout="{}"), PredicateOutcome.INFRASTRUCTURE_INVALID),
        (G2Worker(stdout="not-json"), PredicateOutcome.INFRASTRUCTURE_INVALID),
        (
            G2Worker(
                returncode=1,
                stdout=json.dumps(
                    {"G2": {"detected": False, "control": "passed", "rows": 1, "hidden": 259}}
                ),
            ),
            PredicateOutcome.NOT_REPRODUCED,
        ),
    ),
)
def test_g2_worker_failures_are_typed(
    tmp_path: Path,
    worker: G2Worker,
    expected: PredicateOutcome,
) -> None:
    observation = _predicate(tmp_path, worker).evaluate_g2(_g2())
    assert observation.outcome is expected
    if expected is PredicateOutcome.INFRASTRUCTURE_INVALID:
        assert observation.detail
    else:
        assert observation.evidence_sha256


def test_g2_rejects_unknown_environment_and_nonempty_clean_directory(tmp_path: Path) -> None:
    predicate = _predicate(tmp_path, G2Worker())
    unknown = _g2().model_copy(update={"environment_id": "unlocked"})
    with pytest.raises(InfrastructureError, match="not locked"):
        predicate.evaluate_g2(unknown)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="not empty"):
        predicate.evaluate_g2(_g2(), clean)

    empty = tmp_path / "empty"
    empty.mkdir()
    valid = G2Worker(
        stdout=json.dumps({"G2": {"detected": True, "control": "passed", "rows": 1, "hidden": 259}})
    )
    assert _predicate(tmp_path / "valid", valid).evaluate_g2(_g2(), empty).outcome is (
        PredicateOutcome.REPRODUCED
    )


def _g7_candidate(tmp_path: Path, *, bad_shape: bool = False) -> G7ReductionCandidate:
    model = tmp_path / "model.onnx"
    tokens = tmp_path / "tokens.npy"
    mask = tmp_path / "mask.npy"
    model.write_bytes(b"model")
    token_shape = (8, 8, 256) if bad_shape else (9, 8, 256)
    np.save(tokens, np.zeros(token_shape, np.float32), allow_pickle=False)
    np.save(mask, np.zeros((9, 1, 1, 8), np.float32), allow_pickle=False)
    return G7ReductionCandidate(
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
        tokens_path=tokens,
        tokens_sha256=sha256_file(tokens),
        mask_path=mask,
        mask_sha256=sha256_file(mask),
        workspace_bytes=1024,
        optimization_level=3,
        environment_id="candidate",
    )


def test_g7_environment_execution_preserves_candidate_and_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predicate = _predicate(tmp_path, object())
    candidate = _g7_candidate(tmp_path).model_copy(
        update={"environment_history": ("baseline", "candidate")}
    )
    observed: list[G7ReductionCandidate] = []

    def evaluate(value: G7ReductionCandidate) -> PredicateObservation:
        observed.append(value)
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code=FailureCode.PROFILE_REJECTED,
            predicate_signature_sha256=SIGNATURE,
            evidence_sha256=(SIGNATURE,),
        )

    monkeypatch.setattr(predicate, "evaluate_g7", evaluate)

    first = predicate.evaluate_g7_environment(candidate, "baseline")
    second = predicate.evaluate_g7_environment(candidate, "candidate")

    assert first.predicate_signature_sha256 == second.predicate_signature_sha256 == SIGNATURE
    assert [item.environment_id for item in observed] == ["baseline", "candidate"]
    assert observed[0].model_dump(exclude={"environment_id"}) == observed[1].model_dump(
        exclude={"environment_id"}
    )
    assert observed[0].model_path == observed[1].model_path
    assert observed[0].tokens_sha256 == observed[1].tokens_sha256
    assert observed[0].workspace_bytes == observed[1].workspace_bytes


class G7Worker:
    def __init__(
        self,
        *,
        build: tuple[int, str] = (0, "passed"),
        control: tuple[int, str] = (0, "passed"),
        failure: tuple[int, str] = (1, "failed"),
        failure_code: str = "PROFILE_REJECTED",
        failure_message: str = "input shape was rejected for tokens",
        malformed: str | None = None,
        omit: str | None = None,
    ) -> None:
        self.outcomes = {"build": build, "control": control, "failure": failure}
        self.failure_code = failure_code
        self.failure_message = failure_message
        self.malformed = malformed
        self.omit = omit

    def run(self, **kwargs: object) -> CommandResult:
        command = tuple(cast(Sequence[str], kwargs["command"]))
        mounts = cast(WorkerMounts, kwargs["mounts"])
        mounts.output.mkdir(parents=True, exist_ok=True)
        if "upgrade_guard.worker.build_engine" in command:
            kind = "build"
        else:
            result = Path(command[command.index("--result") + 1]).name
            kind = "control" if result == "control.json" else "failure"
        returncode, status = self.outcomes[kind]
        path = mounts.output / f"{kind}.json"
        if self.omit == kind:
            pass
        elif self.malformed == kind:
            path.write_text("not-json", encoding="utf-8")
        else:
            payload: dict[str, object] = {"status": status}
            if kind == "failure":
                payload.update(failure_code=self.failure_code, message=self.failure_message)
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return CommandResult(command, returncode, "", "", 0.1)


@pytest.mark.parametrize(
    ("worker", "outcome", "detail"),
    (
        (G7Worker(build=(1, "failed")), PredicateOutcome.NOT_REPRODUCED, "did not build"),
        (G7Worker(build=(0, "failed")), PredicateOutcome.INFRASTRUCTURE_INVALID, "disagree"),
        (G7Worker(control=(1, "failed")), PredicateOutcome.NOT_REPRODUCED, "control"),
        (G7Worker(control=(0, "failed")), PredicateOutcome.INFRASTRUCTURE_INVALID, "disagree"),
        (G7Worker(failure=(0, "failed")), PredicateOutcome.INFRASTRUCTURE_INVALID, "disagree"),
        (
            G7Worker(failure_code="EXECUTION_FAILED"),
            PredicateOutcome.NOT_REPRODUCED,
            "did not reproduce",
        ),
        (G7Worker(malformed="failure"), PredicateOutcome.INFRASTRUCTURE_INVALID, "malformed"),
        (G7Worker(omit="failure"), PredicateOutcome.INFRASTRUCTURE_INVALID, "did not retain"),
        (
            G7Worker(failure=(1, "unknown")),
            PredicateOutcome.INFRASTRUCTURE_INVALID,
            "valid status",
        ),
    ),
)
def test_g7_worker_result_branches(
    tmp_path: Path,
    worker: G7Worker,
    outcome: PredicateOutcome,
    detail: str,
) -> None:
    observation = _predicate(tmp_path, worker).evaluate_g7(_g7_candidate(tmp_path))
    assert observation.outcome is outcome
    assert detail in cast(str, observation.detail)


def test_g7_rejects_input_shape_drift_before_worker_execution(tmp_path: Path) -> None:
    observation = _predicate(tmp_path, G7Worker()).evaluate_g7(
        _g7_candidate(tmp_path, bad_shape=True)
    )
    assert observation.outcome is PredicateOutcome.INFRASTRUCTURE_INVALID
    assert "contracted shape" in cast(str, observation.detail)


def test_trial_paths_reject_preexisting_evidence(tmp_path: Path) -> None:
    predicate = _predicate(tmp_path, G2Worker())
    occupied = tmp_path / "evidence" / "0001-G2"
    occupied.mkdir(parents=True)
    with pytest.raises(InfrastructureError, match="already exists"):
        predicate.evaluate_g2(_g2())


class TransformWorker:
    def __init__(self, *, model: bool = True, history: int = 2) -> None:
        self.model = model
        self.history = history
        self.timeout: float | None = None

    def run(self, **kwargs: object) -> CommandResult:
        command = tuple(cast(Sequence[str], kwargs["command"]))
        mounts = cast(WorkerMounts, kwargs["mounts"])
        self.timeout = cast(float, kwargs["timeout_seconds"])
        mounts.output.mkdir(parents=True, exist_ok=True)
        operation = command[command.index("--operation") + 1]
        if self.model:
            (mounts.output / f"{operation}.onnx").write_bytes(operation.encode())
        for index in range(self.history):
            (mounts.output / f"history-{index}.json").write_text("{}\n", encoding="utf-8")
        return CommandResult(command, 0, "", "", 0.1)


def test_graph_transform_retains_history_and_enforces_expected_outputs(tmp_path: Path) -> None:
    worker = TransformWorker()
    transformed = _predicate(tmp_path, worker).transform_g7(
        _g7_candidate(tmp_path), "bisect", maximum_seconds=23
    )
    assert transformed.model_path.read_bytes() == b"bisect"
    assert len(transformed.graph_history_sha256) == 2
    assert worker.timeout == 53

    missing_history = _predicate(tmp_path / "history", TransformWorker(history=1))
    with pytest.raises(InfrastructureError, match="check history"):
        missing_history.transform_g7(
            _g7_candidate(tmp_path / "history"), "linear", maximum_seconds=10
        )

    missing_model = _predicate(tmp_path / "model", TransformWorker(model=False))
    with pytest.raises(InfrastructureError, match="did not produce"):
        missing_model.transform_g7(_g7_candidate(tmp_path / "model"), "fold", maximum_seconds=10)


class PolygraphyRunner:
    def __init__(self, *, failure_call: int | None = None, produce: str = "file") -> None:
        self.failure_call = failure_call
        self.produce = produce
        self.calls = 0

    def run(self, args: Sequence[str], **kwargs: object) -> CommandResult:
        del kwargs
        command = tuple(args)
        self.calls += 1
        if self.failure_call == self.calls:
            return CommandResult(command, 19, "", "failed", 0.1)
        output = Path(command[command.index("--output") + 1])
        if self.produce == "file":
            output.write_bytes(f"model-{self.calls}".encode())
        elif self.produce == "empty":
            output.touch()
        return CommandResult(command, 0, "", "", 0.1)


def test_polygraphy_reduction_rejects_invalid_boundaries_and_outputs(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    common = {
        "model": model,
        "output": tmp_path / "reduced.onnx",
        "predicate_command": ("check",),
        "maximum_seconds": 10,
    }
    with pytest.raises(InvalidInputError, match="positive"):
        run_polygraphy_reduction(**{**common, "maximum_seconds": 0})
    with pytest.raises(InvalidInputError, match="regular file"):
        run_polygraphy_reduction(**{**common, "model": tmp_path / "missing.onnx"})
    common["output"].write_bytes(b"occupied")
    with pytest.raises(InvalidInputError, match="overwrite"):
        run_polygraphy_reduction(**common)
    common["output"].unlink()
    with pytest.raises(InvalidInputError, match="failed") as failure:
        run_polygraphy_reduction(**common, runner=PolygraphyRunner(failure_call=2))
    assert failure.value.details["stage"] == "linear"
    with pytest.raises(InvalidInputError, match="expected reduced model"):
        run_polygraphy_reduction(**common, runner=PolygraphyRunner(produce="empty"))


def test_polygraphy_reduction_stops_when_shared_budget_is_exhausted(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    times = iter((0.0, 1.0, 11.0))
    with pytest.raises(InvalidInputError, match="wall-clock budget"):
        run_polygraphy_reduction(
            model=model,
            output=tmp_path / "reduced.onnx",
            predicate_command=("check",),
            maximum_seconds=10,
            runner=PolygraphyRunner(),
            clock=lambda: next(times),
        )


def _contract():
    environments = (
        ReductionEnvironmentIdentity(id="baseline", worker_manifest_sha256="sha256:" + "1" * 64),
        ReductionEnvironmentIdentity(id="candidate", worker_manifest_sha256="sha256:" + "2" * 64),
    )
    return _predicate_contract(
        failure_code=FailureCode.NUMERICAL_REGRESSION,
        signature_sha256=SIGNATURE,
        environments=environments,
        model_sha256="sha256:" + "3" * 64,
        executor_sha256="sha256:" + "4" * 64,
        output_name="output",
        shapes=(ReductionShapeIdentity(input_name="x", dimensions=(1,)),),
        input_sha256=("sha256:" + "5" * 64,),
        threshold_relationship="error exceeds threshold",
        confirmation_count=2,
    )


def test_g2_resume_preserves_partial_or_tampered_state(tmp_path: Path) -> None:
    output = tmp_path / "work"
    output.mkdir()
    candidate = _g2()
    _write_candidate(output / "G2-candidate.json", candidate)
    assert _load_completed_g2(output, _contract()) is None
    stale = tuple((output / "stale").glob("G2-*/G2-candidate.json"))
    assert len(stale) == 1
    assert G2ReductionCandidate.model_validate_json(stale[0].read_text()) == candidate


def test_g2_environment_history_retains_first_observed_boundary() -> None:
    candidate = _g2().model_copy(
        update={"environment_history": ("baseline", "middle", "candidate")}
    )

    class BoundaryPredicate:
        def evaluate_g2_environment(
            self, value: G2ReductionCandidate, environment_id: str
        ) -> PredicateObservation:
            del value
            digest = "sha256:" + (
                {"baseline": "1", "middle": "2", "candidate": "3"}[environment_id] * 64
            )
            if environment_id == "candidate":
                return PredicateObservation(
                    outcome=PredicateOutcome.REPRODUCED,
                    failure_code=FailureCode.NUMERICAL_REGRESSION,
                    predicate_signature_sha256=SIGNATURE,
                    evidence_sha256=(digest,),
                )
            return PredicateObservation(
                outcome=PredicateOutcome.NOT_REPRODUCED,
                evidence_sha256=(digest,),
            )

    reduced = _reduce_g2_environment_history(
        candidate,
        BoundaryPredicate(),  # type: ignore[arg-type]
        ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2),
        SIGNATURE,
    )

    assert reduced.environment_history == ("middle", "candidate")
    assert reduced.environment_boundary is not None
    assert reduced.environment_boundary.last_passing == "middle"
    assert reduced.environment_boundary.first_failing == "candidate"
    assert reduced.candidate_sha256() != candidate.candidate_sha256()


def test_g2_environment_history_rejects_no_boundary() -> None:
    candidate = _g2().model_copy(update={"environment_history": ("baseline", "candidate")})

    class PassingPredicate:
        def evaluate_g2_environment(
            self, value: G2ReductionCandidate, environment_id: str
        ) -> PredicateObservation:
            del value, environment_id
            return PredicateObservation(
                outcome=PredicateOutcome.NOT_REPRODUCED,
                evidence_sha256=(SIGNATURE,),
            )

    with pytest.raises(InvalidInputError, match="no adjacent"):
        _reduce_g2_environment_history(
            candidate,
            PassingPredicate(),  # type: ignore[arg-type]
            ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2),
            SIGNATURE,
        )


def test_source_induced_seed_records_not_applicable_without_false_boundary() -> None:
    candidate = _g2().model_copy(update={"environment_history": ("baseline", "candidate")})
    seen: list[G2ReductionCandidate] = []

    class SourceSeedPredicate:
        def evaluate_g2_environment(
            self, value: G2ReductionCandidate, environment_id: str
        ) -> PredicateObservation:
            observed = value.model_copy(update={"environment_id": environment_id})
            seen.append(observed)
            return PredicateObservation(
                outcome=PredicateOutcome.REPRODUCED,
                failure_code=FailureCode.NUMERICAL_REGRESSION,
                predicate_signature_sha256=SIGNATURE,
                evidence_sha256=("sha256:" + ("1" if environment_id == "baseline" else "2") * 64,),
            )

    reduced = _reduce_g2_environment_history(
        candidate,
        SourceSeedPredicate(),  # type: ignore[arg-type]
        ReductionLimits(maximum_trials=20, maximum_seconds=10, confirmation_count=2),
        SIGNATURE,
    )

    assert reduced.environment_boundary is None
    assert reduced.environment_history_not_applicable is not None
    assert reduced.environment_history_not_applicable.status == "not_applicable"
    assert [
        item.environment_id for item in reduced.environment_history_not_applicable.observations
    ] == ["baseline", "candidate"]
    assert all(
        item.confirmation_count == 2 and len(item.trial_evidence_sha256) == 2
        for item in reduced.environment_history_not_applicable.observations
    )
    assert len(seen) == 4
    assert all(
        item.model_dump(exclude={"environment_id"})
        == seen[0].model_dump(exclude={"environment_id"})
        for item in seen
    )
    assert reduced.candidate_sha256() != candidate.candidate_sha256()


class RemotePredicate:
    def __init__(self, outcome: PredicateOutcome) -> None:
        self.outcome = outcome

    def evaluate_g2(
        self, candidate: G2ReductionCandidate, output: Path | None = None
    ) -> PredicateObservation:
        del candidate, output
        if self.outcome is PredicateOutcome.INFRASTRUCTURE_INVALID:
            return PredicateObservation(outcome=self.outcome, detail="GPU unavailable")
        return PredicateObservation(
            outcome=self.outcome,
            failure_code=FailureCode.NUMERICAL_REGRESSION,
            predicate_signature_sha256=SIGNATURE,
            evidence_sha256=(SIGNATURE,),
        )

    def evaluate_g2_environment(
        self, candidate: G2ReductionCandidate, environment_id: str
    ) -> PredicateObservation:
        if environment_id == candidate.environment_id:
            return self.evaluate_g2(candidate)
        return PredicateObservation(
            outcome=PredicateOutcome.NOT_REPRODUCED,
            evidence_sha256=("sha256:" + "1" * 64,),
        )


def _remote_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    state = tmp_path / "state"
    core = tmp_path / "core"
    plugin = tmp_path / "plugin"
    project.mkdir()
    (state / "plugin-build" / "candidate" / "build").mkdir(parents=True)
    core.mkdir()
    plugin.mkdir()
    (plugin / "residual-rmsnorm-fp32.onnx").write_bytes(b"model")
    (state / "plugin-build" / "candidate" / "build" / "upgrade_guard_gpu_faults").write_bytes(
        b"executable"
    )
    return project, state, core, plugin


def test_remote_reduction_fails_closed_on_g2_infrastructure_and_missing_g7(
    tmp_path: Path,
) -> None:
    project, state, core, plugin = _remote_inputs(tmp_path)
    arguments = {
        "project": project,
        "state": state,
        "core_corpus": core,
        "plugin_corpus": plugin,
        "matrix": _matrix(),
        "signature_sha256": SIGNATURE,
        "output": tmp_path / "work",
        "limits": ReductionLimits(maximum_trials=100, maximum_seconds=10, confirmation_count=2),
    }
    with pytest.raises(InfrastructureError, match="G2 candidate-aware") as failure:
        run_remote_reductions(
            **arguments,
            predicate=RemotePredicate(PredicateOutcome.INFRASTRUCTURE_INVALID),  # type: ignore[arg-type]
        )
    assert failure.value.details["last_error"] == "GPU unavailable"

    with pytest.raises(InfrastructureError, match="remote reduction artifact"):
        run_remote_reductions(
            **arguments,
            predicate=RemotePredicate(PredicateOutcome.REPRODUCED),  # type: ignore[arg-type]
        )
    session_path = cast(Path, arguments["output"]) / "G2-session.json"
    session = ReductionSessionManifest.model_validate_json(session_path.read_text())
    _write_candidate(
        cast(Path, arguments["output"]) / "G2-candidate.json",
        _g2().model_copy(update={"x_value": 1.0}),
    )
    assert _load_completed_g2(cast(Path, arguments["output"]), session.predicate) is None
    assert tuple((cast(Path, arguments["output"]) / "stale").glob("G2-*"))


def test_remote_reduction_rejects_invalid_pair_and_missing_g2_artifact(tmp_path: Path) -> None:
    project, state, core, plugin = _remote_inputs(tmp_path)
    common = {
        "project": project,
        "state": state,
        "core_corpus": core,
        "plugin_corpus": plugin,
        "signature_sha256": SIGNATURE,
        "predicate": RemotePredicate(PredicateOutcome.REPRODUCED),
    }
    pair = _matrix()
    three = SimpleNamespace(
        gpu_uuid=GPU_UUID,
        environments=(*pair.environments, pair.environments[1]),
    )
    with pytest.raises(InfrastructureError, match="exactly two"):
        run_remote_reductions(
            **common,
            matrix=three,  # type: ignore[arg-type]
            output=tmp_path / "three",
        )
    (plugin / "residual-rmsnorm-fp32.onnx").unlink()
    with pytest.raises(InfrastructureError, match="artifact is unavailable"):
        run_remote_reductions(
            **common,
            matrix=pair,  # type: ignore[arg-type]
            output=tmp_path / "missing",
        )
