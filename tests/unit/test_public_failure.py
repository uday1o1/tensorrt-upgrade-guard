"""Host-side contracts for genuine public failure reduction and replay."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scripts.create_remote_reproductions import _performance
from tests.factories import digest, environment_lock, failure_record, run_result
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.common import (
    ArtifactReference,
    DeterminismPolicy,
    FailureRecord,
    NumericalPolicy,
    NumericalTolerance,
    TensorContract,
)
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.contracts.qualification import QualificationSpec, ReductionBudget
from upgrade_guard.errors import FailureCode, InfrastructureError, InvalidInputError
from upgrade_guard.reduce.candidate import LockedEnvironmentBoundary
from upgrade_guard.reduce.domain import (
    DomainInput,
    DomainNumericalCandidate,
    DomainNumericalGpuPredicate,
    DomainPlugin,
    DomainReductionResult,
    _load_build,
    _load_correctness,
    _repetitions_are_stable,
    run_domain_numerical_reduction,
)
from upgrade_guard.reduce.general import ReductionLimits
from upgrade_guard.reduce.public_failure import (
    DomainReductionRequest,
    NumericalMaterial,
    PublicFailureItem,
    _artifact,
    _export_numerical_bundle,
    _json,
    _minimum_vram_mib,
    _numerical_material,
    _preserve_partial,
    _preserve_staging,
    _replay_signature,
    _source_failures,
    _source_files,
    _specification,
    _verify_artifact,
    process_public_failure,
)
from upgrade_guard.reduce.public_replay import build_replay_predicate
from upgrade_guard.reduce.public_replay import main as replay_main
from upgrade_guard.reduce.workflow import PredicateObservation, PredicateOutcome
from upgrade_guard.reproduce.run import ReplayResult

PROJECT = Path(__file__).resolve().parents[2]


class ReproducingPredicate:
    """Record candidates while standing in for the locked GPU worker boundary."""

    def __init__(self, signature: str) -> None:
        self.signature = signature
        self.candidates: list[DomainNumericalCandidate] = []

    def evaluate(
        self,
        candidate: DomainNumericalCandidate,
        output: Path | None = None,
    ) -> PredicateObservation:
        self.candidates.append(candidate)
        if output is not None:
            assert not any(output.iterdir())
            (output / "predicate.json").write_text("{}\n", encoding="utf-8")
        return PredicateObservation(
            outcome=PredicateOutcome.REPRODUCED,
            failure_code="NUMERICAL_REGRESSION",
            predicate_signature_sha256=self.signature,
            evidence_sha256=(digest("e"),),
        )


def _matrix() -> MatrixLock:
    baseline = environment_lock(environment_id="baseline", worker_manifest_character="1")
    candidate = environment_lock(environment_id="candidate", worker_manifest_character="2")
    lock = MatrixLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="EnvironmentLock",
        source_matrix_sha256=digest("3"),
        gpu_uuid=baseline.probe.gpu.uuid,
        created_at=baseline.probed_at,
        environments=(baseline, candidate),
        lock_sha256=digest("0"),
    )
    return lock.model_copy(update={"lock_sha256": lock.computed_sha256()})


def _candidate(
    tmp_path: Path,
    *,
    reference: np.ndarray,
    semantics: str,
    policy: NumericalPolicy,
) -> DomainNumericalCandidate:
    model = tmp_path / "model.onnx"
    profile = tmp_path / "profile.json"
    inputs = tmp_path / "input.npy"
    reference_path = tmp_path / "reference.npy"
    model.write_bytes(b"model")
    profile.write_text("{}\n", encoding="utf-8")
    np.save(inputs, np.zeros((1,), dtype=np.float32), allow_pickle=False)
    np.save(reference_path, reference, allow_pickle=False)
    return DomainNumericalCandidate(
        model_path=model,
        model_sha256=sha256_file(model),
        profile_path=profile,
        profile_sha256=sha256_file(profile),
        inputs=(
            DomainInput(
                name="x",
                path=inputs,
                sha256=sha256_file(inputs),
                shape=(1,),
            ),
        ),
        reference_path=reference_path,
        reference_sha256=sha256_file(reference_path),
        output_name="output",
        semantics=semantics,
        policy=policy,
        determinism=DeterminismPolicy(
            repetitions=20,
            require_bitwise=False,
            tolerance=NumericalTolerance(atol=1e-5, rtol=1e-4),
        ),
        workspace_bytes=1024**3,
        optimization_level=3,
        environment_history=("baseline", "candidate"),
    )


def _policy(
    *,
    baseline_atol: float,
    reference_atol: float,
    drift_atol: float,
    top1: bool = False,
) -> NumericalPolicy:
    return NumericalPolicy(
        baseline_to_reference=NumericalTolerance(atol=baseline_atol, rtol=0),
        candidate_to_reference=NumericalTolerance(atol=reference_atol, rtol=0),
        candidate_to_baseline=NumericalTolerance(atol=drift_atol, rtol=0),
        require_top1_agreement=top1,
    )


def test_domain_predicate_uses_locked_non_bitwise_determinism_tolerance() -> None:
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    within_tolerance = reference + np.asarray([5e-6, -5e-6], dtype=np.float32)
    outside_tolerance = reference + np.asarray([1e-2, 0], dtype=np.float32)
    tolerant = DeterminismPolicy(
        repetitions=20,
        require_bitwise=False,
        tolerance=NumericalTolerance(atol=1e-5, rtol=1e-4),
    )
    bitwise = tolerant.model_copy(update={"require_bitwise": True})

    assert _repetitions_are_stable((reference, within_tolerance), tolerant)
    assert not _repetitions_are_stable((reference, outside_tolerance), tolerant)
    assert not _repetitions_are_stable((reference, within_tolerance), bitwise)


def test_domain_candidate_contract_and_worker_evidence_fail_closed(tmp_path: Path) -> None:
    candidate = _candidate(
        tmp_path,
        reference=np.zeros((2,), dtype=np.float32),
        semantics="tensor",
        policy=_policy(baseline_atol=0, reference_atol=0, drift_atol=0),
    )

    def invalid(**updates: object) -> None:
        value = candidate.model_dump(mode="python")
        value.update(updates)
        DomainNumericalCandidate.model_validate(value)

    with pytest.raises(ValueError, match="input names"):
        invalid(inputs=(candidate.inputs[0], candidate.inputs[0]))
    with pytest.raises(ValueError, match="two index modes"):
        invalid(comparison_flat_index=0, classification_indexes=(0,))
    with pytest.raises(ValueError, match="tensor candidate"):
        invalid(classification_indexes=(0,))
    with pytest.raises(ValueError, match="environment boundary"):
        invalid(
            environment_boundary=LockedEnvironmentBoundary(
                last_passing="candidate",
                first_failing="baseline",
                passing_evidence_sha256=(digest("1"),),
                failing_evidence_sha256=(digest("2"),),
            )
        )
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"plugin")
    with pytest.raises(ValueError, match="one binary"):
        invalid(
            plugins=(
                DomainPlugin(
                    environment_id="baseline",
                    path=plugin,
                    sha256=sha256_file(plugin),
                ),
            )
        )

    original_hash = candidate.candidate_sha256()
    candidate.verify_artifacts()
    candidate.model_path.write_bytes(b"tampered")
    with pytest.raises(InvalidInputError, match="model identity"):
        candidate.verify_artifacts()
    moved = candidate.model_copy(update={"model_path": tmp_path / "another-host-path"})
    assert moved.candidate_sha256() == original_hash

    with pytest.raises(InvalidInputError, match="numerical failure"):
        DomainNumericalGpuPredicate(
            project=PROJECT,
            matrix=_matrix(),
            failure=failure_record(FailureCode.MEMORY_REGRESSION),
            evidence_root=tmp_path,
        )
    predicate = DomainNumericalGpuPredicate(
        project=PROJECT,
        matrix=_matrix(),
        failure=failure_record(FailureCode.NUMERICAL_REGRESSION),
        evidence_root=tmp_path,
    )
    assert predicate._environment("candidate").id == "candidate"
    with pytest.raises(InfrastructureError, match="absent from matrix"):
        predicate._environment("missing")
    assert predicate._not_reproduced((digest("1"),), "changed").detail == "changed"

    malformed = tmp_path / "malformed-worker.json"
    malformed.write_text("{}\n", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="build result is malformed"):
        _load_build(malformed, ("worker", "build"))
    with pytest.raises(InfrastructureError, match="correctness result is malformed"):
        _load_correctness(malformed, ("worker", "run"))


def test_domain_reducer_keeps_batched_classes_and_low_logit_failures(tmp_path: Path) -> None:
    reference = np.asarray(
        [[10, 9, 8, 7, 6, 0, 0, 0], [10, 9, 8, 7, 6, 0, 0, 0]],
        dtype=np.float32,
    )
    baseline = reference.copy()
    candidate_value = reference.copy()
    candidate_value[1, 0:2] = (9, 10)
    candidate_value[0, 7] = 3.0
    baseline_path = tmp_path / "baseline.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(baseline_path, baseline, allow_pickle=False)
    np.save(candidate_path, candidate_value, allow_pickle=False)
    failure = failure_record()
    predicate = ReproducingPredicate(failure.signature_sha256)

    result = run_domain_numerical_reduction(
        project=tmp_path,
        matrix=_matrix(),
        failure=failure,
        original=_candidate(
            tmp_path,
            reference=reference,
            semantics="classification",
            policy=_policy(
                baseline_atol=0,
                reference_atol=2,
                drift_atol=2,
                top1=True,
            ),
        ),
        retained_baseline_output=baseline_path,
        retained_candidate_output=candidate_path,
        output=tmp_path / "reduction",
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
        predicate=predicate,  # type: ignore[arg-type]
    )

    assert result.candidate.classification_indexes == (0, 1, 2, 3, 4, 7)
    assert result.candidate.environment_boundary is not None
    assert result.candidate.environment_boundary.passing_evidence_sha256 == (
        sha256_file(baseline_path),
    )
    assert result.candidate.environment_boundary.failing_evidence_sha256 == (
        sha256_file(candidate_path),
    )
    assert failure.signature_sha256 not in (
        *result.candidate.environment_boundary.passing_evidence_sha256,
        *result.candidate.environment_boundary.failing_evidence_sha256,
    )
    assert all(candidate.semantics == "classification" for candidate in predicate.candidates)


def test_domain_reducer_selects_candidate_to_baseline_only_drift(tmp_path: Path) -> None:
    reference = np.asarray([0.0, 0.0], dtype=np.float32)
    baseline = np.asarray([1.0, 0.0], dtype=np.float32)
    candidate_value = np.asarray([1.2, 0.0], dtype=np.float32)
    baseline_path = tmp_path / "baseline.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(baseline_path, baseline, allow_pickle=False)
    np.save(candidate_path, candidate_value, allow_pickle=False)
    failure = failure_record()

    result = run_domain_numerical_reduction(
        project=tmp_path,
        matrix=_matrix(),
        failure=failure,
        original=_candidate(
            tmp_path,
            reference=reference,
            semantics="tensor",
            policy=_policy(baseline_atol=2, reference_atol=2, drift_atol=0.1),
        ),
        retained_baseline_output=baseline_path,
        retained_candidate_output=candidate_path,
        output=tmp_path / "reduction",
        limits=ReductionLimits(maximum_trials=100, maximum_seconds=30, confirmation_count=2),
        predicate=ReproducingPredicate(failure.signature_sha256),  # type: ignore[arg-type]
    )

    assert result.candidate.comparison_flat_index == 0


def test_public_replay_preserves_per_row_classification_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = np.asarray([[10, 9, 8, 7, 6, 0], [10, 9, 8, 7, 6, 0]], dtype=np.float32)
    baseline = reference.copy()
    candidate = reference.copy()
    candidate[1, 0:2] = (9, 10)
    paths = []
    for name, value in (("reference", reference), ("baseline", baseline), ("candidate", candidate)):
        path = tmp_path / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        paths.append(path)
    policy = tmp_path / "policy.json"
    policy.write_text(
        _policy(
            baseline_atol=2,
            reference_atol=2,
            drift_atol=2,
            top1=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    predicate = tmp_path / "predicate.json"
    predicate.write_text(
        build_replay_predicate(
            failure_signature_sha256=digest("7"),
            output_name="logits",
            semantics="classification",
            indexes=(0, 1, 2, 3, 4, 5),
            reference=paths[0],
            baseline=paths[1],
            policy=policy,
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "public-replay",
            "--reference",
            str(paths[0]),
            "--baseline",
            str(paths[1]),
            "--candidate",
            str(paths[2]),
            "--policy",
            str(policy),
            "--predicate",
            str(predicate),
        ],
    )

    assert replay_main() == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["failure_code"] == "NUMERICAL_REGRESSION"
    assert replay["semantics"] == "classification"

    np.save(paths[1], baseline + 1, allow_pickle=False)
    with pytest.raises(RuntimeError, match="artifacts changed"):
        replay_main()


def test_public_replay_preserves_tensor_drift_and_rejects_lost_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = np.asarray([0.0, 0.0], dtype=np.float32)
    baseline = np.asarray([0.0, 0.0], dtype=np.float32)
    candidate = np.asarray([1.0, 0.0], dtype=np.float32)
    paths = []
    for name, value in (("reference", reference), ("baseline", baseline), ("candidate", candidate)):
        path = tmp_path / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        paths.append(path)
    policy = tmp_path / "policy.json"
    policy.write_text(
        _policy(baseline_atol=0, reference_atol=0.1, drift_atol=0.1).model_dump_json(),
        encoding="utf-8",
    )
    predicate = tmp_path / "predicate.json"
    predicate.write_text(
        build_replay_predicate(
            failure_signature_sha256=digest("7"),
            output_name="output",
            semantics="tensor",
            indexes=(0,),
            reference=paths[0],
            baseline=paths[1],
            policy=policy,
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "public-replay",
            "--reference",
            str(paths[0]),
            "--baseline",
            str(paths[1]),
            "--candidate",
            str(paths[2]),
            "--policy",
            str(policy),
            "--predicate",
            str(predicate),
        ],
    )

    assert replay_main() == 0
    assert json.loads(capsys.readouterr().out)["semantics"] == "tensor"
    np.save(paths[2], reference, allow_pickle=False)
    with pytest.raises(RuntimeError, match="did not preserve"):
        replay_main()


def test_public_bundle_source_inventory_excludes_cache_and_marker_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "package" / "__pycache__").mkdir(parents=True)
    (tmp_path / "cpp").mkdir()
    (tmp_path / "cmake").mkdir()
    (tmp_path / "CMakeLists.txt").write_text("project(test)\n", encoding="utf-8")
    (tmp_path / "src" / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "src" / "package" / "py.typed").write_text("", encoding="utf-8")
    (tmp_path / "src" / "package" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (tmp_path / "cpp" / "kernel.cu").write_text("// source\n", encoding="utf-8")

    files = _source_files(tmp_path)

    assert set(files) == {
        "CMakeLists.txt",
        "src/package/module.py",
        "cpp/kernel.cu",
    }


def test_g5_reduction_retains_smaller_pairs_and_fresh_replay(tmp_path: Path) -> None:
    records = [
        {
            "G5": {
                "baseline_ms": 1.0 + index / 1000,
                "candidate_ms": (1.0 + index / 1000) * 1.10,
            }
        }
        for index in range(24)
    ]
    signature = digest("8")
    result = _performance(
        tmp_path,
        records,
        signature,
        ReductionBudget(maximum_trials=100, maximum_seconds=1800, confirmation_count=2),
    )

    assert result["original_pairs"] == 24
    assert result["reduced_pairs"] == 20
    reduced = tmp_path / "G5-reduced" / "reduced-pairs.json"
    replay = result["clean_replay"]
    assert isinstance(replay, dict)
    assert replay["fresh_directory"] is True
    assert replay["expected_failure_code"] == "PERFORMANCE_REGRESSION"
    assert replay["observed_failure_code"] == "PERFORMANCE_REGRESSION"
    assert replay["reduced_pairs_sha256"] == sha256_file(reduced)
    assert replay["maximum_trials"] == 100
    assert replay["maximum_seconds"] == 1800
    assert replay["confirmation_count"] == 2


def _state_with_core_failure(tmp_path: Path, code: FailureCode) -> tuple[Path, FailureRecord]:
    state = tmp_path / "state"
    core = state / "core-run"
    core.mkdir(parents=True)
    evidence_path = core / "failure-detail.json"
    evidence_path.write_text('{"observed": 2048}\n', encoding="utf-8")
    evidence = ArtifactReference(
        path=evidence_path.relative_to(core).as_posix(),
        sha256=sha256_file(evidence_path),
        bytes=evidence_path.stat().st_size,
        media_type="application/json",
    )
    failure = failure_record(code).model_copy(update={"evidence": (evidence,)})
    (core / "qualification-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": "failed",
                "failure_codes": [code.value],
                "failures": [failure.model_dump(mode="json")],
                "cases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = _matrix()
    (state / "matrix.lock.json").write_text(matrix.model_dump_json(indent=2) + "\n")
    shutil.copyfile(PROJECT / "qualification" / "full.yaml", state / "full.yaml")
    return state, failure


def test_public_failure_routes_unsupported_core_failure_and_resumes(tmp_path: Path) -> None:
    state, failure = _state_with_core_failure(tmp_path, FailureCode.MEMORY_REGRESSION)
    output = state / "public-failure"

    disposition = process_public_failure(
        state=state,
        project=PROJECT,
        source_step="core-qualification",
        core_corpus=tmp_path,
        plugin_corpus=tmp_path,
        mobilenet_corpus=tmp_path,
        output=output,
    )

    assert disposition.items[0].failure == failure
    assert disposition.items[0].disposition == "not_applicable"
    assert "memory confirmation builds" in (disposition.items[0].reason or "")
    assert (
        process_public_failure(
            state=state,
            project=PROJECT,
            source_step="core-qualification",
            core_corpus=tmp_path,
            plugin_corpus=tmp_path,
            mobilenet_corpus=tmp_path,
            output=output,
        )
        == disposition
    )

    source = state / "core-run" / "qualification-summary.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    value["failures"][0]["observed"] = "tampered"
    source.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="artifact identity differs"):
        process_public_failure(
            state=state,
            project=PROJECT,
            source_step="core-qualification",
            core_corpus=tmp_path,
            plugin_corpus=tmp_path,
            mobilenet_corpus=tmp_path,
            output=output,
        )


def test_public_failure_refuses_corpus_invalid_and_invalid_contract_claims(
    tmp_path: Path,
) -> None:
    state, failure = _state_with_core_failure(tmp_path, FailureCode.CORPUS_INVALID)
    with pytest.raises(InvalidInputError, match="cannot become a failed publication"):
        process_public_failure(
            state=state,
            project=PROJECT,
            source_step="core-qualification",
            core_corpus=tmp_path,
            plugin_corpus=tmp_path,
            mobilenet_corpus=tmp_path,
            output=state / "public-failure",
        )

    numerical = failure_record(FailureCode.NUMERICAL_REGRESSION)
    with pytest.raises(ValueError, match="supported failures"):
        PublicFailureItem(
            failure=numerical,
            disposition="not_applicable",
            reason="incorrect claim",
        )
    with pytest.raises(ValueError, match="reduction claims"):
        PublicFailureItem(
            failure=failure_record(FailureCode.MEMORY_REGRESSION),
            disposition="not_applicable",
            reason="unsupported",
            request_sha256=digest("1"),
        )
    request = DomainReductionRequest.model_construct(
        source_step="core-qualification",
        source_artifact_sha256=digest("1"),
        matrix_lock_sha256=digest("2"),
        failure=numerical,
        candidate_sha256=digest("3"),
        reduction_budget=ReductionBudget(
            maximum_trials=100,
            maximum_seconds=1800,
            confirmation_count=2,
        ),
        request_sha256=digest("0"),
    )
    with pytest.raises(ValueError, match="self-hash differs"):
        DomainReductionRequest.model_validate(request)


def test_public_failure_source_parsing_and_artifact_helpers_fail_closed(tmp_path: Path) -> None:
    state, failure = _state_with_core_failure(tmp_path, FailureCode.MEMORY_REGRESSION)
    summary = state / "core-run" / "qualification-summary.json"
    assert _source_failures(summary, "core-qualification") == (failure,)

    extended = tmp_path / "extended.json"
    extended.write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_code": failure.code.value,
                "failure": failure.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    assert _source_failures(extended, "plugin-matrix") == (failure,)
    extended.write_text("[]\n", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="not an object"):
        _source_failures(extended, "plugin-matrix")

    retained = tmp_path / "retained.json"
    retained.write_text("{}\n", encoding="utf-8")
    artifact = _artifact(tmp_path, retained)
    _verify_artifact(tmp_path, artifact)
    retained.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(InfrastructureError, match="identity differs"):
        _verify_artifact(tmp_path, artifact)
    assert _minimum_vram_mib(PROJECT) > 0
    with pytest.raises(InfrastructureError, match="minimum_vram_mib"):
        _minimum_vram_mib(tmp_path)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="JSON is invalid"):
        _json(malformed)
    replay = tmp_path / "replay-step.json"
    replay.write_text(
        json.dumps({"stdout": json.dumps({"signature_sha256": digest("5")})}),
        encoding="utf-8",
    )
    assert _replay_signature(replay) == digest("5")
    partial = tmp_path / "partial"
    partial.mkdir()
    _preserve_partial(partial)
    assert not partial.exists()
    assert len(tuple(tmp_path.glob("partial.partial-*"))) == 1

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / ".bundle.interrupted").mkdir()
    (staging / ".clean-replay.interrupted").mkdir()
    (staging / ".unrelated").mkdir()
    _preserve_staging(staging)
    assert not tuple(staging.glob(".bundle.*"))
    assert not tuple(staging.glob(".clean-replay.*"))
    assert len(tuple(staging.glob("stale-bundle.interrupted-*"))) == 1
    assert len(tuple(staging.glob("stale-clean-replay.interrupted-*"))) == 1
    assert (staging / ".unrelated").is_dir()
    (staging / ".bundle.link").symlink_to(staging / ".unrelated", target_is_directory=True)
    with pytest.raises(InfrastructureError, match="staging path is unsafe"):
        _preserve_staging(staging)


def test_public_numerical_bundle_retains_locked_recipe_and_failure_evidence(
    tmp_path: Path,
) -> None:
    state, raw_failure = _state_with_core_failure(tmp_path, FailureCode.NUMERICAL_REGRESSION)
    assert raw_failure.precision is not None
    failure = raw_failure
    specification = QualificationSpec.model_validate(
        _specification(state / "full.yaml").model_dump(mode="python")
    )
    candidate = _candidate(
        tmp_path,
        reference=np.asarray([0.0, 0.0], dtype=np.float32),
        semantics="tensor",
        policy=specification.numerical_policy(failure.precision),
    ).model_copy(update={"comparison_flat_index": 0})
    baseline = tmp_path / "baseline.npy"
    np.save(baseline, np.asarray([0.0, 0.0], dtype=np.float32), allow_pickle=False)

    manifest = _export_numerical_bundle(
        project=PROJECT,
        state=state,
        source_step="core-qualification",
        matrix=_matrix(),
        specification=specification,
        failure=failure,
        candidate=candidate,
        baseline_output=baseline,
        destination=tmp_path / "bundle",
    )

    recipe = json.loads((tmp_path / "bundle" / "commands" / "replay.json").read_text())
    configure = recipe["steps"][0]["command"]
    correctness = recipe["steps"][-2]["command"]
    predicate = recipe["steps"][-1]["command"]
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in configure
    assert str(candidate.determinism.repetitions) in correctness
    assert "--predicate" in predicate
    assert manifest.expected_failure == failure
    assert (tmp_path / "bundle" / "logs" / "failure-record.json").is_file()
    assert (tmp_path / "bundle" / "failure-detail.json").is_file()


def test_public_classification_bundle_retains_plugin_build_and_indexes(
    tmp_path: Path,
) -> None:
    state, failure = _state_with_core_failure(tmp_path, FailureCode.NUMERICAL_REGRESSION)
    specification = _specification(state / "full.yaml")
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"plugin")
    candidate = _candidate(
        tmp_path,
        reference=np.zeros((1, 6), dtype=np.float32),
        semantics="classification",
        policy=specification.numerical_policy(failure.precision),
    ).model_copy(
        update={
            "classification_indexes": (0, 1, 2, 3, 4),
            "plugins": tuple(
                DomainPlugin(
                    environment_id=environment_id,
                    path=plugin,
                    sha256=sha256_file(plugin),
                )
                for environment_id in ("baseline", "candidate")
            ),
        }
    )
    baseline = tmp_path / "baseline-classification.npy"
    np.save(baseline, np.zeros((1, 6), dtype=np.float32), allow_pickle=False)

    _export_numerical_bundle(
        project=PROJECT,
        state=state,
        source_step="core-qualification",
        matrix=_matrix(),
        specification=specification,
        failure=failure,
        candidate=candidate,
        baseline_output=baseline,
        destination=tmp_path / "classification-bundle",
    )

    recipe = json.loads(
        (tmp_path / "classification-bundle" / "commands" / "replay.json").read_text()
    )
    assert recipe["steps"][1]["id"] == "compile-plugin"
    assert "--plugin" in recipe["steps"][2]["command"]
    predicate = json.loads(
        (tmp_path / "classification-bundle" / "reduction" / "replay-predicate.json").read_text()
    )
    assert predicate["semantics"] == "classification"
    assert predicate["indexes"] == [0, 1, 2, 3, 4]


def test_supported_public_failure_runs_fake_gpu_boundary_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import upgrade_guard.reduce.public_failure as public_failure

    state, failure = _state_with_core_failure(tmp_path, FailureCode.NUMERICAL_REGRESSION)
    specification = _specification(state / "full.yaml")
    reference = np.asarray([0.0, 0.0], dtype=np.float32)
    candidate_value = np.asarray([1.0, 0.0], dtype=np.float32)
    candidate = _candidate(
        tmp_path,
        reference=reference,
        semantics="tensor",
        policy=specification.numerical_policy(failure.precision),
    )
    baseline_path = tmp_path / "baseline.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(baseline_path, reference, allow_pickle=False)
    np.save(candidate_path, candidate_value, allow_pickle=False)
    material = NumericalMaterial(candidate, baseline_path, candidate_path)
    predicate = ReproducingPredicate(failure.signature_sha256)

    monkeypatch.setattr(public_failure, "_numerical_material", lambda **_kwargs: material)

    def reduce_with_fake_boundary(**kwargs: object) -> DomainReductionResult:
        return run_domain_numerical_reduction(
            project=kwargs["project"],  # type: ignore[arg-type]
            matrix=kwargs["matrix"],  # type: ignore[arg-type]
            failure=kwargs["failure"],  # type: ignore[arg-type]
            original=kwargs["original"],  # type: ignore[arg-type]
            retained_baseline_output=kwargs["retained_baseline_output"],  # type: ignore[arg-type]
            retained_candidate_output=kwargs["retained_candidate_output"],  # type: ignore[arg-type]
            output=kwargs["output"],  # type: ignore[arg-type]
            limits=kwargs["limits"],  # type: ignore[arg-type]
            predicate=predicate,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        public_failure,
        "run_domain_numerical_reduction",
        reduce_with_fake_boundary,
    )
    monkeypatch.setattr(public_failure, "observe_replay_target", lambda _gpu: object())
    replay_calls = 0

    def fake_execute_replay(
        bundle_root: Path,
        replay_root: Path,
        **_kwargs: object,
    ) -> ReplayResult:
        nonlocal replay_calls
        replay_calls += 1
        manifest = public_failure.verify_bundle(bundle_root).manifest
        log = replay_root / "logs" / "worker-build.log"
        step = replay_root / "steps" / "three-way-failure.json"
        log.parent.mkdir(parents=True)
        step.parent.mkdir(parents=True)
        log.write_text("bounded fake worker rebuild\n", encoding="utf-8")
        step.write_text(
            json.dumps({"stdout": json.dumps({"signature_sha256": failure.signature_sha256})})
            + "\n",
            encoding="utf-8",
        )
        log_reference = ArtifactReference(
            path="logs/worker-build.log",
            sha256=sha256_file(log),
            bytes=log.stat().st_size,
            media_type="text/plain",
        )
        replay = ReplayResult(
            schema_version="upgradeguard.dev/replay-result/v1",
            status="passed",
            bundle_id=manifest.id,
            bundle_manifest_sha256=manifest.manifest_sha256,
            worker_image="127.0.0.1:5500/worker@" + digest("a"),
            worker_rebuild_recipe_sha256=digest("b"),
            worker_build_log_sha256=log_reference.sha256,
            worker_build_log=log_reference,
            original_gpu_uuid=_matrix().gpu_uuid,
            selected_gpu_uuid=_matrix().gpu_uuid,
            expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
            observed_failure_code=FailureCode.NUMERICAL_REGRESSION,
            step_results=("configure", "build-engine", "three-way-failure"),
        )
        payload = asdict(replay)
        payload["worker_build_log"] = log_reference.model_dump(mode="json")
        (replay_root / "replay-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return replay

    monkeypatch.setattr(public_failure, "execute_replay", fake_execute_replay)
    output = state / "public-failure"
    disposition = process_public_failure(
        state=state,
        project=PROJECT,
        source_step="core-qualification",
        core_corpus=tmp_path,
        plugin_corpus=tmp_path,
        mobilenet_corpus=tmp_path,
        output=output,
    )

    assert disposition.items[0].disposition == "reduced_replayed"
    assert disposition.items[0].observed_signature_sha256 == failure.signature_sha256
    assert replay_calls == 1

    (output / "disposition.json").unlink()
    resumed = process_public_failure(
        state=state,
        project=PROJECT,
        source_step="core-qualification",
        core_corpus=tmp_path,
        plugin_corpus=tmp_path,
        mobilenet_corpus=tmp_path,
        output=output,
    )
    assert resumed == disposition
    assert replay_calls == 1


def test_core_numerical_material_resolves_exact_locked_artifacts(tmp_path: Path) -> None:
    state, failure = _state_with_core_failure(tmp_path, FailureCode.NUMERICAL_REGRESSION)
    matrix = _matrix()
    corpus = tmp_path / "core-corpus"
    model = corpus / "models" / "tiny-transformer-fp32.onnx"
    input_root = corpus / "inputs" / "tiny-transformer-fp32" / str(failure.shape_id)
    reference = (
        corpus / "reference" / f"tiny-transformer-fp32-{failure.shape_id}-{failure.output_name}.npy"
    )
    profile = state / "core-run" / "candidate" / "fp32" / "profile.json"
    model.parent.mkdir(parents=True)
    input_root.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    profile.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    profile.write_text("{}\n", encoding="utf-8")
    for name in ("tokens", "mask"):
        np.save(input_root / f"{name}.npy", np.ones((1, 2), dtype=np.int32))
    np.save(reference, np.zeros((1, 2, 4), dtype=np.float32))

    typed_runs = {}
    for environment_id, fill in (("baseline", 0.0), ("candidate", 1.0)):
        output = state / "core-run" / environment_id / "output.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.full((1, 2, 4), fill, dtype=np.float32))
        artifact = ArtifactReference(
            path=output.relative_to(state / "core-run").as_posix(),
            sha256=sha256_file(output),
            bytes=output.stat().st_size,
            media_type="application/x-npy",
        )
        base = run_result()
        hardware = base.hardware.model_copy(update={"environment_lock_sha256": matrix.lock_sha256})
        typed = base.model_copy(
            update={
                "id": f"{environment_id}-run",
                "environment_lock_sha256": matrix.lock_sha256,
                "hardware": hardware,
                "output_schema": (
                    TensorContract(
                        name=str(failure.output_name),
                        dtype="float32",
                        shape=(1, 2, 4),
                    ),
                ),
                "output_artifacts": (artifact,),
            }
        )
        typed_runs[environment_id] = typed.model_dump(mode="json")
    summary_path = state / "core-run" / "qualification-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cases"] = [
        {
            "failure_code": failure.code.value,
            "precision": "fp32",
            "shape_id": failure.shape_id,
            "typed_run_results": typed_runs,
        }
    ]
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    material = _numerical_material(
        state=state,
        source_step="core-qualification",
        failure=failure,
        specification=_specification(state / "full.yaml"),
        matrix=matrix,
        core_corpus=corpus,
        plugin_corpus=tmp_path,
        mobilenet_corpus=tmp_path,
    )

    assert material.candidate.model_path == model
    assert material.candidate.determinism.repetitions == 20
    assert tuple(item.name for item in material.candidate.inputs) == ("tokens", "mask")
    assert material.retained_baseline_output.name == "output.npy"
    assert material.retained_candidate_output.name == "output.npy"


def test_plugin_numerical_material_binds_both_plugin_binaries(tmp_path: Path) -> None:
    state, failure = _state_with_core_failure(tmp_path, FailureCode.NUMERICAL_REGRESSION)
    matrix = _matrix()
    corpus = tmp_path / "plugin-corpus"
    case_id = "tail"
    model = corpus / "residual-rmsnorm-fp32.onnx"
    input_root = corpus / "fp32" / case_id
    profile = state / "plugin-runs" / "candidate" / "fp32" / "profile.json"
    model.parent.mkdir(parents=True)
    input_root.mkdir(parents=True)
    profile.parent.mkdir(parents=True)
    model.write_bytes(b"plugin-model")
    profile.write_text("{}\n", encoding="utf-8")
    for name in ("x", "residual"):
        np.save(input_root / f"{name}.npy", np.ones((1, 2, 4), dtype=np.float32))
    np.save(input_root / "gamma.npy", np.ones((4,), dtype=np.float32))
    np.save(input_root / "expected.npy", np.zeros((1, 2, 4), dtype=np.float32))

    stable: dict[str, dict[str, dict[str, str]]] = {}
    for environment_id, fill in (("baseline", 0.0), ("candidate", 1.0)):
        run_root = state / "plugin-runs" / environment_id
        run_root.mkdir(parents=True, exist_ok=True)
        output = run_root / "output.npy"
        np.save(output, np.full((1, 2, 4), fill, dtype=np.float32))
        artifact = ArtifactReference(
            path=output.relative_to(state / "plugin-runs").as_posix(),
            sha256=sha256_file(output),
            bytes=output.stat().st_size,
            media_type="application/x-npy",
        )
        base = run_result()
        hardware = base.hardware.model_copy(update={"environment_lock_sha256": matrix.lock_sha256})
        typed = base.model_copy(
            update={
                "id": f"{environment_id}-plugin-run",
                "environment_lock_sha256": matrix.lock_sha256,
                "hardware": hardware,
                "output_schema": (
                    TensorContract(
                        name=str(failure.output_name),
                        dtype="float32",
                        shape=(1, 2, 4),
                    ),
                ),
                "output_artifacts": (artifact,),
            }
        )
        run_path = run_root / "run-result.json"
        run_path.write_text(typed.model_dump_json(indent=2) + "\n", encoding="utf-8")
        stable[environment_id] = {
            "run_result": {"path": run_path.relative_to(state / "plugin-runs").as_posix()}
        }
        plugin = (
            state
            / "plugin-build"
            / environment_id
            / "build"
            / "libupgrade_guard_residual_rmsnorm.so"
        )
        plugin.parent.mkdir(parents=True)
        plugin.write_bytes(f"{environment_id}-plugin".encode())
    validation = state / "plugin-runs" / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "status": "failed",
                "cases": [
                    {
                        "case": case_id,
                        "failure_code": failure.code.value,
                        "stable_artifacts": stable,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    material = _numerical_material(
        state=state,
        source_step="plugin-matrix",
        failure=failure,
        specification=_specification(state / "full.yaml"),
        matrix=matrix,
        core_corpus=tmp_path,
        plugin_corpus=corpus,
        mobilenet_corpus=tmp_path,
    )

    assert material.candidate.model_path == model
    assert tuple(item.name for item in material.candidate.inputs) == (
        "x",
        "residual",
        "gamma",
    )
    assert {item.environment_id for item in material.candidate.plugins} == {
        "baseline",
        "candidate",
    }
