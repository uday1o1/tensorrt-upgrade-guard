"""Tests for hash-addressed remote qualification resume state."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from scripts.three_way_validation import ThreeWayValidationResult
from tests.factories import FIXED_TIME, digest, environment_lock, failure_record
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode
from upgrade_guard.reduce.public_failure import PublicFailureDisposition, PublicFailureItem

SCRIPT = Path(__file__).parents[2] / "scripts" / "qualification_state.py"
SPEC = importlib.util.spec_from_file_location("qualification_state", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification_state)

SOURCE = "a" * 40
GPU = "GPU-11111111-1111-1111-1111-111111111111"
FIXED_RECONCILE_TIME = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    state = tmp_path / "state"
    project = tmp_path / "project"
    state.mkdir()
    project.mkdir()
    qualification = project / "qualification" / "full.yaml"
    qualification.parent.mkdir()
    qualification.write_text("kind: Qualification\n", encoding="utf-8")
    return state, project


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_matrix(state: Path) -> MatrixLock:
    baseline = environment_lock(environment_id="baseline", worker_manifest_character="1")
    candidate = environment_lock(environment_id="candidate", worker_manifest_character="2")
    lock = MatrixLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="EnvironmentLock",
        source_matrix_sha256=digest("9"),
        gpu_uuid=GPU,
        created_at=FIXED_TIME,
        environments=(baseline, candidate),
        lock_sha256=digest("0"),
    )
    lock = lock.model_copy(update={"lock_sha256": lock.computed_sha256()})
    (state / "matrix.lock.json").write_text(lock.model_dump_json(indent=2) + "\n")
    return lock


def _write_corpora(state: Path, *, suffix: str = "1") -> dict[str, dict[str, str]]:
    project = state.parent / "project"
    lock_names = {
        "core": "corpus.lock.json",
        "plugin": "plugin-corpus.lock.json",
        "mobilenet": "mobilenet-corpus.lock.json",
    }
    identities: dict[str, dict[str, str]] = {}
    reference_environment_sha256 = digest("d")
    for kind, character in (("core", "a"), ("plugin", "b"), ("mobilenet", "c")):
        root_relative = f".upgrade-guard/corpora/by-id/{kind}/{character * 64}"
        root = project / root_relative
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / "artifact.bin"
        artifact.write_bytes(f"{kind}-{suffix if kind == 'core' else character}".encode())
        lock = root / lock_names[kind]
        _write_json(
            lock,
            {
                "reference_environment_sha256": reference_environment_sha256,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": qualification_state._sha256(artifact),
                    }
                ],
            },
        )
        materializer_sha256 = digest(character)
        _write_json(
            root / "materializer.json",
            {"materializer_sha256": materializer_sha256},
        )
        identities[kind] = {
            "root": root_relative,
            "lock": lock.relative_to(project).as_posix(),
            "lock_sha256": qualification_state._sha256(lock),
            "materializer_sha256": materializer_sha256,
            "inventory_sha256": qualification_state._corpus_inventory_sha256(root),
            "reference_environment_sha256": reference_environment_sha256,
        }
    _write_json(
        state / "corpora.json",
        {
            "schema_version": "upgradeguard.dev/corpus-index/v1",
            "reference_environment_sha256": reference_environment_sha256,
            "corpora": identities,
        },
    )
    return identities


def _prepare_step(state: Path, step: str, *, invoked_step: str | None = None) -> None:
    canonical = qualification_state._canonical_step(step)
    for authored in qualification_state.STEP_OWNED_PATHS[canonical]:
        relative = authored.removesuffix("/")
        path = state / relative
        if authored.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "artifact.txt").write_text(f"{canonical}\n", encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                _write_json(path, {"status": "passed", "step": canonical})
            else:
                path.write_text(f"{canonical}\n", encoding="utf-8")
    log_name = invoked_step or canonical
    log = state / "logs" / f"{log_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"closed {log_name}\n", encoding="utf-8")

    if canonical == "matrix-lock":
        _write_matrix(state)
    if canonical == "corpus-materialization":
        _write_corpora(state)
    for relative in qualification_state.PASSING_JSON.get(canonical, ()):
        value: dict[str, object] = {"status": "passed"}
        if canonical == "aa-pilot":
            value.update({"false_positive": False, "accepted_pairs": 20})
        _write_json(state / relative, value)


def _closure(mode: str, targets: set[str]) -> set[str]:
    result: set[str] = set()

    def include(step: str) -> None:
        canonical = qualification_state._canonical_step(step)
        if canonical in result:
            return
        for dependency in qualification_state._dependencies_for(canonical, mode):
            include(dependency)
        result.add(canonical)

    for target in targets:
        include(target)
    return result


def _record_closure(
    state: Path,
    project: Path,
    targets: set[str],
    *,
    mode: str = "full",
    qualification_spec: Path | None = None,
) -> None:
    required = _closure(mode, targets)
    for step in qualification_state._topological_steps(mode):
        if step not in required:
            continue
        _prepare_step(state, step)
        qualification_state.record_marker(
            state,
            project,
            step,
            SOURCE,
            GPU,
            mode,
            qualification_spec=qualification_spec,
        )


def test_marker_v2_binds_context_inventory_and_closed_log(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _prepare_step(state, "preflight")
    payload = qualification_state.record_marker(state, project, "preflight", SOURCE, GPU, "full")
    assert payload["schema_version"] == qualification_state.MARKER_SCHEMA
    assert payload["step"] == "preflight"
    assert payload["source_git_commit"] == SOURCE
    assert payload["gpu_uuid"] == GPU
    assert payload["mode"] == "full"
    assert [item["path"] for item in payload["inventory"]] == [
        "gpu-preflight.csv",
        "gpu.uuid",
        "logs/preflight.log",
        "source.commit",
    ]
    assert payload["direct_dependency_marker_sha256s"] == {}
    assert payload["matrix_lock_sha256"] is None
    assert payload["corpus_identities"] == []
    assert payload["qualification_spec_lineage"] is None
    assert qualification_state.verify_marker(state, project, "preflight", SOURCE, GPU, "full")


def test_completed_core_failure_marker_survives_reconcile_and_closes_publication(
    tmp_path: Path,
) -> None:
    state, project = _roots(tmp_path)
    full_steps = qualification_state.MODE_STEPS["full"]
    prerequisites = set(full_steps[: full_steps.index("core-qualification")])
    _record_closure(state, project, prerequisites)
    _prepare_step(state, "core-qualification")
    matrix = MatrixLock.model_validate_json((state / "matrix.lock.json").read_text())
    failure = failure_record(FailureCode.MEMORY_REGRESSION)
    failure_path = state / "core-run" / "qualification-summary.json"
    _write_json(
        failure_path,
        {
            "schema_version": "upgradeguard.dev/qualification-summary/v1",
            "status": "failed",
            "failure_codes": ["MEMORY_REGRESSION"],
            "failures": [failure.model_dump(mode="json")],
            "started_at": "2026-08-13T20:00:00+00:00",
            "ended_at": "2026-08-13T20:00:01+00:00",
            "environment_lock_sha256": matrix.lock_sha256,
            "gpu_uuid": matrix.gpu_uuid,
            "stack_attribution": "locked stacks",
            "cases": [],
        },
    )
    marker = qualification_state.record_marker(
        state,
        project,
        "core-qualification",
        SOURCE,
        GPU,
        "full",
    )
    assert marker["outcome"] == "failed"

    _prepare_step(state, "public-failure")
    source_artifact = ArtifactReference(
        path=failure_path.relative_to(state).as_posix(),
        sha256=qualification_state._sha256(failure_path),
        bytes=failure_path.stat().st_size,
        media_type="application/json",
    )
    unvalidated = PublicFailureDisposition.model_construct(
        source_step="core-qualification",
        source_artifact=source_artifact,
        items=(
            PublicFailureItem(
                failure=failure,
                disposition="not_applicable",
                reason="no authored V1 memory confirmation-build reducer",
            ),
        ),
        disposition_sha256=digest("0"),
    )
    disposition = PublicFailureDisposition.model_validate(
        unvalidated.model_copy(update={"disposition_sha256": unvalidated.computed_sha256()})
    )
    _write_json(
        state / "public-failure" / "disposition.json",
        disposition.model_dump(mode="json"),
    )
    public_marker = qualification_state.record_marker(
        state,
        project,
        "public-failure",
        SOURCE,
        GPU,
        "full",
    )
    assert public_marker["outcome"] == "passed"

    _prepare_step(state, "final-evidence")
    for name in ("results.json", "evidence.json"):
        _write_json(state / name, {"status": "failed"})
    _write_json(state / "report-model.json", {"status": "passed"})
    (state / "report.md").write_text("# Qualification passed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed publication report"):
        qualification_state.record_marker(
            state,
            project,
            "final-evidence",
            SOURCE,
            GPU,
            "full",
        )
    _write_json(state / "report-model.json", {"status": "failed"})
    (state / "report.md").write_text("# Qualification failure\n", encoding="utf-8")
    final_marker = qualification_state.record_marker(
        state,
        project,
        "final-evidence",
        SOURCE,
        GPU,
        "full",
    )
    assert final_marker["outcome"] == "failed"
    assert "core-qualification" in final_marker["direct_dependency_marker_sha256s"]
    assert "plugin-matrix" not in final_marker["direct_dependency_marker_sha256s"]

    reconciled = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        dry_run=True,
        now=FIXED_RECONCILE_TIME,
    )
    assert "core-qualification" in reconciled["valid_steps"]
    assert "final-evidence" in reconciled["valid_steps"]
    assert all(move["step"] != "core-qualification" for move in reconciled["moves"])


@pytest.mark.parametrize(
    ("step", "relative", "schema"),
    (
        (
            "plugin-matrix",
            "plugin-runs/validation.json",
            "upgradeguard.dev/plugin-validation/v2",
        ),
        (
            "mobilenet-matrix",
            "mobilenet-runs/validation.json",
            "upgradeguard.dev/mobilenet-validation/v2",
        ),
    ),
)
def test_completed_extended_failure_markers_survive_reconcile(
    tmp_path: Path,
    step: str,
    relative: str,
    schema: str,
) -> None:
    state, project = _roots(tmp_path)
    prerequisites = set(qualification_state._dependencies_for(step, "full"))
    _record_closure(state, project, prerequisites)
    _prepare_step(state, step)
    failure = failure_record()
    root = state / relative.split("/", maxsplit=1)[0]
    stable = {}
    for name in ("case_manifest", "build_manifest", "run_result"):
        artifact = root / f"{name}.json"
        artifact.write_text("{}\n", encoding="utf-8")
        stable[name] = ArtifactReference(
            path=artifact.relative_to(root).as_posix(),
            sha256=qualification_state._sha256(artifact),
            bytes=artifact.stat().st_size,
            media_type="application/json",
        ).model_dump(mode="json")
    result = ThreeWayValidationResult(
        schema_version=schema,
        status="failed",
        failure_code=failure.code,
        failure=failure,
        specification_sha256=digest("1"),
        invocation_manifest=ArtifactReference(
            path="invocation-manifest.json",
            sha256=digest("2"),
            bytes=1,
            media_type="application/json",
        ),
        invocation_manifest_sha256=digest("3"),
        repetitions=20,
        cases=({"stable_artifacts": {"candidate": stable}},),
    )
    (state / relative).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    marker = qualification_state.record_marker(
        state,
        project,
        step,
        SOURCE,
        GPU,
        "full",
    )
    assert marker["outcome"] == "failed"
    reconciled = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        dry_run=True,
        now=FIXED_RECONCILE_TIME,
    )
    assert step in reconciled["valid_steps"]
    assert all(move["step"] != step for move in reconciled["moves"])


@pytest.mark.parametrize("mutation", ["content", "remove", "symlink", "special"])
def test_marker_rejects_mutation_removal_symlink_and_special_file(
    tmp_path: Path, mutation: str
) -> None:
    state, project = _roots(tmp_path)
    _prepare_step(state, "preflight")
    qualification_state.record_marker(state, project, "preflight", SOURCE, GPU, "full")
    target = state / "gpu.uuid"
    if mutation == "content":
        target.write_text("changed\n", encoding="utf-8")
    elif mutation == "remove":
        target.unlink()
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(state / "source.commit")
    else:
        target.unlink()
        os.mkfifo(target)
    assert not qualification_state.verify_marker(state, project, "preflight", SOURCE, GPU, "full")


def test_marker_rejects_added_file_and_symlink_parent_escape(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"plugin-build"})
    (state / "plugin-build" / "added.txt").write_text("drift\n", encoding="utf-8")
    assert not qualification_state.verify_marker(
        state, project, "plugin-build", SOURCE, GPU, "full"
    )

    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "artifact.txt").write_text("outside\n", encoding="utf-8")
    shutil_target = state / "plugin-build"
    for child in shutil_target.iterdir():
        child.unlink()
    shutil_target.rmdir()
    shutil_target.symlink_to(escaped, target_is_directory=True)
    assert not qualification_state.verify_marker(
        state, project, "plugin-build", SOURCE, GPU, "full"
    )


def test_dependency_marker_byte_drift_invalidates_only_descendants(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"worker-images"})
    cpu_marker = state / "done" / "cpu-verify.json"
    cpu_marker.write_text(cpu_marker.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert qualification_state.verify_marker(state, project, "cpu-verify", SOURCE, GPU, "full")
    assert not qualification_state.verify_marker(
        state, project, "worker-images", SOURCE, GPU, "full"
    )


def test_matrix_and_corpus_bindings_remain_independent_before_gpu_work(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(
        state,
        project,
        {"matrix-live-verification", "corpus-materialization"},
    )
    matrix = json.loads((state / "done" / "matrix-lock.json").read_text(encoding="utf-8"))
    corpus = json.loads(
        (state / "done" / "corpus-materialization.json").read_text(encoding="utf-8")
    )
    lock = MatrixLock.model_validate_json((state / "matrix.lock.json").read_text())
    assert matrix["matrix_lock_sha256"] == lock.lock_sha256
    assert matrix["corpus_identities"] == []
    assert corpus["matrix_lock_sha256"] is None
    assert [item["kind"] for item in corpus["corpus_identities"]] == [
        "core",
        "mobilenet",
        "plugin",
    ]

    _write_corpora(state, suffix="3")
    assert not qualification_state.verify_marker(
        state, project, "corpus-materialization", SOURCE, GPU, "full"
    )


def test_matrix_self_hash_must_validate_before_record(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"worker-images", "gpu-runtime-preflight"})
    _prepare_step(state, "matrix-lock")
    value = json.loads((state / "matrix.lock.json").read_text(encoding="utf-8"))
    value["lock_sha256"] = digest("f")
    _write_json(state / "matrix.lock.json", value)
    with pytest.raises(ValueError, match="self-hash"):
        qualification_state.record_marker(state, project, "matrix-lock", SOURCE, GPU, "full")


def test_external_spec_mutation_invalidates_matrix_lineage_and_descendants(
    tmp_path: Path,
) -> None:
    state, project = _roots(tmp_path)
    specification = tmp_path / "ignored-external-qualification.yaml"
    specification.write_text("kind: Qualification\npolicy: original\n", encoding="utf-8")
    _record_closure(
        state,
        project,
        {"plugin-build"},
        qualification_spec=specification,
    )
    marker = json.loads((state / "done" / "matrix-lock.json").read_text(encoding="utf-8"))
    lineage = marker["qualification_spec_lineage"]
    assert lineage == {
        "resolved_path": str(specification.resolve()),
        "source_sha256": qualification_state._sha256(specification),
        "locked_sha256": qualification_state._sha256(state / "full.yaml"),
    }

    specification.write_text("kind: Qualification\npolicy: mutated\n", encoding="utf-8")

    assert qualification_state.verify_marker(
        state,
        project,
        "worker-images",
        SOURCE,
        GPU,
        "full",
        qualification_spec=specification,
    )
    assert not qualification_state.verify_marker(
        state,
        project,
        "matrix-lock",
        SOURCE,
        GPU,
        "full",
        qualification_spec=specification,
    )
    assert not qualification_state.verify_marker(
        state,
        project,
        "plugin-build",
        SOURCE,
        GPU,
        "full",
        qualification_spec=specification,
    )
    result = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        qualification_spec=specification,
        dry_run=True,
        now=FIXED_RECONCILE_TIME,
    )
    assert {"matrix-lock", "matrix-live-verification", "plugin-build"}.issubset(
        result["invalid_steps"]
    )
    assert {"worker-images", "corpus-materialization"}.issubset(result["valid_steps"])


def test_locked_spec_mutation_invalidates_matrix_lineage(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"matrix-live-verification"})
    (state / "full.yaml").write_text("kind: Qualification\nmutated: true\n", encoding="utf-8")

    assert not qualification_state.verify_marker(state, project, "matrix-lock", SOURCE, GPU, "full")
    assert not qualification_state.verify_marker(
        state, project, "matrix-live-verification", SOURCE, GPU, "full"
    )


def test_runtime_fingerprint_change_invalidates_matrix_but_not_worker_images(
    tmp_path: Path,
) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"matrix-lock"})
    runtime = state / "gpu-runtime-preflight.json"
    _write_json(runtime, {"status": "passed", "fingerprint": digest("d")})
    qualification_state.record_marker(
        state,
        project,
        "gpu-runtime-preflight",
        SOURCE,
        GPU,
        "full",
    )

    assert qualification_state.verify_marker(state, project, "worker-images", SOURCE, GPU, "full")
    assert not qualification_state.verify_marker(state, project, "matrix-lock", SOURCE, GPU, "full")
    result = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        now=FIXED_RECONCILE_TIME,
    )
    assert "matrix-lock" in result["invalid_steps"]
    assert "worker-images" in result["valid_steps"]
    assert not (state / "matrix.lock.json").exists()
    assert (state / "worker-images.json").exists()


def test_current_step_aliases_share_canonical_authority_and_marker(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(
        state,
        project,
        {"matrix-live-verification", "corpus-materialization"},
    )
    _prepare_step(state, "plugin-build", invoked_step="plugin-compile-test")
    payload = qualification_state.record_marker(
        state, project, "plugin-compile-test", SOURCE, GPU, "full"
    )
    assert payload["step"] == "plugin-build"
    assert (state / "done" / "plugin-build.json").is_file()
    assert qualification_state.verify_marker(state, project, "plugin-build", SOURCE, GPU, "full")


def test_authority_paths_are_disjoint_and_dag_is_acyclic() -> None:
    qualification_state.validate_authority()
    observed: list[tuple[str, PurePosixPath]] = []
    for step in qualification_state.STEP_OWNED_PATHS:
        for authored in qualification_state._reconcile_owned_paths(step):
            path = PurePosixPath(authored)
            assert path.parts[0] not in {"done", "stale", "diagnostics"}
            for other_step, other in observed:
                assert path != other, (step, other_step)
                assert not path.is_relative_to(other), (step, other_step)
                assert not other.is_relative_to(path), (step, other_step)
            observed.append((step, path))
    for mode in qualification_state.MODES:
        order = qualification_state._topological_steps(mode)
        assert set(order) == set(qualification_state.MODE_STEPS[mode])
        assert (
            order.index("replay-G2") < order.index("reduction-validation")
            if mode == "full"
            else True
        )
        assert (
            order.index("replay-G7") < order.index("reduction-validation")
            if mode == "full"
            else True
        )


def test_final_dependencies_are_exact_mode_specific_pre_final_set() -> None:
    full = set(qualification_state._dependencies_for("final-evidence", "full"))
    assert full == set(qualification_state.MODE_STEPS["full"]) - {
        "public-failure",
        "final-evidence",
        "terminal-cleanup",
    }
    assert "gpu-smoke" not in full
    assert "dependency-audit" in full
    assert "target-readiness" in full
    assert set(qualification_state._dependencies_for("aa-pilot", "full")) == {
        "matrix-lock",
        "corpus-materialization",
    }
    assert "final-evidence" not in qualification_state.MODE_STEPS["smoke"]
    assert "dependency-audit" not in qualification_state.MODE_STEPS["smoke"]
    assert "dependency-audit" not in qualification_state.MODE_STEPS["sanitizer"]
    assert "profiles" not in qualification_state.MODE_STEPS["sanitizer"]


def test_reconcile_invalidates_descendants_but_preserves_valid_siblings(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"reduction-validation"})
    (state / "reductions" / "G7" / "artifact.txt").write_text("tampered\n", encoding="utf-8")
    dry_run = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        dry_run=True,
        now=FIXED_RECONCILE_TIME,
    )
    assert "replay-G7" in dry_run["invalid_steps"]
    assert "reduction-validation" in dry_run["invalid_steps"]
    assert "replay-G2" in dry_run["valid_steps"]
    assert all(move["step"] != "replay-G2" for move in dry_run["moves"])

    result = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        now=FIXED_RECONCILE_TIME,
    )
    assert (state / "reductions" / "G2").is_dir()
    assert (state / "done" / "replay-G2.json").is_file()
    assert not (state / "reductions" / "G7").exists()
    stale_root = state / result["stale_root"]
    assert (stale_root / "replay-G7" / "reductions" / "G7").is_dir()
    assert (stale_root / "replay-G7" / "done" / "replay-G7.json").is_file()


def test_reconcile_independent_gate_preserves_aa_and_core(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"final-evidence"})
    (state / "profiler-preflight" / "artifact.txt").write_text("tampered\n", encoding="utf-8")

    result = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        dry_run=True,
        now=FIXED_RECONCILE_TIME,
    )

    assert {"profiler-preflight", "plugin-benchmark", "profiles", "final-evidence"}.issubset(
        result["invalid_steps"]
    )
    assert {"aa-pilot", "core-qualification"}.issubset(result["valid_steps"])


def test_reconcile_prepare_drift_invalidates_both_replays(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"reduction-validation"})
    (state / "reductions" / "prepared" / "artifact.txt").write_text("tampered\n", encoding="utf-8")
    result = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        dry_run=True,
        now=FIXED_RECONCILE_TIME,
    )
    assert {
        "reduction-prepare",
        "replay-G2",
        "replay-G7",
        "reduction-validation",
    }.issubset(result["invalid_steps"])


def test_reconcile_preserves_mutated_candidate_reduction_work(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _record_closure(state, project, {"reduction-validation"})
    candidate = state / "reductions" / "candidate-reduction-work" / "artifact.txt"
    candidate.write_text("tampered candidate\n", encoding="utf-8")

    result = qualification_state.reconcile(
        state,
        project,
        source=SOURCE,
        gpu=GPU,
        mode="full",
        now=FIXED_RECONCILE_TIME,
    )

    assert {
        "reduction-prepare",
        "replay-G2",
        "replay-G7",
        "reduction-validation",
    }.issubset(result["invalid_steps"])
    assert not (state / "reductions" / "candidate-reduction-work").exists()
    stale_root = state / result["stale_root"]
    preserved = (
        stale_root
        / "reduction-prepare"
        / "reductions"
        / "candidate-reduction-work"
        / "artifact.txt"
    )
    assert preserved.read_text(encoding="utf-8") == "tampered candidate\n"


def test_stale_and_diagnostics_are_excluded_from_active_inventory(tmp_path: Path) -> None:
    state, project = _roots(tmp_path)
    _prepare_step(state, "preflight")
    qualification_state.record_marker(state, project, "preflight", SOURCE, GPU, "full")
    (state / "stale" / "old").mkdir(parents=True)
    (state / "stale" / "old" / "artifact.txt").write_text("old\n", encoding="utf-8")
    (state / "diagnostics").mkdir()
    (state / "diagnostics" / "failure.json").write_text("{}\n", encoding="utf-8")
    assert qualification_state.verify_marker(state, project, "preflight", SOURCE, GPU, "full")


def _corpus(tmp_path: Path) -> tuple[Path, str, Path]:
    materializer_id = digest("c")
    root = tmp_path / materializer_id.removeprefix("sha256:")
    root.mkdir()
    artifact = root / "model.onnx"
    artifact.write_bytes(b"model")
    _write_json(
        root / "corpus.lock.json",
        {
            "artifacts": [
                {
                    "path": "model.onnx",
                    "bytes": artifact.stat().st_size,
                    "sha256": qualification_state._sha256(artifact),
                }
            ]
        },
    )
    sidecar = root / "materializer.json"
    _write_json(sidecar, {"materializer_sha256": materializer_id})
    return root, materializer_id, sidecar


def test_corpus_verifier_requires_materializer_and_preserves_artifact_validation(
    tmp_path: Path,
) -> None:
    root, materializer_id, _ = _corpus(tmp_path)
    qualification_state._verify_corpus(root.resolve(), materializer_id=materializer_id)
    qualification_state._verify_corpus(root.resolve())
    with pytest.raises(ValueError, match="differs from the expected identity"):
        qualification_state._verify_corpus(root.resolve(), materializer_id=digest("d"))
    (root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its lock"):
        qualification_state._verify_corpus(root.resolve(), materializer_id=materializer_id)


def test_corpus_verifier_accepts_matching_sidecar_and_rejects_inventory_drift(
    tmp_path: Path,
) -> None:
    root, _, sidecar = _corpus(tmp_path)
    qualification_state._verify_corpus(root.resolve(), materializer_sidecar=sidecar)
    (root / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        qualification_state._verify_corpus(root.resolve(), materializer_sidecar=sidecar)


def test_worker_lock_consistency_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digests = [digest(character) for character in ("1", "2")]
    workers = tmp_path / "workers.json"
    matrix = tmp_path / "matrix.json"
    _write_json(workers, {"images": [{"manifest_digest": value} for value in digests]})
    _write_json(
        matrix,
        {
            "environments": [
                {"worker_image": {"manifest_digest": value}} for value in reversed(digests)
            ]
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "verify-worker-lock",
            "--workers",
            str(workers),
            "--matrix",
            str(matrix),
        ],
    )
    assert qualification_state.main() == 0
