"""Cross-file validation for terminal public qualification evidence."""

from __future__ import annotations

import copy
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from scripts.generate_remote_evidence import _failed_publication
from tests.factories import (
    FIXED_TIME,
    environment_lock,
    failure_record,
    reference_environment_lock,
)
from upgrade_guard import publication as publication_module
from upgrade_guard.cli import app
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.common import ArtifactReference, ResultStatus
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode
from upgrade_guard.gates import (
    STEP_ALIASES,
    STEP_OWNED_PATHS,
    direct_step_dependencies,
    expected_publication_steps,
    step_is_bound_to,
)
from upgrade_guard.publication import PublicationValidationError, validate_publication
from upgrade_guard.qualification import compare_stored_run
from upgrade_guard.reduce.public_failure import PublicFailureDisposition, PublicFailureItem
from upgrade_guard.report.model import ReportModel


def _write_passing_publication(root: Path) -> None:
    root.mkdir()
    (root / "done").mkdir()
    (root / "core-run").mkdir()
    baseline = environment_lock(environment_id="baseline", worker_manifest_character="1")
    candidate = environment_lock(environment_id="candidate", worker_manifest_character="2")
    matrix = MatrixLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="EnvironmentLock",
        source_matrix_sha256="sha256:" + "3" * 64,
        gpu_uuid=baseline.probe.gpu.uuid,
        created_at=baseline.probed_at,
        environments=(baseline, candidate),
        lock_sha256="sha256:" + "0" * 64,
    )
    matrix = matrix.model_copy(update={"lock_sha256": matrix.computed_sha256()})
    reference = reference_environment_lock()
    source_commit = "4" * 40
    (root / "matrix.lock.json").write_text(matrix.model_dump_json(indent=2) + "\n")
    (root / "reference-environment.lock.json").write_text(
        reference.model_dump_json(indent=2) + "\n"
    )
    (root / "source.commit").write_text(source_commit + "\n")
    corpus_identities = [
        {
            "kind": kind,
            "root": f".upgrade-guard/corpora/{kind}",
            "lock": f".upgrade-guard/corpora/{kind}/corpus.lock.json",
            "lock_sha256": "sha256:" + characters[0] * 64,
            "materializer_sha256": "sha256:" + characters[1] * 64,
            "inventory_sha256": "sha256:" + characters[2] * 64,
            "reference_environment_sha256": reference.lock_sha256,
        }
        for kind, characters in (
            ("core", "567"),
            ("mobilenet", "89a"),
            ("plugin", "bcd"),
        )
    ]
    corpora = {
        "schema_version": "upgradeguard.dev/corpus-index/v1",
        "reference_environment_sha256": reference.lock_sha256,
        "corpora": {
            value["kind"]: {name: item for name, item in value.items() if name != "kind"}
            for value in corpus_identities
        },
    }
    (root / "corpora.json").write_text(json.dumps(corpora, sort_keys=True) + "\n")
    core = {
        "schema_version": "upgradeguard.dev/qualification-summary/v1",
        "status": "passed",
        "failure_codes": [],
    }
    (root / "core-run" / "qualification-summary.json").write_text(json.dumps(core) + "\n")
    steps = expected_publication_steps("passed")
    _write_fixture_step_outputs(root, steps)
    qualification_lineage = {
        "resolved_path": str((root / "full.yaml").resolve()),
        "source_sha256": sha256_file(root / "full.yaml"),
        "locked_sha256": sha256_file(root / "full.yaml"),
    }
    gate_evidence: dict[str, dict[str, str]] = {}
    for step in steps:
        marker = root / "done" / f"{step}.json"
        dependencies = direct_step_dependencies(step)
        matrix_bound = step == "matrix-lock" or step_is_bound_to(step, "matrix-lock")
        corpus_bound = step == "corpus-materialization" or step_is_bound_to(
            step, "corpus-materialization"
        )
        marker.write_text(
            json.dumps(
                {
                    "schema_version": "upgradeguard.dev/qualification-step/v3",
                    "step": step,
                    "source_git_commit": source_commit,
                    "gpu_uuid": matrix.gpu_uuid,
                    "mode": "full",
                    "outcome": "passed",
                    "inventory": _fixture_marker_inventory(root, step),
                    "direct_dependency_marker_sha256s": {
                        dependency: sha256_file(root / "done" / f"{dependency}.json")
                        for dependency in dependencies
                    },
                    "matrix_lock_sha256": matrix.lock_sha256 if matrix_bound else None,
                    "corpus_identities": corpus_identities if corpus_bound else [],
                    "qualification_spec_lineage": (qualification_lineage if matrix_bound else None),
                },
                sort_keys=True,
            )
            + "\n"
        )
        gate_evidence[step] = {
            "status": "passed",
            "marker_sha256": sha256_file(marker),
        }
    gate_status = dict.fromkeys(steps, "passed")
    images = {
        environment.id: environment.worker_image.canonical_reference
        for environment in matrix.environments
    }
    results = {
        "schema_version": "upgradeguard.dev/published-result-table/v1",
        "status": "passed",
        "failure_codes": [],
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference.lock_sha256,
        "environment_images": images,
        "gate_sequence": list(steps),
        "gate_status": gate_status,
        "gate_evidence": gate_evidence,
        "core_qualification": core,
    }
    results_path = root / "results.json"
    results_path.write_text(json.dumps(results, sort_keys=True) + "\n")

    def artifact(relative: str) -> ArtifactReference:
        path = root / relative
        return ArtifactReference(
            path=relative,
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            media_type="application/json",
        )

    report = ReportModel(
        api_version="upgradeguard.dev/report/v1",
        title="Passing qualification",
        generated_at=FIXED_TIME,
        status=ResultStatus.PASSED,
        baseline_environment_id="baseline",
        candidate_environment_id="candidate",
        stack_attribution="Complete locked stacks",
        result_count=len(steps),
        passed_count=len(steps),
        failed_count=0,
        unsupported_count=0,
        infrastructure_invalid_count=0,
        inconclusive_count=0,
        failures=(),
        evidence=tuple(artifact(f"done/{step}.json") for step in steps),
        warnings=(),
        publication_complete=True,
        source_git_commit=source_commit,
        gpu_uuid=matrix.gpu_uuid,
        matrix_lock_sha256=matrix.lock_sha256,
        reference_environment_lock_sha256=reference.lock_sha256,
        environment_images=images,
        acceptance_gates=dict.fromkeys(steps, ResultStatus.PASSED),
        corpus_provenance=(artifact("corpora.json"),),
        results_artifact=artifact("results.json"),
        measured_sections={"core": "results.json#/core_qualification"},
        reproduction_commands=(("upgrade-guard", "qualify"),),
        methodology=("Frozen inputs",),
        limitations=("One selected GPU",),
    )
    report_model_path = root / "report-model.json"
    report_model_path.write_text(report.model_dump_json(indent=2) + "\n")
    report_path = root / "report.md"
    report_path.write_text("# Passing qualification\n")
    generated_names = {"results.json", "report-model.json", "report.md", "evidence.json"}
    inventory = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in generated_names
    }
    evidence = {
        "schema_version": "upgradeguard.dev/remote-evidence/v1",
        "status": "passed",
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference.lock_sha256,
        "environment_images": images,
        "gate_sequence": list(steps),
        "gate_status": gate_status,
        "gate_evidence": gate_evidence,
        "core_qualification": core,
        "artifacts": inventory,
        "generated_artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (results_path, report_model_path, report_path)
        },
    }
    (root / "evidence.json").write_text(json.dumps(evidence, sort_keys=True) + "\n")
    (root / "logs" / "final-evidence.log").write_text("publication closed\n", encoding="utf-8")
    (root / "done" / "final-evidence.json").write_text("{}\n", encoding="utf-8")
    (root / "cleanup.json").write_text('{"status":"passed"}\n', encoding="utf-8")
    (root / "logs" / "terminal-cleanup.log").write_text("cleanup closed\n", encoding="utf-8")
    (root / "done" / "terminal-cleanup.json").write_text("{}\n", encoding="utf-8")


def _write_fixture_step_outputs(root: Path, steps: tuple[str, ...]) -> None:
    (root / "logs").mkdir()
    for step in steps:
        for authored in STEP_OWNED_PATHS[step]:
            relative = authored.removesuffix("/")
            path = root / relative
            if authored.endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
                if not any(path.iterdir()):
                    (path / "fixture.txt").write_text(f"{step}\n", encoding="utf-8")
            elif not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{step}\n", encoding="utf-8")
        (root / "logs" / f"{step}.log").write_text(f"{step} complete\n", encoding="utf-8")


def _fixture_marker_inventory(root: Path, step: str) -> list[dict[str, int | str]]:
    entries: dict[str, dict[str, int | str]] = {}

    def add(path: Path) -> None:
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        assert stat.S_ISREG(mode)
        entries[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    for authored in STEP_OWNED_PATHS[step]:
        path = root / authored.removesuffix("/")
        if authored.endswith("/"):
            for current_root, directory_names, file_names in os.walk(path):
                assert all(
                    stat.S_ISDIR((Path(current_root) / name).lstat().st_mode)
                    for name in directory_names
                )
                for name in file_names:
                    add(Path(current_root) / name)
        else:
            add(path)
    log_names = (step, *(alias for alias, target in STEP_ALIASES.items() if target == step))
    for name in log_names:
        path = root / "logs" / f"{name}.log"
        if path.exists():
            add(path)
    return [entries[name] for name in sorted(entries)]


def _refresh_terminal_hashes(root: Path) -> None:
    results_path = root / "results.json"
    report_model_path = root / "report-model.json"
    report_path = root / "report.md"
    report = json.loads(report_model_path.read_text(encoding="utf-8"))
    report["results_artifact"].update(
        sha256=sha256_file(results_path),
        bytes=results_path.stat().st_size,
    )
    report_model_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    excluded = {
        "cleanup.json",
        "done/final-evidence.json",
        "done/terminal-cleanup.json",
        "evidence.json",
        "logs/final-evidence.log",
        "logs/terminal-cleanup.log",
        "report-model.json",
        "report.md",
        "results.json",
    }
    evidence["artifacts"] = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    evidence["generated_artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (results_path, report_model_path, report_path)
    }
    (root / "evidence.json").write_text(
        json.dumps(evidence, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_marker_references(root: Path, step: str) -> None:
    marker_path = root / "done" / f"{step}.json"
    marker_sha256 = sha256_file(marker_path)
    for name in ("results.json", "evidence.json"):
        value = json.loads((root / name).read_text(encoding="utf-8"))
        value["gate_evidence"][step]["marker_sha256"] = marker_sha256
        (root / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    report_path = root / "report-model.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for reference in report["evidence"]:
        if reference["path"] == f"done/{step}.json":
            reference["sha256"] = marker_sha256
            reference["bytes"] = marker_path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_terminal_hashes(root)


def test_passing_publication_validates_every_cross_file_identity(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _write_passing_publication(root)

    decision = validate_publication(root, expected_status="passed")

    assert decision.status == "passed"
    assert decision.failure_codes == ()
    assert decision.results["core_qualification"] == decision.core_qualification
    assert compare_stored_run(root) == decision.results
    assert compare_stored_run(root / "core-run") == decision.core_qualification

    result = CliRunner().invoke(app, ["compare", str(root), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == decision.results


def test_publication_rejects_tamper_unindexed_file_and_symlink_parent(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _write_passing_publication(root)
    (root / "report.md").write_text("tampered\n")
    with pytest.raises(PublicationValidationError, match="generated artifact identity"):
        validate_publication(root)

    root = tmp_path / "unindexed"
    _write_passing_publication(root)
    (root / "unindexed.bin").write_bytes(b"not indexed")
    with pytest.raises(PublicationValidationError, match="inventory is incomplete"):
        validate_publication(root)

    root = tmp_path / "nested-generated-name"
    _write_passing_publication(root)
    (root / "core-run" / "results.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="inventory is incomplete"):
        validate_publication(root)

    root = tmp_path / "linked"
    _write_passing_publication(root)
    real_core = tmp_path / "external-core"
    (root / "core-run").rename(real_core)
    (root / "core-run").symlink_to(real_core, target_is_directory=True)
    with pytest.raises(PublicationValidationError, match="parent is unsafe"):
        validate_publication(root)

    root = tmp_path / "unindexed-link"
    _write_passing_publication(root)
    (root / "extra-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(PublicationValidationError, match="contains a symlink"):
        validate_publication(root)

    if hasattr(os, "mkfifo"):
        root = tmp_path / "unindexed-special"
        _write_passing_publication(root)
        os.mkfifo(root / "extra-fifo")
        with pytest.raises(PublicationValidationError, match="special filesystem node"):
            validate_publication(root)


def test_publication_rejects_partial_gate_set_and_contradictory_core(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    _write_passing_publication(root)
    omitted = expected_publication_steps("passed")[-1]
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report-model.json").read_text(encoding="utf-8"))
    for value in (results, evidence):
        value["gate_sequence"].remove(omitted)
        value["gate_status"].pop(omitted)
        value["gate_evidence"].pop(omitted)
    report["acceptance_gates"].pop(omitted)
    report["result_count"] -= 1
    report["passed_count"] -= 1
    (root / "results.json").write_text(json.dumps(results) + "\n", encoding="utf-8")
    (root / "evidence.json").write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    (root / "report-model.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    _refresh_terminal_hashes(root)
    with pytest.raises(PublicationValidationError, match="gate statuses differ"):
        validate_publication(root)

    root = tmp_path / "contradictory-core"
    _write_passing_publication(root)
    core = {
        "schema_version": "upgradeguard.dev/qualification-summary/v1",
        "status": "failed",
        "failure_codes": ["NUMERICAL_REGRESSION"],
    }
    (root / "core-run" / "qualification-summary.json").write_text(
        json.dumps(core) + "\n",
        encoding="utf-8",
    )
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results["core_qualification"] = core
    evidence["core_qualification"] = core
    (root / "results.json").write_text(json.dumps(results) + "\n", encoding="utf-8")
    (root / "evidence.json").write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    _refresh_terminal_hashes(root)
    with pytest.raises(PublicationValidationError, match="non-passing core"):
        validate_publication(root)


def test_publication_rejects_marker_without_v3_lineage(tmp_path: Path) -> None:
    root = tmp_path / "marker-lineage"
    _write_passing_publication(root)
    marker = root / "done" / "matrix-live-verification.json"
    marker.write_text('{"outcome":"passed"}\n', encoding="utf-8")
    _refresh_marker_references(root, "matrix-live-verification")

    with pytest.raises(PublicationValidationError, match="marker lineage differs"):
        validate_publication(root)


def _write_failed_publication(root: Path) -> None:
    _write_passing_publication(root)
    matrix = MatrixLock.model_validate_json((root / "matrix.lock.json").read_text())
    reference = reference_environment_lock()
    source_commit = (root / "source.commit").read_text(encoding="utf-8").strip()
    failure_path = root / "core-run" / "qualification-summary.json"
    failure_detail = root / "core-run" / "failure-detail.json"
    failure_detail.write_text('{"observed_peak_bytes": 2048}\n', encoding="utf-8")
    failure_evidence = ArtifactReference(
        path=failure_detail.relative_to(failure_path.parent).as_posix(),
        sha256=sha256_file(failure_detail),
        bytes=failure_detail.stat().st_size,
        media_type="application/json",
    )
    failure = failure_record(FailureCode.MEMORY_REGRESSION).model_copy(
        update={"evidence": (failure_evidence,)}
    )
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": "failed",
                "failure_codes": [failure.code.value],
                "failures": [failure.model_dump(mode="json")],
                "cases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    disposition_root = root / "public-failure"
    disposition_root.mkdir()
    source_artifact = ArtifactReference(
        path=failure_path.relative_to(root).as_posix(),
        sha256=sha256_file(failure_path),
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
        disposition_sha256="sha256:" + "0" * 64,
    )
    disposition = PublicFailureDisposition.model_validate(
        unvalidated.model_copy(update={"disposition_sha256": unvalidated.computed_sha256()})
    )
    (disposition_root / "disposition.json").write_text(
        disposition.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    failed_sequence = expected_publication_steps("failed", failure_step="core-qualification")
    for marker in (root / "done").glob("*.json"):
        if marker.stem not in failed_sequence:
            marker.unlink()
    for generated in (
        "cleanup.json",
        "evidence.json",
        "report-model.json",
        "report.md",
        "results.json",
        "done/final-evidence.json",
        "done/terminal-cleanup.json",
        "logs/final-evidence.log",
        "logs/terminal-cleanup.log",
    ):
        path = root / generated
        if path.exists():
            path.unlink()
    passing_sequence = expected_publication_steps("passed")
    for step in passing_sequence[passing_sequence.index("core-qualification") + 1 :]:
        (root / "logs" / f"{step}.log").unlink(missing_ok=True)
    core_marker_path = root / "done" / "core-qualification.json"
    core_marker = json.loads(core_marker_path.read_text(encoding="utf-8"))
    core_marker["outcome"] = "failed"
    core_marker["inventory"] = _fixture_marker_inventory(root, "core-qualification")
    core_marker_path.write_text(json.dumps(core_marker, sort_keys=True) + "\n", encoding="utf-8")
    (root / "logs" / "public-failure.log").write_text(
        "public failure disposition closed\n", encoding="utf-8"
    )
    public_marker_path = root / "done" / "public-failure.json"
    public_marker = {
        "schema_version": "upgradeguard.dev/qualification-step/v3",
        "step": "public-failure",
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "mode": "full",
        "outcome": "passed",
        "inventory": _fixture_marker_inventory(root, "public-failure"),
        "direct_dependency_marker_sha256s": {
            dependency: sha256_file(root / "done" / f"{dependency}.json")
            for dependency in direct_step_dependencies(
                "public-failure", failure_step="core-qualification"
            )
        },
        "matrix_lock_sha256": matrix.lock_sha256,
        "corpus_identities": core_marker["corpus_identities"],
        "qualification_spec_lineage": core_marker["qualification_spec_lineage"],
    }
    public_marker_path.write_text(
        json.dumps(public_marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_evidence = {
        name: {
            "status": "failed" if name == "core-qualification" else "passed",
            "marker_sha256": sha256_file(root / "done" / f"{name}.json"),
        }
        for name in failed_sequence
    }
    _failed_publication(
        state=root,
        output=root / "evidence.json",
        matrix=matrix,
        reference_environment=reference,
        source_commit=source_commit,
        gate_evidence=gate_evidence,
    )


def test_failed_publication_validates_typed_not_applicable_disposition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "failed-publication"
    _write_failed_publication(root)

    decision = validate_publication(root, expected_status="failed")

    assert decision.status == "failed"
    assert decision.failure_codes == (FailureCode.MEMORY_REGRESSION,)
    assert decision.results["failure_step"] == "core-qualification"
    assert decision.results["reduction"]["status"] == "not_applicable"
    assert decision.results["reproduction"] == decision.results["reduction"]


def test_failed_publication_rejects_tampered_disposition(tmp_path: Path) -> None:
    root = tmp_path / "failed-disposition"
    _write_failed_publication(root)
    disposition_path = root / "public-failure" / "disposition.json"
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["disposition_sha256"] = "sha256:" + "0" * 64
    disposition_path.write_text(json.dumps(disposition) + "\n", encoding="utf-8")
    report_path = root / "report-model.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for reference in report["evidence"]:
        if reference["path"] == "public-failure/disposition.json":
            reference["sha256"] = sha256_file(disposition_path)
            reference["bytes"] = disposition_path.stat().st_size
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    marker_path = root / "done" / "public-failure.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["inventory"] = _fixture_marker_inventory(root, "public-failure")
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    _refresh_marker_references(root, "public-failure")

    with pytest.raises(PublicationValidationError, match="disposition is invalid"):
        validate_publication(root)


def test_publication_rejects_unsafe_roots_paths_and_json(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(PublicationValidationError, match="unavailable"):
        publication_module._safe_root(missing)

    regular = tmp_path / "regular"
    regular.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="not a directory"):
        publication_module._safe_root(regular)

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked-root"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(PublicationValidationError, match="cannot be a symlink"):
        publication_module._safe_root(linked)

    with pytest.raises(PublicationValidationError, match="path is unsafe"):
        publication_module._regular_file(real, "../escape")
    with pytest.raises(PublicationValidationError, match="parent is unavailable"):
        publication_module._regular_file(real, "missing/child.json")
    with pytest.raises(PublicationValidationError, match="publication is missing"):
        publication_module._regular_file(real, "missing.json")
    directory = real / "directory.json"
    directory.mkdir()
    with pytest.raises(PublicationValidationError, match="is not regular"):
        publication_module._regular_file(real, "directory.json")

    invalid = real / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="JSON is invalid"):
        publication_module._json_object(invalid)
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="not an object"):
        publication_module._json_object(invalid)


def test_publication_rejects_invalid_report_and_empty_markdown(tmp_path: Path) -> None:
    root = tmp_path / "invalid-report"
    _write_passing_publication(root)
    (root / "report-model.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="report model is invalid"):
        validate_publication(root)

    root = tmp_path / "empty-report"
    _write_passing_publication(root)
    (root / "report.md").write_text("\n", encoding="utf-8")
    _refresh_terminal_hashes(root)
    with pytest.raises(PublicationValidationError, match="Markdown report is empty"):
        validate_publication(root)


def test_terminal_status_failure_code_and_schema_branches(tmp_path: Path) -> None:
    root = tmp_path / "terminal-contract"
    _write_passing_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    report = ReportModel.model_validate_json((root / "report-model.json").read_text())

    with pytest.raises(PublicationValidationError, match="status differs"):
        publication_module._status(
            {**evidence, "status": "failed"},
            results,
            report,
            expected_status=None,
        )
    with pytest.raises(PublicationValidationError, match="wrong terminal status"):
        publication_module._status(evidence, results, report, expected_status="failed")
    with pytest.raises(PublicationValidationError, match="report model is incomplete"):
        publication_module._status(
            evidence,
            results,
            report.model_copy(update={"publication_complete": False}),
            expected_status=None,
        )

    with pytest.raises(PublicationValidationError, match="failure codes are invalid"):
        publication_module._failure_codes({}, evidence, status="passed")
    with pytest.raises(PublicationValidationError, match="failure codes are unknown"):
        publication_module._failure_codes({"failure_codes": ["UNKNOWN"]}, evidence, status="passed")
    with pytest.raises(PublicationValidationError, match="contain duplicates"):
        publication_module._failure_codes(
            {"failure_codes": ["MEMORY_REGRESSION", "MEMORY_REGRESSION"]},
            evidence,
            status="passed",
        )
    with pytest.raises(PublicationValidationError, match="passing publication contains"):
        publication_module._failure_codes(
            {"failure_codes": ["MEMORY_REGRESSION"]}, evidence, status="passed"
        )
    with pytest.raises(PublicationValidationError, match="not domain failures"):
        publication_module._failure_codes(
            {"failure_codes": ["INFRASTRUCTURE_INVALID"]},
            {**evidence, "failure_codes": ["INFRASTRUCTURE_INVALID"]},
            status="failed",
        )

    with pytest.raises(PublicationValidationError, match="remote evidence schema"):
        publication_module._schemas({}, results)
    with pytest.raises(PublicationValidationError, match="result schema"):
        publication_module._schemas(evidence, {})


def test_generated_reference_and_inventory_branches(tmp_path: Path) -> None:
    root = tmp_path / "artifact-contract"
    _write_passing_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))

    with pytest.raises(PublicationValidationError, match="inventory is missing"):
        publication_module._generated_artifacts(root, {})
    changed_generated = copy.deepcopy(evidence)
    changed_generated["generated_artifacts"]["results.json"]["bytes"] += 1
    with pytest.raises(PublicationValidationError, match="identity differs"):
        publication_module._generated_artifacts(root, changed_generated)

    with pytest.raises(PublicationValidationError, match="lacks a required"):
        publication_module._artifact_reference(root, None)
    report_path = root / "report-model.json"
    report_reference = ArtifactReference(
        path="report-model.json",
        sha256=sha256_file(report_path),
        bytes=report_path.stat().st_size,
        media_type="application/json",
    )
    with pytest.raises(PublicationValidationError, match="path differs"):
        publication_module._artifact_reference(root, report_reference, expected_path="results.json")
    with pytest.raises(PublicationValidationError, match="artifact hash differs"):
        publication_module._artifact_reference(
            root,
            report_reference.model_copy(update={"bytes": report_reference.bytes + 1}),
        )

    with pytest.raises(PublicationValidationError, match="inventory is missing"):
        publication_module._inventory(root, {})
    unsafe_inventory = copy.deepcopy(evidence)
    unsafe_inventory["artifacts"] = {"../escape": {}}
    with pytest.raises(PublicationValidationError, match="inventory is incomplete"):
        publication_module._inventory(root, unsafe_inventory)
    bad_identity = copy.deepcopy(evidence)
    first = next(iter(bad_identity["artifacts"].values()))
    first["bytes"] += 1
    with pytest.raises(PublicationValidationError, match="inventory differs"):
        publication_module._inventory(root, bad_identity)

    assert not publication_module._safe_relative("")
    assert not publication_module._safe_relative("/absolute")
    assert not publication_module._safe_relative("a/../b")
    assert not publication_module._safe_relative("a\\b")
    assert not publication_module._safe_relative("a\x00b")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("invalid-matrix", "provenance lock is invalid"),
        ("matrix-self-hash", "matrix lock self-hash differs"),
        ("reference-self-hash", "reference environment lock self-hash differs"),
        ("terminal-provenance", "publication provenance differs"),
        ("report-provenance", "report provenance differs"),
        ("worker-images", "worker image provenance differs"),
    ],
)
def test_publication_provenance_rejects_malformed_or_tampered_identity(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = tmp_path / case
    _write_passing_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    report = ReportModel.model_validate_json((root / "report-model.json").read_text())
    if case == "invalid-matrix":
        (root / "matrix.lock.json").write_text("{\n", encoding="utf-8")
    elif case == "matrix-self-hash":
        matrix = json.loads((root / "matrix.lock.json").read_text(encoding="utf-8"))
        matrix["lock_sha256"] = "sha256:" + "0" * 64
        (root / "matrix.lock.json").write_text(json.dumps(matrix), encoding="utf-8")
    elif case == "reference-self-hash":
        reference = json.loads(
            (root / "reference-environment.lock.json").read_text(encoding="utf-8")
        )
        reference["lock_sha256"] = "sha256:" + "0" * 64
        (root / "reference-environment.lock.json").write_text(
            json.dumps(reference), encoding="utf-8"
        )
    elif case == "terminal-provenance":
        evidence["source_git_commit"] = "5" * 40
    elif case == "report-provenance":
        report = report.model_copy(update={"source_git_commit": "5" * 40})
    else:
        evidence["environment_images"] = {}

    with pytest.raises(PublicationValidationError, match=message):
        publication_module._provenance(root, evidence, results, report)


def test_gate_contract_rejects_prefix_status_evidence_and_marker_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gate-contract"
    _write_passing_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    report = ReportModel.model_validate_json((root / "report-model.json").read_text())

    with pytest.raises(PublicationValidationError, match="gate prefix is invalid"):
        publication_module._gates(
            root,
            evidence,
            {**results, "failure_step": "not-a-domain-step"},
            report,
            status="failed",
        )

    invalid_statuses = dict.fromkeys(expected_publication_steps("passed"), "unsupported")
    invalid_terminal = {
        **results,
        "gate_status": invalid_statuses,
    }
    invalid_evidence = {
        **evidence,
        "gate_status": invalid_statuses,
    }
    invalid_report = SimpleNamespace(
        acceptance_gates={name: SimpleNamespace(value="unsupported") for name in invalid_statuses}
    )
    with pytest.raises(PublicationValidationError, match="gate status is invalid"):
        publication_module._gates(
            root,
            invalid_evidence,
            invalid_terminal,
            invalid_report,
            status="passed",
        )

    bad_evidence = copy.deepcopy(results)
    bad_evidence["gate_evidence"] = []
    with pytest.raises(PublicationValidationError, match="gate evidence differs"):
        publication_module._gates(
            root,
            {**evidence, "gate_evidence": []},
            bad_evidence,
            report,
            status="passed",
        )

    bad_marker = copy.deepcopy(results)
    first_step = expected_publication_steps("passed")[0]
    bad_marker["gate_evidence"][first_step]["marker_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(PublicationValidationError, match="gate marker differs"):
        publication_module._gates(
            root,
            {**evidence, "gate_evidence": bad_marker["gate_evidence"]},
            bad_marker,
            report,
            status="passed",
        )

    with pytest.raises(PublicationValidationError, match="lineage identity is invalid"):
        publication_module._validate_marker_chain(
            root,
            (),
            {},
            source_git_commit=None,
            gpu_uuid=results["gpu_uuid"],
            matrix_lock_sha256=results["matrix_lock_sha256"],
            failure_step=None,
        )


def test_retained_marker_inventory_rejects_missing_unsafe_empty_and_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "marker-inventory"
    _write_passing_publication(root)
    with pytest.raises(PublicationValidationError, match="unknown step"):
        publication_module._retained_step_inventory(root, "unknown-step")

    (root / "logs" / "preflight.log").unlink()
    with pytest.raises(PublicationValidationError, match="lacks a closed step log"):
        publication_module._retained_step_inventory(root, "preflight")

    capacity = root / "capacity"
    shutil.rmtree(capacity)
    with pytest.raises(PublicationValidationError, match="directory is absent"):
        publication_module._retained_step_inventory(root, "capacity-preflight")

    capacity.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="directory is unsafe"):
        publication_module._retained_step_inventory(root, "capacity-preflight")
    capacity.unlink()
    capacity.mkdir()
    with pytest.raises(PublicationValidationError, match="directory is empty"):
        publication_module._retained_step_inventory(root, "capacity-preflight")

    outside = tmp_path / "outside"
    outside.mkdir()
    (capacity / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PublicationValidationError, match="directory is unsafe"):
        publication_module._retained_step_inventory(root, "capacity-preflight")

    entries = {
        "source.commit": {
            "path": "source.commit",
            "bytes": 0,
            "sha256": "sha256:" + "0" * 64,
        }
    }
    with pytest.raises(PublicationValidationError, match="inventory duplicates"):
        publication_module._inventory_retained_file(root, "source.commit", entries)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("lineage-shape", "qualification lineage is invalid"),
        ("lineage-value", "qualification lineage differs"),
        ("corpus-schema", "corpus index is invalid"),
        ("corpus-set", "corpus index is incomplete"),
        ("corpus-shape", "corpus identity is invalid"),
        ("corpus-value", "corpus identity differs"),
    ],
)
def test_retained_lineage_and_corpus_identity_fail_closed(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = tmp_path / case
    _write_passing_publication(root)
    if case.startswith("lineage"):
        marker_path = root / "done" / "matrix-lock.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if case == "lineage-shape":
            marker["qualification_spec_lineage"] = {}
        else:
            marker["qualification_spec_lineage"]["locked_sha256"] = "sha256:" + "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    else:
        corpora_path = root / "corpora.json"
        corpora = json.loads(corpora_path.read_text(encoding="utf-8"))
        if case == "corpus-schema":
            corpora["unexpected"] = True
        elif case == "corpus-set":
            corpora["corpora"].pop("mobilenet")
        elif case == "corpus-shape":
            corpora["corpora"]["plugin"] = {}
        else:
            corpora["corpora"]["plugin"]["root"] = "/absolute"
        corpora_path.write_text(json.dumps(corpora), encoding="utf-8")
    with pytest.raises(PublicationValidationError, match=message):
        if case.startswith("lineage"):
            publication_module._retained_qualification_lineage(root)
        else:
            publication_module._retained_corpus_identities(root)

    assert not publication_module._sha256_string(None)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("embedded", "core qualification differ"),
        ("schema", "schema version is unsupported"),
        ("codes-type", "failure codes are invalid"),
        ("codes-unknown", "failure code is unknown"),
        ("status-codes", "status and failure codes differ"),
        ("failed-core", "failed core gate differs"),
        ("later-core", "later failed gate has a non-passing core"),
        ("failed-step", "failed publication step is invalid"),
    ],
)
def test_core_qualification_rejects_malformed_or_contradictory_decisions(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = tmp_path / case
    _write_passing_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    core_path = root / "core-run" / "qualification-summary.json"
    core = json.loads(core_path.read_text(encoding="utf-8"))
    status = "passed"
    codes: tuple[FailureCode, ...] = ()
    if case == "embedded":
        results["core_qualification"] = {"different": True}
    else:
        if case == "schema":
            core["schema_version"] = "unsupported"
        elif case == "codes-type":
            core["failure_codes"] = None
        elif case == "codes-unknown":
            core["failure_codes"] = ["UNKNOWN"]
        elif case == "status-codes":
            core["status"] = "failed"
        elif case in {"failed-core", "later-core"}:
            core["status"] = "failed"
            core["failure_codes"] = ["MEMORY_REGRESSION"]
            status = "failed"
            codes = (FailureCode.NUMERICAL_REGRESSION,)
            results["failure_step"] = (
                "core-qualification" if case == "failed-core" else "plugin-matrix"
            )
        else:
            status = "failed"
            codes = (FailureCode.MEMORY_REGRESSION,)
            results["failure_step"] = "invalid-step"
        core_path.write_text(json.dumps(core), encoding="utf-8")
        results["core_qualification"] = core
        evidence["core_qualification"] = core
    with pytest.raises(PublicationValidationError, match=message):
        publication_module._core_qualification(
            root,
            evidence,
            results,
            status=status,
            codes=codes,
        )

    assert (
        publication_module._summary_status((FailureCode.INFRASTRUCTURE_INVALID,))
        == "infrastructure_invalid"
    )
    assert publication_module._summary_status((FailureCode.INCONCLUSIVE,)) == "inconclusive"


def test_failure_payload_rejects_malformed_passing_failure_inventory(tmp_path: Path) -> None:
    root = tmp_path / "passing-failures"
    _write_passing_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    report = ReportModel.model_validate_json((root / "report-model.json").read_text())
    failure = failure_record(FailureCode.MEMORY_REGRESSION)

    with pytest.raises(PublicationValidationError, match="published failures are invalid"):
        publication_module._failure_payload(
            root,
            {**evidence, "failures": {}},
            {**results, "failures": {}},
            report,
            status="passed",
            codes=(),
        )
    with pytest.raises(PublicationValidationError, match="failure record is invalid"):
        publication_module._failure_payload(
            root,
            {**evidence, "failures": [{}]},
            {**results, "failures": [{}]},
            report,
            status="passed",
            codes=(),
        )
    with pytest.raises(PublicationValidationError, match="report failure records differ"):
        publication_module._failure_payload(
            root,
            evidence,
            results,
            report.model_copy(update={"failures": (failure,)}),
            status="passed",
            codes=(),
        )
    serialized = [failure.model_dump(mode="json")]
    with pytest.raises(PublicationValidationError, match="passing publication contains"):
        publication_module._failure_payload(
            root,
            {**evidence, "failures": serialized},
            {**results, "failures": serialized},
            report.model_copy(update={"failures": (failure,)}),
            status="passed",
            codes=(),
        )


def test_failed_payload_and_disposition_reject_tampered_links(tmp_path: Path) -> None:
    root = tmp_path / "failed-links"
    _write_failed_publication(root)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    report = ReportModel.model_validate_json((root / "report-model.json").read_text())
    failure = report.failures[0]
    codes = (FailureCode.MEMORY_REGRESSION,)

    other = failure_record(FailureCode.PERFORMANCE_REGRESSION)
    other_serialized = [other.model_dump(mode="json")]
    with pytest.raises(PublicationValidationError, match="exact typed failure records"):
        publication_module._failure_payload(
            root,
            {**evidence, "failures": other_serialized},
            {**results, "failures": other_serialized},
            report.model_copy(update={"failures": (other,)}),
            status="failed",
            codes=codes,
        )

    with pytest.raises(PublicationValidationError, match="no typed failure root"):
        publication_module._failure_payload(
            root,
            evidence,
            {**results, "gate_status": {}},
            report,
            status="failed",
            codes=codes,
        )

    no_evidence = failure.model_copy(update={"evidence": ()})
    no_evidence_serialized = [no_evidence.model_dump(mode="json")]
    with pytest.raises(PublicationValidationError, match="record lacks evidence"):
        publication_module._failure_payload(
            root,
            {**evidence, "failures": no_evidence_serialized},
            {**results, "failures": no_evidence_serialized},
            report.model_copy(update={"failures": (no_evidence,)}),
            status="failed",
            codes=codes,
        )

    with pytest.raises(PublicationValidationError, match="failed publication step differs"):
        publication_module._failure_payload(
            root,
            {**evidence, "failure_step": "plugin-matrix"},
            results,
            report,
            status="failed",
            codes=codes,
        )
    with pytest.raises(PublicationValidationError, match="evidence is missing"):
        publication_module._failure_payload(
            root,
            evidence,
            {key: value for key, value in results.items() if key != "failure_evidence"},
            report,
            status="failed",
            codes=codes,
        )
    with pytest.raises(PublicationValidationError, match="artifact is invalid"):
        publication_module._failure_payload(
            root,
            evidence,
            {**results, "failure_evidence": {"artifact": {}, "value": {}}},
            report,
            status="failed",
            codes=codes,
        )
    with pytest.raises(PublicationValidationError, match="artifact value differs"):
        publication_module._failure_payload(
            root,
            evidence,
            {
                **results,
                "failure_evidence": {
                    **results["failure_evidence"],
                    "value": {"tampered": True},
                },
            },
            report,
            status="failed",
            codes=codes,
        )

    disposition_reference = next(
        item for item in report.evidence if item.path == "public-failure/disposition.json"
    )
    without_disposition = report.model_copy(
        update={
            "evidence": tuple(item for item in report.evidence if item != disposition_reference)
        }
    )
    failure_artifact = ArtifactReference.model_validate(results["failure_evidence"]["artifact"])
    with pytest.raises(PublicationValidationError, match="disposition identity differs"):
        publication_module._failure_disposition(
            root,
            evidence,
            results,
            without_disposition,
            report.failures,
            failure_artifact,
        )
    with pytest.raises(PublicationValidationError, match="reduction disposition differs"):
        publication_module._failure_disposition(
            root,
            evidence,
            {**results, "reduction": {}},
            report,
            report.failures,
            failure_artifact,
        )
