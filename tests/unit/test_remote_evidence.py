"""Fail-closed validation for generated remote publication evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_remote_evidence import (
    AUDITED_DEPENDENCY_SCOPES,
    BENCHMARK_ORDER_SCHEDULE,
    _build_manifest_table,
    _failed_publication,
    _inventory,
    _source_commit,
    _validate_clean_replays,
    _validate_cuda_benchmark,
    _validate_dependency_triage,
    _validate_pip_audit,
    _validate_sanitizer_evidence,
    _validate_spdx,
)
from scripts.validate_profiler_outputs import validate_summary
from tests.factories import failure_record, reference_environment_lock
from tests.unit.test_publication import _fixture_marker_inventory, _write_passing_publication
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode
from upgrade_guard.gates import (
    direct_step_dependencies,
    expected_publication_steps,
    step_is_bound_to,
)
from upgrade_guard.publication import validate_publication
from upgrade_guard.reduce.public_failure import PublicFailureDisposition, PublicFailureItem
from upgrade_guard.report.model import ReportModel


def test_semantic_publication_inputs_are_validated(tmp_path: Path) -> None:
    image = "registry.example/worker@sha256:" + "1" * 64
    sbom = tmp_path / "worker.spdx.json"
    sbom.write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "documentComment": f"Observed package inventory for {image}",
                "packages": [{"name": "package"}],
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps([{"name": "package", "vulns": []}]), encoding="utf-8")
    profile = tmp_path / "profile.csv"
    profile.write_text("Name,Time\nresidualRmsNormFloat4,1\n", encoding="utf-8")

    _validate_spdx(sbom, expected_image=image)
    _validate_pip_audit(audit)
    validate_summary(profile, summary_kind="nsys")

    audit.write_text(
        json.dumps([{"name": "package", "vulns": [{"id": "CVE-TEST"}]}]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="require triage"):
        _validate_pip_audit(audit)
    profile.write_text("TensorRT Release banner only\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="selected kernel"):
        validate_summary(profile, summary_kind="nsys")


def test_dependency_triage_requires_every_hash_locked_python_scope(tmp_path: Path) -> None:
    triage = tmp_path / "triage.json"
    value = {
        "status": "passed",
        "audited_scopes": list(AUDITED_DEPENDENCY_SCOPES),
        "claim": (
            "Passing audits apply only to the explicit Python scopes and do not claim "
            "vulnerability-free images."
        ),
    }
    triage.write_text(json.dumps(value), encoding="utf-8")
    assert _validate_dependency_triage(triage) == value
    value["audited_scopes"].pop()
    triage.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="every hash-locked"):
        _validate_dependency_triage(triage)


def test_inventory_excludes_active_log_and_generated_outputs(tmp_path: Path) -> None:
    state = tmp_path / ("1" * 40)
    (state / "logs").mkdir(parents=True)
    retained = state / "retained.json"
    retained.write_text("{}\n", encoding="utf-8")
    (state / "stale").mkdir()
    (state / "stale" / "old.bin").write_bytes(b"old")
    (state / "diagnostics").mkdir()
    (state / "diagnostics" / "failure.json").write_text("{}\n", encoding="utf-8")
    (state / "logs" / "final-evidence.log").write_text("still changing", encoding="utf-8")
    output = state / "evidence.json"
    output.write_text("stale", encoding="utf-8")

    inventory = _inventory(state, output)

    assert set(inventory) == {"retained.json"}


def test_inventory_rejects_active_symlink_nodes(tmp_path: Path) -> None:
    state = tmp_path / ("1" * 40)
    state.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (state / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="directory is unsafe"):
        _inventory(state, state / "evidence.json")


def test_source_commit_identity_is_independent_of_public_output_directory_name(
    tmp_path: Path,
) -> None:
    state = tmp_path / "operator-selected-output"
    state.mkdir()
    commit = "1" * 40
    (state / "source.commit").write_text(commit + "\n", encoding="utf-8")
    assert _source_commit(state) == commit
    (state / "source.commit").write_text("not-a-commit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid"):
        _source_commit(state)


def test_failed_publication_is_distinct_truthful_and_not_reducible(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write_passing_publication(state)
    matrix = MatrixLock.model_validate_json((state / "matrix.lock.json").read_text())
    reference = reference_environment_lock()
    source_commit = (state / "source.commit").read_text(encoding="utf-8").strip()
    failure_path = state / "core-run" / "qualification-summary.json"
    failure_detail = state / "core-run" / "failure-detail.json"
    failure_detail.write_text('{"observed_peak_bytes": 2048}\n', encoding="utf-8")
    evidence = ArtifactReference(
        path=failure_detail.relative_to(failure_path.parent).as_posix(),
        sha256=sha256_file(failure_detail),
        bytes=failure_detail.stat().st_size,
        media_type="application/json",
    )
    failure = failure_record(FailureCode.MEMORY_REGRESSION).model_copy(
        update={"evidence": (evidence,)}
    )
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": "failed",
                "failure_codes": ["MEMORY_REGRESSION"],
                "failures": [failure.model_dump(mode="json")],
                "cases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    disposition_root = state / "public-failure"
    disposition_root.mkdir()
    source_artifact = ArtifactReference(
        path=failure_path.relative_to(state).as_posix(),
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
    for marker in (state / "done").glob("*.json"):
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
        path = state / generated
        if path.exists():
            path.unlink()
    passing_sequence = expected_publication_steps("passed")
    for step in passing_sequence[passing_sequence.index("core-qualification") + 1 :]:
        log = state / "logs" / f"{step}.log"
        if log.exists():
            log.unlink()
    core_marker_path = state / "done" / "core-qualification.json"
    core_marker = json.loads(core_marker_path.read_text(encoding="utf-8"))
    core_marker["outcome"] = "failed"
    core_marker["inventory"] = _fixture_marker_inventory(state, "core-qualification")
    core_marker_path.write_text(json.dumps(core_marker, sort_keys=True) + "\n", encoding="utf-8")
    (state / "logs" / "public-failure.log").write_text(
        "public failure disposition closed\n", encoding="utf-8"
    )
    public_marker_path = state / "done" / "public-failure.json"
    public_marker = {
        "schema_version": "upgradeguard.dev/qualification-step/v3",
        "step": "public-failure",
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "mode": "full",
        "outcome": "passed",
        "inventory": _fixture_marker_inventory(state, "public-failure"),
        "direct_dependency_marker_sha256s": {
            dependency: sha256_file(state / "done" / f"{dependency}.json")
            for dependency in direct_step_dependencies(
                "public-failure", failure_step="core-qualification"
            )
        },
        "matrix_lock_sha256": matrix.lock_sha256,
        "corpus_identities": core_marker["corpus_identities"],
        "qualification_spec_lineage": core_marker["qualification_spec_lineage"],
    }
    assert step_is_bound_to("public-failure", "matrix-lock", failure_step="core-qualification")
    public_marker_path.write_text(
        json.dumps(public_marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_evidence = {}
    for name in failed_sequence:
        status = "failed" if name == "core-qualification" else "passed"
        marker = state / "done" / f"{name}.json"
        gate_evidence[name] = {
            "status": status,
            "marker_sha256": sha256_file(marker),
        }

    _failed_publication(
        state=state,
        output=state / "evidence.json",
        matrix=matrix,
        reference_environment=reference,
        source_commit=source_commit,
        gate_evidence=gate_evidence,
    )

    validate_publication(state, expected_status="failed")

    results = json.loads((state / "results.json").read_text(encoding="utf-8"))
    evidence = json.loads((state / "evidence.json").read_text(encoding="utf-8"))
    report = ReportModel.model_validate_json((state / "report-model.json").read_text())
    assert results["status"] == evidence["status"] == report.status.value == "failed"
    assert results["failure_codes"] == ["MEMORY_REGRESSION"]
    assert results["gate_status"]["core-qualification"] == "failed"
    assert results["gate_status"]["preflight"] == "passed"
    assert results["reduction"]["status"] == "not_applicable"
    assert results["reproduction"]["status"] == "not_applicable"
    assert "plugin-matrix" not in results["gate_status"]
    assert "explicitly not applicable" in (state / "report.md").read_text()


def test_cuda_benchmark_requires_twenty_seeded_balanced_pairs() -> None:
    pairs = [{"order": order} for order in BENCHMARK_ORDER_SCHEDULE]
    _validate_cuda_benchmark({"order_seed": 20260813, "cases": [{"pairs": pairs}]})
    pairs[-1]["order"] = "scalar_then_optimized"
    with pytest.raises(RuntimeError, match="balanced"):
        _validate_cuda_benchmark({"order_seed": 20260813, "cases": [{"pairs": pairs}]})


def test_reduction_requires_both_clean_typed_cli_replays() -> None:
    digest = "sha256:" + "1" * 64
    value = {
        "clean_bundles": {seed: {"bundle_manifest_sha256": digest} for seed in ("G2", "G7")},
        "clean_replays": {
            "G2": {
                "status": "passed",
                "expected_failure_code": "NUMERICAL_REGRESSION",
                "observed_failure_code": "NUMERICAL_REGRESSION",
                "bundle_manifest_sha256": digest,
                "step_results": ["build", "seed"],
            },
            "G7": {
                "status": "passed",
                "expected_failure_code": "PROFILE_REJECTED",
                "observed_failure_code": "PROFILE_REJECTED",
                "bundle_manifest_sha256": digest,
                "step_results": ["build", "seed"],
            },
            "G5": {
                "status": "passed",
                "fresh_directory": True,
                "expected_failure_code": "PERFORMANCE_REGRESSION",
                "observed_failure_code": "PERFORMANCE_REGRESSION",
                "reduced_pairs_sha256": digest,
                "observations": [{"outcome": "regression"}],
            },
        },
    }
    _validate_clean_replays(value)
    value["clean_replays"].pop("G7")
    with pytest.raises(RuntimeError, match="exactly G2, G5, and G7"):
        _validate_clean_replays(value)


def test_reduction_rejects_declared_code_that_differs_from_observed_replay() -> None:
    digest = "sha256:" + "1" * 64
    value = {
        "clean_bundles": {seed: {"bundle_manifest_sha256": digest} for seed in ("G2", "G7")},
        "clean_replays": {
            "G2": {
                "status": "passed",
                "expected_failure_code": "NUMERICAL_REGRESSION",
                "observed_failure_code": "EXECUTION_FAILED",
                "bundle_manifest_sha256": digest,
                "step_results": ["build", "seed"],
            },
            "G7": {
                "status": "passed",
                "expected_failure_code": "PROFILE_REJECTED",
                "observed_failure_code": "PROFILE_REJECTED",
                "bundle_manifest_sha256": digest,
                "step_results": ["build", "seed"],
            },
            "G5": {
                "status": "passed",
                "fresh_directory": True,
                "expected_failure_code": "PERFORMANCE_REGRESSION",
                "observed_failure_code": "PERFORMANCE_REGRESSION",
                "reduced_pairs_sha256": digest,
                "observations": [{"outcome": "regression"}],
            },
        },
    }

    with pytest.raises(RuntimeError, match="expected failure"):
        _validate_clean_replays(value)


def test_sanitizer_controls_and_diagnostic_are_hash_bound(tmp_path: Path) -> None:
    root = tmp_path / "sanitizers"
    root.mkdir()
    diagnostic = root / "sanitizer-tail-oob.log"
    diagnostic.write_text("Invalid __global__ read\nERROR SUMMARY: 1 error\n", encoding="utf-8")
    from upgrade_guard.contracts.base import sha256_file

    (root / "sanitizer-seed.json").write_text(
        json.dumps(
            {
                "expected": "SANITIZER_FAILURE",
                "classification": "SANITIZER_FAILURE",
                "mechanism": "vectorized_tail_out_of_bounds",
                "control": "passed",
                "observed_exit_code": 86,
                "diagnostic": "out_of_bounds_global_access",
                "diagnostic_log_sha256": sha256_file(diagnostic),
            }
        ),
        encoding="utf-8",
    )
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        (root / f"sanitizer-{tool}-control.log").write_text(
            "ERROR SUMMARY: 0 errors\n", encoding="utf-8"
        )
    _validate_sanitizer_evidence(tmp_path)
    diagnostic.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact diagnostic"):
        _validate_sanitizer_evidence(tmp_path)


def _worker_build_payload() -> dict[str, object]:
    command = ["python3", "-m", "upgrade_guard.worker.build_engine"]
    return {
        "schema_version": "upgradeguard.dev/worker-build/v1",
        "status": "passed",
        "command": command,
        "command_sha256": command_sha256(command),
        "model": {"path": "/corpus/model.onnx", "sha256": "sha256:" + "1" * 64, "bytes": 1},
        "engine": {
            "path": "/output/engine.plan",
            "sha256": "sha256:" + "2" * 64,
            "bytes": 2,
            "device_memory_bytes": 3,
        },
        "memory_diagnostics": {},
        "inspector": {
            "path": "/output/inspector.json",
            "sha256": "sha256:" + "3" * 64,
            "bytes": 4,
        },
        "timing_cache": {
            "path": "/output/timing.cache",
            "input_sha256": None,
            "output_sha256": "sha256:" + "4" * 64,
            "bytes": 5,
        },
        "builder_configuration": {"strongly_typed": "true"},
        "timing_cache_state": "cold",
        "tensorrt_version": "11.2.0",
        "started_unix_seconds": 1.0,
        "ended_unix_seconds": 2.0,
        "duration_seconds": 1.0,
        "strongly_typed": True,
    }


def _stable_build_payload() -> dict[str, object]:
    command = ["python3", "-m", "upgrade_guard.worker.build_engine"]
    return {
        "api_version": "upgradeguard.dev/v1alpha1",
        "kind": "BuildManifest",
        "id": "baseline-fp32-build-0",
        "case_manifest_sha256": "sha256:" + "5" * 64,
        "environment_lock_sha256": "sha256:" + "6" * 64,
        "command": command,
        "command_sha256": command_sha256(command),
        "parser_warnings": [],
        "parser_errors": [],
        "builder_configuration": {"strongly_typed": "true"},
        "plugin_source_sha256": None,
        "plugin_binary": None,
        "plugin_compile_command": None,
        "plugin_build_log": None,
        "timing_cache_mode": "cold",
        "timing_cache": {
            "path": "baseline/fp32/timing.cache",
            "sha256": "sha256:" + "4" * 64,
            "bytes": 5,
            "media_type": "application/octet-stream",
        },
        "started_at": "2026-08-14T00:00:01Z",
        "ended_at": "2026-08-14T00:00:02Z",
        "duration_seconds": 1.0,
        "engine": {
            "path": "baseline/fp32/engine-0.plan",
            "sha256": "sha256:" + "2" * 64,
            "bytes": 2,
            "media_type": "application/octet-stream",
        },
        "engine_inspector": {
            "path": "baseline/fp32/engine-0.inspector.json",
            "sha256": "sha256:" + "3" * 64,
            "bytes": 4,
            "media_type": "application/json",
        },
        "engine_device_memory_bytes": 3,
        "engine_bytes": 2,
        "status": "passed",
        "failure": None,
        "warnings": [],
    }


def test_build_table_parses_raw_and_stable_records_without_glob_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "core-run" / "baseline" / "fp32" / "build-0.json"
    path.parent.mkdir(parents=True)
    value = _worker_build_payload()
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest = path.with_name("build-0.manifest.json")
    manifest.write_text(json.dumps(_stable_build_payload()), encoding="utf-8")

    records = _build_manifest_table(tmp_path)

    assert len(records) == 2
    assert {record["record_kind"] for record in records} == {
        "BuildManifest",
        "WorkerBuildResult",
    }
    assert {record["path"] for record in records} == {
        "core-run/baseline/fp32/build-0.json",
        "core-run/baseline/fp32/build-0.manifest.json",
    }
    value["strongly_typed"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="build record is invalid"):
        _build_manifest_table(tmp_path)
