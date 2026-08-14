"""Validate remote gates and publish a hash-addressed evidence index and report."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.qualification_state import DOMAIN_OUTCOME_JSON, _dependencies_for, verify_marker
from scripts.validate_profiler_outputs import validate_summary
from scripts.validate_seeded_gpu_faults import classify_seed_record
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import (
    canonical_json_bytes,
    model_sha256,
    sha256_bytes,
    sha256_file,
)
from upgrade_guard.contracts.build import BuildManifest, WorkerBuildResult
from upgrade_guard.contracts.case import CaseManifest, adapt_case_manifest
from upgrade_guard.contracts.common import ArtifactReference, ResultStatus
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.contracts.extended import ExtendedInvocationManifest
from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock
from upgrade_guard.contracts.results import RunResult
from upgrade_guard.gates import POST_PUBLICATION_ARTIFACTS, expected_publication_steps
from upgrade_guard.report.model import ReportModel

GENERATED_NAMES = frozenset({"evidence.json", "report-model.json", "report.md", "results.json"})
BENCHMARK_ORDER_SEED = 20260813
BENCHMARK_ORDER_SCHEDULE = (
    "scalar_then_optimized",
    "optimized_then_scalar",
    "optimized_then_scalar",
    "scalar_then_optimized",
    "scalar_then_optimized",
    "optimized_then_scalar",
    "scalar_then_optimized",
    "optimized_then_scalar",
    "optimized_then_scalar",
    "scalar_then_optimized",
    "optimized_then_scalar",
    "scalar_then_optimized",
    "scalar_then_optimized",
    "optimized_then_scalar",
    "optimized_then_scalar",
    "scalar_then_optimized",
    "scalar_then_optimized",
    "optimized_then_scalar",
    "scalar_then_optimized",
    "optimized_then_scalar",
)
AUDITED_DEPENDENCY_SCOPES = (
    "uv.lock Python dependencies",
    "hash-locked Python dependencies added by containers/Dockerfile.worker",
    "hash-locked Python dependencies added by containers/Dockerfile.reference",
)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required evidence is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"required evidence is not a JSON object: {path}")
    return value


def _passed(path: Path) -> dict[str, Any]:
    value = _json_object(path)
    if value.get("status") != "passed":
        raise RuntimeError(f"required evidence did not pass: {path}")
    return value


def _source_commit(state: Path) -> str:
    """Validate the retained source identity without constraining the output path name."""

    try:
        value = (state / "source.commit").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError("source commit evidence is unavailable") from error
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError("source commit evidence is invalid")
    return value


def _validate_spdx(
    path: Path,
    *,
    expected_image: str | None = None,
    expected_lock_sha256: str | None = None,
) -> None:
    value = _json_object(path)
    packages = value.get("packages")
    if value.get("spdxVersion") != "SPDX-2.3" or not isinstance(packages, list) or not packages:
        raise RuntimeError(f"SBOM is not a populated SPDX 2.3 document: {path}")
    if expected_image is not None and expected_image not in str(value.get("documentComment", "")):
        raise RuntimeError(f"worker SBOM is not bound to its locked image: {path}")
    if expected_lock_sha256 is not None and expected_lock_sha256 not in str(
        value.get("documentComment", "")
    ):
        raise RuntimeError(f"host SBOM is not bound to its exact lock: {path}")


def _validate_pip_audit(path: Path) -> int:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"pip-audit output is invalid: {path}") from error
    dependencies = value.get("dependencies") if isinstance(value, dict) else value
    if not isinstance(dependencies, list):
        raise RuntimeError(f"pip-audit output has an unsupported schema: {path}")
    vulnerable = [
        item
        for item in dependencies
        if isinstance(item, dict) and isinstance(item.get("vulns"), list) and item["vulns"]
    ]
    if vulnerable:
        raise RuntimeError(f"dependency vulnerabilities require triage before release: {path}")
    return len(dependencies)


def _validate_dependency_triage(path: Path) -> dict[str, Any]:
    value = _passed(path)
    if tuple(value.get("audited_scopes", ())) != AUDITED_DEPENDENCY_SCOPES:
        raise RuntimeError("dependency triage does not cover every hash-locked Python scope")
    claim = value.get("claim")
    if not isinstance(claim, str) or "do not claim vulnerability-free images" not in claim:
        raise RuntimeError("dependency triage does not preserve the bounded image claim")
    return value


def _validate_step_markers(
    state: Path,
    project: Path,
    *,
    source_commit: str,
    gpu_uuid: str,
) -> dict[str, dict[str, str]]:
    """Require every full-mode pre-publication marker and retain its identity."""

    memo: dict[str, bool] = {}
    gates: dict[str, dict[str, str]] = {}
    for step in _dependencies_for("final-evidence", "full", state=state):
        if not verify_marker(
            state,
            project,
            step,
            source_commit,
            gpu_uuid,
            "full",
            _memo=memo,
        ):
            raise RuntimeError(f"required qualification marker is invalid: {step}")
        marker = state / "done" / f"{step}.json"
        marker_value = _json_object(marker)
        outcome = marker_value.get("outcome")
        if outcome not in {"passed", "failed"}:
            raise RuntimeError(f"qualification marker has no terminal outcome: {step}")
        gates[step] = {"status": outcome, "marker_sha256": sha256_file(marker)}
    return gates


def _validate_sanitizer_evidence(state: Path) -> dict[str, Any]:
    seed_path = state / "sanitizers" / "sanitizer-seed.json"
    seed = _json_object(seed_path)
    diagnostic = state / "sanitizers" / "sanitizer-tail-oob.log"
    if (
        seed.get("classification") != "SANITIZER_FAILURE"
        or seed.get("mechanism") != "vectorized_tail_out_of_bounds"
        or seed.get("control") != "passed"
        or seed.get("observed_exit_code") != 86
        or seed.get("diagnostic") != "out_of_bounds_global_access"
        or seed.get("diagnostic_log_sha256") != sha256_file(diagnostic)
    ):
        raise RuntimeError("sanitizer seed evidence differs from its exact diagnostic")
    controls: dict[str, dict[str, int | str]] = {}
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        path = state / "sanitizers" / f"sanitizer-{tool}-control.log"
        text = path.read_text(encoding="utf-8")
        if not re.search(r"ERROR SUMMARY: 0 errors?", text):
            raise RuntimeError(f"sanitizer clean control did not retain zero errors: {tool}")
        controls[tool] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"seed": seed, "controls": controls}


def _build_manifest_table(state: Path) -> list[dict[str, Any]]:
    raw_patterns = (
        "target-readiness/**/build.json",
        "core-run/*/*/build-*.json",
        "plugin-runs/*/*/build.json",
        "mobilenet-runs/*/build.json",
        "memory-seed/*/build-*.json",
    )
    raw_paths = {
        path
        for pattern in raw_patterns
        for path in state.glob(pattern)
        if not path.name.endswith(".manifest.json")
    }
    stable_patterns = (
        "target-readiness/**/build.manifest.json",
        "core-run/*/*/build-*.manifest.json",
        "plugin-runs/*/*/*/build.manifest.json",
        "mobilenet-runs/*/*/build.manifest.json",
        "memory-seed/*/build-*.manifest.json",
    )
    stable_paths = {path for pattern in stable_patterns for path in state.glob(pattern)}
    records = sorted(
        (("WorkerBuildResult", path) for path in raw_paths),
        key=lambda item: item[1],
    ) + sorted(
        (("BuildManifest", path) for path in stable_paths),
        key=lambda item: item[1],
    )
    if not records:
        raise RuntimeError("qualification retained no engine build records")
    table: list[dict[str, Any]] = []
    for record_kind, path in records:
        try:
            if record_kind == "WorkerBuildResult":
                worker = WorkerBuildResult.model_validate(_json_object(path))
                if worker.status != "passed" or worker.engine is None:
                    raise ValueError("worker build did not pass with an engine")
                if (
                    worker.command_sha256 != command_sha256(worker.command)
                    or worker.strongly_typed is not True
                    or worker.builder_configuration.get("strongly_typed") != "true"
                ):
                    raise ValueError("worker build command or strong-typing evidence differs")
                row = {
                    "record_kind": record_kind,
                    "command": list(worker.command),
                    "command_sha256": worker.command_sha256,
                    "model": worker.model.model_dump(mode="json") if worker.model else None,
                    "engine": worker.engine.model_dump(mode="json"),
                    "strongly_typed": True,
                    "timing_cache_state": worker.timing_cache_state,
                    "builder_warnings": [
                        item.model_dump(mode="json") for item in worker.builder_warnings
                    ],
                    "memory_diagnostics": worker.memory_diagnostics,
                    "inspector": (
                        worker.inspector.model_dump(mode="json") if worker.inspector else None
                    ),
                }
            else:
                manifest = BuildManifest.model_validate(_json_object(path))
                if manifest.status is not ResultStatus.PASSED or manifest.engine is None:
                    raise ValueError("stable build manifest did not pass with an engine")
                if (
                    manifest.command_sha256 != command_sha256(manifest.command)
                    or manifest.builder_configuration.get("strongly_typed") != "true"
                ):
                    raise ValueError("stable build command or strong-typing evidence differs")
                row = {
                    "record_kind": record_kind,
                    "command": list(manifest.command),
                    "command_sha256": manifest.command_sha256,
                    "model": None,
                    "engine": manifest.engine.model_dump(mode="json"),
                    "strongly_typed": True,
                    "timing_cache_state": manifest.timing_cache_mode,
                    "builder_warnings": list(manifest.warnings),
                    "memory_diagnostics": {
                        "serialized_engine_bytes": manifest.engine_bytes,
                        "engine_device_memory_bytes": manifest.engine_device_memory_bytes,
                    },
                    "inspector": (
                        manifest.engine_inspector.model_dump(mode="json")
                        if manifest.engine_inspector
                        else None
                    ),
                }
        except ValueError as error:
            raise RuntimeError(f"engine build record is invalid: {path}") from error
        table.append(
            {
                "path": path.relative_to(state).as_posix(),
                "sha256": sha256_file(path),
                **row,
            }
        )
    return table


def _validate_extended_typed_chains(
    state: Path,
    validation: dict[str, Any],
    *,
    suite: str,
    matrix: MatrixLock,
    specification_sha256: str,
    source_commit: str,
    corpus_lock_sha256: str,
) -> dict[str, Any]:
    """Require every retained extended case to name one valid stable chain."""

    runs = state / f"{suite}-runs"
    invocation_value = validation.get("invocation_manifest")
    if not isinstance(invocation_value, dict):
        raise RuntimeError(f"{suite} validation has no typed invocation artifact")
    invocation_path = _declared_artifact(runs, invocation_value)
    invocation = ExtendedInvocationManifest.model_validate(_json_object(invocation_path))
    if (
        invocation.suite != suite
        or invocation.computed_sha256() != invocation.manifest_sha256
        or validation.get("invocation_manifest_sha256") != invocation.manifest_sha256
        or invocation.matrix_lock_sha256 != matrix.lock_sha256
        or invocation.specification_sha256 != specification_sha256
        or invocation.source_git_commit != source_commit
        or invocation.corpus_lock_sha256 != corpus_lock_sha256
    ):
        raise RuntimeError(f"{suite} invocation manifest identity differs")
    for plugin in invocation.plugin_builds:
        for artifact in (
            *plugin.source_inventory,
            plugin.binary,
            plugin.compile_commands,
            plugin.build_log,
        ):
            _declared_artifact(runs, artifact.model_dump(mode="json"))
        if plugin.source_inventory_sha256 != sha256_bytes(
            canonical_json_bytes([item.model_dump(mode="json") for item in plugin.source_inventory])
        ):
            raise RuntimeError("plugin source inventory aggregate differs")
    cases = validation.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(f"{suite} validation retained no extended cases")
    recorded: set[Path] = {invocation_path}
    chain_count = 0
    for case in cases:
        stable = case.get("stable_artifacts") if isinstance(case, dict) else None
        if not isinstance(stable, dict) or set(stable) != {"baseline", "candidate"}:
            raise RuntimeError(f"{suite} case has no exact baseline/candidate stable chain")
        for environment_id, chain in stable.items():
            if not isinstance(chain, dict):
                raise RuntimeError(f"{suite} stable chain is malformed")
            case_path = _declared_artifact(runs, chain.get("case_manifest"))
            build_path = _declared_artifact(runs, chain.get("build_manifest"))
            run_path = _declared_artifact(runs, chain.get("run_result"))
            manifest = adapt_case_manifest(CaseManifest.model_validate(_json_object(case_path)))
            build = BuildManifest.model_validate(_json_object(build_path))
            run = RunResult.model_validate(_json_object(run_path))
            artifacts: tuple[ArtifactReference | None, ...] = (
                build.engine,
                build.engine_inspector,
                build.timing_cache,
                build.plugin_binary,
                build.plugin_build_log,
                *run.output_artifacts,
                *run.logs,
                *run.diagnostics,
            )
            for retained_artifact in artifacts:
                if retained_artifact is not None:
                    _declared_artifact(runs, retained_artifact.model_dump(mode="json"))
            if (
                chain.get("case_manifest_sha256") != manifest.manifest_sha256
                or build.case_manifest_sha256 != manifest.manifest_sha256
                or run.case_manifest_sha256 != manifest.manifest_sha256
                or run.build_manifest_sha256 != sha256_file(build_path)
                or build.environment_lock_sha256 != matrix.lock_sha256
                or run.environment_lock_sha256 != matrix.lock_sha256
                or run.hardware.environment_lock_sha256 != matrix.lock_sha256
                or run.hardware_sha256 != model_sha256(run.hardware)
                or run.status is not ResultStatus.PASSED
                or build.status is not ResultStatus.PASSED
                or environment_id not in run.id
            ):
                raise RuntimeError(f"{suite} stable artifact chain identity differs")
            recorded.update((case_path, build_path, run_path))
            chain_count += 1
    observed = {
        path
        for pattern in (
            "*/**/case-manifest.json",
            "*/**/build.manifest.json",
            "*/**/run-result.json",
        )
        for path in runs.glob(pattern)
        if path.is_file()
    }
    if observed != recorded - {invocation_path}:
        raise RuntimeError(f"{suite} retained unindexed or missing stable artifacts")
    return {
        "invocation_manifest": invocation_value,
        "invocation_manifest_sha256": invocation.manifest_sha256,
        "chain_count": chain_count,
        "stable_artifact_count": len(recorded),
    }


def _declared_artifact(root: Path, value: object) -> Path:
    if not isinstance(value, dict):
        raise RuntimeError("extended stable artifact declaration is malformed")
    relative = value.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("extended stable artifact declaration is malformed")
    path = root / relative
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or path.is_symlink() or not resolved.is_file():
        raise RuntimeError("extended stable artifact escaped its run root")
    if (
        value.get("sha256") != sha256_file(resolved)
        or value.get("bytes") != resolved.stat().st_size
    ):
        raise RuntimeError("extended stable artifact identity differs")
    return resolved


def _validate_cuda_benchmark(value: dict[str, Any]) -> None:
    if value.get("order_seed") != BENCHMARK_ORDER_SEED:
        raise RuntimeError("CUDA benchmark order seed differs from the locked policy")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("CUDA benchmark retained no cases")
    for case in cases:
        pairs = case.get("pairs") if isinstance(case, dict) else None
        if not isinstance(pairs, list) or len(pairs) != 20:
            raise RuntimeError("CUDA benchmark requires 20 paired blocks per case")
        observed_orders = [pair.get("order") for pair in pairs if isinstance(pair, dict)]
        if observed_orders != list(BENCHMARK_ORDER_SCHEDULE):
            raise RuntimeError("CUDA benchmark did not use the locked balanced order schedule")


def _tactic_selection_table(value: dict[str, Any]) -> list[dict[str, Any]]:
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("plugin validation retained no tactic-bearing cases")
    selections: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("workers"), dict):
            raise RuntimeError("plugin tactic case evidence is invalid")
        for environment, worker in sorted(case["workers"].items()):
            tactic = worker.get("tactic_diagnostic") if isinstance(worker, dict) else None
            if not isinstance(tactic, dict):
                raise RuntimeError("plugin worker lacks selected-tactic evidence")
            selections.append(
                {
                    "environment": environment,
                    "precision": case.get("precision"),
                    "case": case.get("case"),
                    **tactic,
                }
            )
    return selections


def _validate_clean_replays(value: dict[str, Any]) -> None:
    prepared = value.get("clean_bundles")
    replays = value.get("clean_replays")
    expected = {
        "G2": "NUMERICAL_REGRESSION",
        "G5": "PERFORMANCE_REGRESSION",
        "G7": "PROFILE_REJECTED",
    }
    if not isinstance(prepared, dict) or not isinstance(replays, dict):
        raise RuntimeError("reduction evidence lacks prepared bundles or clean CLI replays")
    if set(prepared) != {"G2", "G7"} or set(replays) != set(expected):
        raise RuntimeError("reduction evidence must retain exactly G2, G5, and G7 replays")
    for seed, failure_code in expected.items():
        replay = replays[seed]
        if not isinstance(replay, dict):
            raise RuntimeError(f"{seed} clean replay evidence is malformed")
        declared_code = replay.get("expected_failure_code")
        observed_code = replay.get("observed_failure_code")
        if (
            replay.get("status") != "passed"
            or not isinstance(declared_code, str)
            or not isinstance(observed_code, str)
            or declared_code != failure_code
            or observed_code != declared_code
        ):
            raise RuntimeError(f"{seed} did not clean-replay its expected failure")
        if seed == "G5":
            if (
                replay.get("fresh_directory") is not True
                or not isinstance(replay.get("reduced_pairs_sha256"), str)
                or not replay.get("observations")
            ):
                raise RuntimeError("G5 did not retain a fresh-directory reduced-pair replay")
            continue
        bundle = prepared[seed]
        if (
            not isinstance(bundle, dict)
            or replay.get("bundle_manifest_sha256") != bundle.get("bundle_manifest_sha256")
            or not replay.get("step_results")
        ):
            raise RuntimeError(f"{seed} clean replay differs from its bundle")


def _inventory(state: Path, output: Path) -> dict[str, dict[str, int | str]]:
    excluded = {state / name for name in GENERATED_NAMES}
    excluded.add(output)
    excluded.update(state / relative for relative in POST_PUBLICATION_ARTIFACTS)
    files: list[Path] = []
    for current_root, directory_names, file_names in os.walk(state, followlinks=False):
        current = Path(current_root)
        if current == state:
            directory_names[:] = [
                name for name in directory_names if name not in {"stale", "diagnostics"}
            ]
        for name in directory_names:
            path = current / name
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise RuntimeError(f"evidence directory is unavailable: {path}") from error
            if not stat.S_ISDIR(mode):
                raise RuntimeError(f"evidence directory is unsafe: {path}")
        for name in file_names:
            path = current / name
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise RuntimeError(f"evidence artifact is unavailable: {path}") from error
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"evidence artifact is unsafe: {path}")
            if path not in excluded:
                files.append(path)
    if not files:
        raise RuntimeError("remote qualification retained no evidence artifacts")
    artifacts: dict[str, dict[str, int | str]] = {
        path.relative_to(state).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(files)
    }
    for relative, identity in artifacts.items():
        path = state / relative
        if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"evidence artifact changed while indexing: {path}")
    return artifacts


def _report_model(
    *,
    generated_at: str,
    matrix: MatrixLock,
    state: Path,
    gate_evidence: dict[str, dict[str, str]],
    source_commit: str,
    reference_environment_lock_sha256: str,
) -> dict[str, Any]:
    environments = {environment.id: environment for environment in matrix.environments}
    references = [
        {
            "path": f"done/{name}.json",
            "sha256": evidence["marker_sha256"],
            "bytes": (state / "done" / f"{name}.json").stat().st_size,
            "media_type": "application/json",
        }
        for name, evidence in sorted(gate_evidence.items())
    ]
    statuses = [evidence["status"] for evidence in gate_evidence.values()]
    return {
        "api_version": "upgradeguard.dev/report/v1",
        "title": "TensorRT UpgradeGuard qualification",
        "generated_at": generated_at,
        "status": "passed",
        "baseline_environment_id": (
            "baseline" if "baseline" in environments else matrix.environments[0].id
        ),
        "candidate_environment_id": (
            "candidate" if "candidate" in environments else matrix.environments[1].id
        ),
        "stack_attribution": (
            "Observed changes belong to the compared locked stacks. "
            "They are not attributed to TensorRT alone without a smaller controlled experiment."
        ),
        "result_count": len(statuses),
        "passed_count": statuses.count("passed"),
        "failed_count": statuses.count("failed"),
        "unsupported_count": statuses.count("unsupported"),
        "infrastructure_invalid_count": statuses.count("infrastructure_invalid"),
        "inconclusive_count": statuses.count("inconclusive"),
        "failures": [],
        "evidence": references,
        "warnings": [
            "Serialized engines, containers, bundled source, and profiler reports remain "
            "trust boundaries."
        ],
        "publication_complete": True,
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference_environment_lock_sha256,
        "environment_images": {
            environment.id: environment.worker_image.canonical_reference
            for environment in matrix.environments
        },
        "acceptance_gates": {
            name: evidence["status"] for name, evidence in sorted(gate_evidence.items())
        },
        "corpus_provenance": [
            {
                "path": "corpora.json",
                "sha256": sha256_file(state / "corpora.json"),
                "bytes": (state / "corpora.json").stat().st_size,
                "media_type": "application/json",
            }
        ],
        "results_artifact": {
            "path": "results.json",
            "sha256": sha256_file(state / "results.json"),
            "bytes": (state / "results.json").stat().st_size,
            "media_type": "application/json",
        },
        "measured_sections": {
            "numerical_and_determinism": "results.json#/core_qualification/cases",
            "performance": "results.json#/core_qualification/cases",
            "memory": "results.json#/core_qualification/cases",
            "plugin_and_tactics": "results.json#/plugin_validation",
            "mobilenet": "results.json#/mobilenet_validation",
            "faults_and_reduction": "results.json#/reduction_replay",
            "sanitizers_and_profiles": "results.json#/profile_validation",
        },
        "reproduction_commands": [
            ["bash", "scripts/run_cuda_pm_qualification.sh"],
            [
                "upgrade-guard",
                "qualify",
                "qualification/full.yaml",
                "--project-root",
                ".",
                "--out",
                ".upgrade-guard/cuda-pm/manual-run",
            ],
        ],
        "methodology": [
            "Frozen inputs and independent CPU references are compared three ways.",
            "Primary performance uses unprofiled paired blocks and retained hardware validity.",
            "Correctness, determinism, performance, memory, sanitizers, and profiles "
            "are separate gates.",
        ],
        "limitations": [
            "Results apply only to the selected GPU and exact locked complete stacks.",
            "Profiled results are diagnostic and excluded from the primary timing decision.",
            "Preinstalled proprietary and operating-system packages are inventoried "
            "but not claimed vulnerability-free.",
        ],
    }


def _failed_publication(
    *,
    state: Path,
    output: Path,
    matrix: MatrixLock,
    reference_environment: ReferenceEnvironmentLock,
    source_commit: str,
    gate_evidence: dict[str, dict[str, str]],
) -> None:
    """Publish a truthful terminal failure without claiming skipped gates passed."""

    failed_steps = [
        name for name, evidence in gate_evidence.items() if evidence["status"] == "failed"
    ]
    if len(failed_steps) != 1 or failed_steps[0] not in DOMAIN_OUTCOME_JSON:
        raise RuntimeError("failed publication requires exactly one typed qualification failure")
    failed_step = failed_steps[0]
    gate_sequence = expected_publication_steps("failed", failure_step=failed_step)
    if tuple(gate_evidence) != gate_sequence:
        raise RuntimeError("failed publication gate sequence differs from the package contract")
    failure_path = state / DOMAIN_OUTCOME_JSON[failed_step]
    failure_value = _json_object(failure_path)
    from upgrade_guard.reduce.public_failure import validate_public_failure_disposition

    disposition = validate_public_failure_disposition(state / "public-failure", state=state)
    if disposition.source_step != failed_step:
        raise RuntimeError("public failure disposition belongs to another domain step")
    failure_records = [item.failure.model_dump(mode="json") for item in disposition.items]
    if failed_step == "core-qualification":
        raw_codes = failure_value.get("failure_codes")
        if (
            failure_value.get("status") != "failed"
            or not isinstance(raw_codes, list)
            or not raw_codes
            or not all(isinstance(item, str) for item in raw_codes)
        ):
            raise RuntimeError("core failed publication lacks exact failure codes")
        failure_codes = list(dict.fromkeys(raw_codes))
    else:
        from scripts.three_way_validation import ThreeWayValidationResult

        extended = ThreeWayValidationResult.model_validate(failure_value)
        if extended.status is not ResultStatus.FAILED or extended.failure is None:
            raise RuntimeError("extended failed publication lacks a typed failure record")
        failure_codes = [extended.failure.code.value]
    if set(failure_codes) != {item["code"] for item in failure_records}:
        raise RuntimeError("failure disposition and source failure codes differ")
    generated_at_value = datetime.now(UTC)
    generated_at = generated_at_value.isoformat()
    environment_images = {
        environment.id: environment.worker_image.canonical_reference
        for environment in matrix.environments
    }
    failure_artifact = {
        "path": failure_path.relative_to(state).as_posix(),
        "sha256": sha256_file(failure_path),
        "bytes": failure_path.stat().st_size,
        "media_type": "application/json",
    }
    disposition_path = state / "public-failure" / "disposition.json"
    disposition_artifact = {
        "path": disposition_path.relative_to(state).as_posix(),
        "sha256": sha256_file(disposition_path),
        "bytes": disposition_path.stat().st_size,
        "media_type": "application/json",
    }
    reduced_items = [item for item in disposition.items if item.disposition == "reduced_replayed"]
    disposition_items = [item.model_dump(mode="json") for item in disposition.items]
    reduction = {
        "status": "passed" if reduced_items else "not_applicable",
        "reason": (
            None
            if reduced_items
            else "; ".join(
                sorted(
                    {
                        item.reason
                        for item in disposition.items
                        if item.disposition == "not_applicable" and item.reason is not None
                    }
                )
            )
        ),
        "disposition": disposition_artifact,
        "disposition_sha256": disposition.disposition_sha256,
        "items": disposition_items,
    }
    reproduction = {
        "status": "passed" if reduced_items else "not_applicable",
        "reason": reduction["reason"],
        "disposition": disposition_artifact,
        "disposition_sha256": disposition.disposition_sha256,
        "items": disposition_items,
    }
    limitations = [
        "Qualification stopped after the first completed domain failure.",
        "Gates after the failed step were not executed and are not represented as passing.",
        (
            "Every supported failure was confirmed, reduced, bundled, and clean-replayed."
            if reduced_items
            else "The retained failure classes have explicit typed not-applicable dispositions."
        ),
        "Results apply only to the selected GPU and exact locked complete stacks.",
    ]
    results = {
        "schema_version": "upgradeguard.dev/published-result-table/v1",
        "status": "failed",
        "generated_at": generated_at,
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference_environment.lock_sha256,
        "environment_images": environment_images,
        "gate_sequence": list(gate_sequence),
        "gate_status": {
            name: evidence["status"] for name, evidence in sorted(gate_evidence.items())
        },
        "gate_evidence": gate_evidence,
        "core_qualification": _json_object(state / "core-run" / "qualification-summary.json"),
        "failure_step": failed_step,
        "failure_codes": failure_codes,
        "failures": failure_records,
        "failure_evidence": {"artifact": failure_artifact, "value": failure_value},
        "reduction": reduction,
        "reproduction": reproduction,
        "limitations": limitations,
    }
    results_path = state / "results.json"
    results_path.write_text(
        json.dumps(results, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    references = [
        ArtifactReference(
            path=f"done/{name}.json",
            sha256=evidence["marker_sha256"],
            bytes=(state / "done" / f"{name}.json").stat().st_size,
            media_type="application/json",
        )
        for name, evidence in sorted(gate_evidence.items())
    ]
    references.append(ArtifactReference.model_validate(disposition_artifact))
    corpus_reference = ArtifactReference(
        path="corpora.json",
        sha256=sha256_file(state / "corpora.json"),
        bytes=(state / "corpora.json").stat().st_size,
        media_type="application/json",
    )
    results_reference = ArtifactReference(
        path="results.json",
        sha256=sha256_file(results_path),
        bytes=results_path.stat().st_size,
        media_type="application/json",
    )
    report_model = ReportModel(
        api_version="upgradeguard.dev/report/v1",
        title="TensorRT UpgradeGuard qualification failure",
        generated_at=generated_at_value,
        status=ResultStatus.FAILED,
        baseline_environment_id=matrix.environments[0].id,
        candidate_environment_id=matrix.environments[1].id,
        stack_attribution=(
            "The failure belongs to the compared locked stacks and is not attributed to "
            "TensorRT alone without a smaller controlled experiment."
        ),
        result_count=len(gate_evidence),
        passed_count=sum(item["status"] == "passed" for item in gate_evidence.values()),
        failed_count=1,
        unsupported_count=0,
        infrastructure_invalid_count=0,
        inconclusive_count=0,
        failures=tuple(item.failure for item in disposition.items),
        evidence=tuple(references),
        warnings=(
            "Serialized engines, containers, bundled source, and profiler reports remain "
            "trust boundaries.",
            "Unexecuted gates make no acceptance claim.",
        ),
        publication_complete=True,
        source_git_commit=source_commit,
        gpu_uuid=matrix.gpu_uuid,
        matrix_lock_sha256=matrix.lock_sha256,
        reference_environment_lock_sha256=reference_environment.lock_sha256,
        environment_images=environment_images,
        acceptance_gates={
            name: ResultStatus(str(evidence["status"]))
            for name, evidence in sorted(gate_evidence.items())
        },
        corpus_provenance=(corpus_reference,),
        results_artifact=results_reference,
        measured_sections={"terminal_failure": "results.json#/failure_evidence"},
        reproduction_commands=(("bash", "scripts/run_cuda_pm_qualification.sh"),),
        methodology=(
            "Completed steps are accepted only through hash-addressed markers.",
            "A typed domain failure terminates unrelated qualification work and remains resumable.",
        ),
        limitations=tuple(limitations),
    )
    report_model_path = state / "report-model.json"
    report_model_path.write_text(report_model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report_path = state / "report.md"
    gate_rows = "\n".join(
        f"| `{name}` | `{evidence['status']}` |" for name, evidence in sorted(gate_evidence.items())
    )
    report_path.write_text(
        "# TensorRT UpgradeGuard qualification failure\n\n"
        f"The completed qualification decision is `failed` at `{failed_step}`.\n"
        f"The exact failure codes are `{', '.join(failure_codes)}`.\n\n"
        "## Completed gates\n\n"
        "| Gate | Outcome |\n"
        "| --- | --- |\n"
        f"{gate_rows}\n\n"
        "## Failure evidence\n\n"
        f"The typed terminal artifact is `{failure_artifact['path']}` with SHA-256 "
        f"`{failure_artifact['sha256']}`.\n\n"
        "## Reduction and reproduction\n\n"
        f"The typed disposition is `{disposition_artifact['path']}` with SHA-256 "
        f"`{disposition_artifact['sha256']}`.\n"
        + (
            "Supported failures were clean-replayed from source-bearing bundles.\n\n"
            if reduced_items
            else "All retained failure classes are explicitly not applicable to an authored "
            "V1 reducer.\n\n"
        )
        + "## Limitations\n\n"
        + "".join(f"- {item}\n" for item in limitations),
        encoding="utf-8",
    )
    artifacts = _inventory(state, output)
    payload = {
        "schema_version": "upgradeguard.dev/remote-evidence/v1",
        "status": "failed",
        "generated_at": generated_at,
        "source_git_commit": source_commit,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference_environment.lock_sha256,
        "failure_step": failed_step,
        "failure_codes": failure_codes,
        "failures": failure_records,
        "core_qualification": results["core_qualification"],
        "environment_images": environment_images,
        "gate_sequence": list(gate_sequence),
        "reduction": reduction,
        "reproduction": reproduction,
        "gate_status": results["gate_status"],
        "gate_evidence": gate_evidence,
        "artifacts": artifacts,
        "generated_artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (results_path, report_model_path, report_path)
        },
        "limitations": limitations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return


def _markdown_report(payload: dict[str, Any], values: dict[str, dict[str, Any]]) -> str:
    """Render the successful publication from its validated result tables."""

    gate_lines = "\n".join(
        f"| {name} | {status} |" for name, status in sorted(payload["gate_status"].items())
    )
    environment_lines = "\n".join(
        f"- {name}: {image}" for name, image in payload["environment_images"].items()
    )
    core = values["core_qualification"]
    correctness = [case for case in core.get("cases", []) if "shape_id" in case]
    memory = [case for case in core.get("cases", []) if "memory" in case]
    performance = [case for case in core.get("cases", []) if "performance" in case]
    numerical_lines = []
    for case in correctness:
        for gate in (
            "baseline_to_reference",
            "candidate_to_reference",
            "candidate_to_baseline",
        ):
            result = case[gate]
            numerical_lines.append(
                f"| {case['precision']} | {case['shape_id']} | {gate} | "
                f"{result['elementwise_passed']} | {result['maximum_absolute_error']} | "
                f"{result['candidate_nonfinite_count']} |"
            )
    performance_lines = []
    for case in performance:
        result = case["performance"]
        for shape, estimate in sorted(result["shapes"].items()):
            performance_lines.append(
                f"| {case['precision']} | {shape} | {estimate['accepted_pairs']} | "
                f"{estimate['point']} | {estimate['one_sided_lower']} | "
                f"{estimate['one_sided_upper']} | {estimate['outcome']} |"
            )
        estimate = result["aggregate"]
        performance_lines.append(
            f"| {case['precision']} | weighted | {estimate['accepted_pairs']} | "
            f"{estimate['point']} | {estimate['one_sided_lower']} | "
            f"{estimate['one_sided_upper']} | {estimate['outcome']} |"
        )
    memory_lines = []
    for case in memory:
        for source in ("engine_bytes", "device_memory"):
            result = case["memory"][source]
            memory_lines.append(
                f"| {case['precision']} | {source} | {result['baseline_bytes']} | "
                f"{list(result['candidate_bytes'])} | {result['allowance_bytes']} | "
                f"{result['outcome']} |"
            )
    tactic_lines = "\n".join(
        f"| {item['environment']} | {item['precision']} | {item['case']} | "
        f"{item['selected_tactic']} | {item['engine_sha256']} | "
        f"{item['rows']}x{item['hidden']} |"
        for item in payload["tactic_selections"]
    )
    optimized_candidate = all(
        item["selected_tactic"] == "kVECTORIZED_WARP"
        for item in payload["tactic_selections"]
        if item["environment"] == "candidate"
    )
    toolkit = payload["toolkit_version_observation"]
    return (
        "# TensorRT UpgradeGuard qualification report\n\n"
        "## Observed facts\n\n"
        f"The source commit is {payload['source_git_commit']}.\n"
        f"The selected GPU is {payload['gpu_name']} with UUID {payload['gpu_uuid']}.\n"
        f"The observed driver is {payload['driver']}.\n"
        f"The environment-lock hash is {payload['matrix_lock_sha256']}.\n"
        "The independent reference-environment lock is "
        f"{payload['reference_environment_lock_sha256']}.\n\n"
        f"Host toolkit-version provenance is {toolkit['status']}.\n"
        "GPU injection capability passed through both exact immutable worker probes.\n\n"
        f"{environment_lines}\n\n"
        "## Acceptance gates\n\n"
        "| Gate | Status |\n"
        "| --- | --- |\n"
        f"{gate_lines}\n\n"
        "## Core numerical and determinism evidence\n\n"
        "| Precision | Shape | Comparison | Passed | Maximum absolute error | "
        "Candidate nonfinite |\n"
        "| --- | --- | --- | --- | ---: | ---: |\n"
        f"{chr(10).join(numerical_lines)}\n\n"
        f"The core retained {len(correctness)} three-way correctness cases and all "
        "repetition-level hashes in results.json.\n\n"
        "## Unprofiled performance and memory evidence\n\n"
        "| Precision | Shape | Accepted pairs | Point | Lower 95% one-sided | "
        "Upper 95% one-sided | Outcome |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | --- |\n"
        f"{chr(10).join(performance_lines)}\n\n"
        "| Precision | Memory source | Baseline bytes | Candidate bytes | Allowance bytes | "
        "Outcome |\n"
        "| --- | --- | ---: | --- | ---: | --- |\n"
        f"{chr(10).join(memory_lines)}\n\n"
        "Accepted and rejected raw blocks, hardware observations, workload weights, builder "
        "memory, execution-context memory, coarse process GPU memory, and host RSS are in "
        "results.json. Profiled measurements are diagnostic and excluded from the primary "
        "performance decision.\n\n"
        "## Build, tactic, reduction, sanitizer, and profiling evidence\n\n"
        f"results.json contains {payload['build_record_count']} exact typed build records: "
        f"{payload['stable_build_manifest_count']} stable BuildManifest contracts and "
        f"{payload['worker_build_result_count']} raw WorkerBuildResult contracts.\n\n"
        "| Environment | Precision | Case | Selected tactic | Engine SHA-256 | Shape |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"{tactic_lines}\n\n"
        "The G2 and G7 bundles rebuild their engine from hash-verified source in clean output "
        "directories through the public CLI. The G4 defect retains its sanitizer diagnostic, "
        "and focused Nsight artifacts remain diagnostic. Each retained artifact is "
        "content-addressed in evidence.json.\n\n"
        "## Policy decision\n\n"
        "Every full-mode acceptance marker and semantic validator passed its checked-in policy.\n"
        "The decision applies to the complete locked stacks, not TensorRT in isolation.\n\n"
        "## Inferences\n\n"
        + (
            "The candidate engine selected the vectorized tactic in every retained plugin case, "
            "and the focused-kernel profile is consistent with its measured behavior.\n"
            if optimized_candidate
            else "The retained tactic table is descriptive and does not establish universal "
            "selection of the vectorized tactic.\n"
        )
        + "This is a scoped inference, not a universal causal claim.\n\n"
        "## Threats to validity and non-goals\n\n"
        f"{payload['claim_scope']}\n"
        "The measurements cover one selected GPU and the exact locked software stacks. "
        "They are not universal TensorRT, CUDA, model-quality, energy, or production-SLO "
        "claims. Preinstalled proprietary and operating-system packages are inventoried but "
        "are not claimed vulnerability-free.\n\n"
        "## Reproduction and trust\n\n"
        "Run bash scripts/run_cuda_pm_qualification.sh to resume the full qualification. "
        "Review source hashes before granting source trust, and do not execute unreviewed "
        "bundled scripts, engines, containers, or profiler artifacts.\n"
    )


def generate(state: Path, output: Path) -> None:
    state = state.resolve(strict=True)
    output = output.resolve()
    project = Path(__file__).resolve().parents[1]
    matrix_path = state / "matrix.lock.json"
    matrix = MatrixLock.model_validate_json(matrix_path.read_text(encoding="utf-8"))
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise RuntimeError("remote matrix lock self-hash differs")
    reference_environment = ReferenceEnvironmentLock.model_validate_json(
        (state / "reference-environment.lock.json").read_text(encoding="utf-8")
    )
    if reference_environment.computed_sha256() != reference_environment.lock_sha256:
        raise RuntimeError("reference environment lock self-hash differs")
    source_commit = _source_commit(state)
    gate_evidence = _validate_step_markers(
        state,
        project,
        source_commit=source_commit,
        gpu_uuid=matrix.gpu_uuid,
    )
    if any(evidence["status"] == "failed" for evidence in gate_evidence.values()):
        _failed_publication(
            state=state,
            output=output,
            matrix=matrix,
            reference_environment=reference_environment,
            source_commit=source_commit,
            gate_evidence=gate_evidence,
        )
        return
    required_json = {
        "target_readiness": state / "target-readiness" / "validation.json",
        "profiler_preflight": state / "profiler-preflight" / "validation.json",
        "core_qualification": state / "core-run" / "qualification-summary.json",
        "plugin_validation": state / "plugin-runs" / "validation.json",
        "mobilenet_validation": state / "mobilenet-runs" / "validation.json",
        "aa_pilot": state / "aa" / "validation.json",
        "cuda_benchmark": state / "plugin-benchmark" / "plugin-benchmark.json",
        "gpu_faults": state / "gpu-faults" / "validation.json",
        "reduction_replay": state / "reductions" / "validation.json",
        "memory_seed": state / "memory-seed" / "validation.json",
        "profile_validation": state / "profiles" / "validation.json",
        "sbom_validation": state / "sbom" / "validation.json",
    }
    values = {name: _passed(path) for name, path in required_json.items()}
    tactic_selections = _tactic_selection_table(values["plugin_validation"])
    _validate_cuda_benchmark(values["cuda_benchmark"])
    _validate_clean_replays(values["reduction_replay"])
    sanitizer = _validate_sanitizer_evidence(state)
    for name in ("G1", "G6", "G7"):
        value = _json_object(state / "gpu-faults" / f"{name}.json")
        if not value.get("detected") or value.get("control") != "passed":
            raise RuntimeError(f"{name} or its control did not pass")
        classification = classify_seed_record(name, value)
        if value.get("classification", classification.value) != classification.value:
            raise RuntimeError(f"{name} retained a classification different from production")
    environment_images = {
        environment.id: environment.worker_image.canonical_reference
        for environment in matrix.environments
    }
    required_files = [
        state / "gpu-preflight.csv",
        state / "profiles" / "residual-rmsnorm-timeline.nsys-rep",
        state / "profiles" / "residual-rmsnorm-timeline.sqlite",
        state / "profiles" / "nsys-kernel-summary.csv",
        state / "profiles" / "residual-rmsnorm-kernel.ncu-rep",
        state / "profiles" / "ncu-kernel-summary.csv",
        state / "plugin-build" / "baseline" / "build" / "compile_commands.json",
        state / "plugin-build" / "candidate" / "build" / "compile_commands.json",
        state / "plugin-build" / "candidate" / "build" / "libupgrade_guard_residual_rmsnorm.so",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required remote artifact is absent or empty: {path}")
    validate_summary(state / "profiles" / "nsys-kernel-summary.csv", summary_kind="nsys")
    validate_summary(state / "profiles" / "ncu-kernel-summary.csv", summary_kind="ncu")
    sbom_paths = {
        "baseline": state / "sbom" / "baseline.spdx.json",
        "candidate": state / "sbom" / "candidate.spdx.json",
    }
    for name, path in sbom_paths.items():
        _validate_spdx(path, expected_image=environment_images[name])
    _validate_spdx(
        state / "sbom" / "host.spdx.json",
        expected_lock_sha256=sha256_file(project / "uv.lock"),
    )
    dependency_audit_counts = {
        "host_locked_python_dependencies": _validate_pip_audit(
            state / "supply-chain" / "pip-audit.json"
        ),
        "worker_added_python_dependencies": _validate_pip_audit(
            state / "supply-chain" / "worker-pip-audit.json"
        ),
        "reference_added_python_dependencies": _validate_pip_audit(
            state / "supply-chain" / "reference-pip-audit.json"
        ),
    }
    dependency_triage = _validate_dependency_triage(state / "supply-chain" / "triage.json")
    corpus_provenance = _json_object(state / "corpora.json")
    if corpus_provenance.get("reference_environment_sha256") != reference_environment.lock_sha256:
        raise RuntimeError("corpus and reference environment identities differ")
    corpus_entries = corpus_provenance.get("corpora")
    if not isinstance(corpus_entries, dict):
        raise RuntimeError("corpus provenance inventory is invalid")
    extended_corpus_locks: dict[str, str] = {}
    for suite in ("plugin", "mobilenet"):
        entry = corpus_entries.get(suite)
        relative = entry.get("lock") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise RuntimeError(f"{suite} corpus provenance lock is invalid")
        lock_path = (project / relative).resolve(strict=True)
        if not lock_path.is_relative_to(project) or lock_path.is_symlink():
            raise RuntimeError(f"{suite} corpus provenance lock is unsafe")
        extended_corpus_locks[suite] = sha256_file(lock_path)
    extended_typed_chains = {
        suite: _validate_extended_typed_chains(
            state,
            values[f"{suite}_validation"],
            suite=suite,
            matrix=matrix,
            specification_sha256=sha256_file(state / "full.yaml"),
            source_commit=source_commit,
            corpus_lock_sha256=extended_corpus_locks[suite],
        )
        for suite in ("plugin", "mobilenet")
    }
    build_manifests = _build_manifest_table(state)
    stable_build_manifest_count = sum(
        item["record_kind"] == "BuildManifest" for item in build_manifests
    )
    worker_build_result_count = sum(
        item["record_kind"] == "WorkerBuildResult" for item in build_manifests
    )
    corpus_artifacts: dict[str, dict[str, int | str]] = {}
    for kind, entry in sorted(corpus_provenance["corpora"].items()):
        if not isinstance(entry, dict):
            raise RuntimeError(f"corpus provenance entry is invalid: {kind}")
        for label, relative in (
            ("lock", entry.get("lock")),
            ("materializer", f"{entry.get('root')}/materializer.json"),
        ):
            if not isinstance(relative, str):
                raise RuntimeError(f"corpus provenance path is invalid: {kind}/{label}")
            path = (project / relative).resolve(strict=True)
            if not path.is_relative_to(project) or path.is_symlink() or not path.is_file():
                raise RuntimeError(f"corpus provenance path is unsafe: {kind}/{label}")
            corpus_artifacts[f"{kind}/{label}"] = {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    worker_images = _json_object(state / "worker-images.json")
    qualification_policy = yaml.safe_load((state / "full.yaml").read_text(encoding="utf-8"))
    if not isinstance(qualification_policy, dict):
        raise RuntimeError("qualification policy is not a mapping")

    generated_at = datetime.now(UTC).isoformat()
    gate_sequence = expected_publication_steps("passed")
    if tuple(gate_evidence) != gate_sequence:
        raise RuntimeError("passing publication gate sequence differs from the package contract")
    gate_status = {name: evidence["status"] for name, evidence in gate_evidence.items()}
    claim_scope = (
        "Results apply to the complete locked stacks and selected GPU. "
        "They do not isolate TensorRT as the sole cause of a difference."
    )
    base_payload = {
        "generated_at": generated_at,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference_environment.lock_sha256,
        "source_git_commit": source_commit,
        "environment_images": environment_images,
        "gpu_name": matrix.environments[0].probe.gpu.name,
        "driver": matrix.environments[0].probe.observed_driver,
        "toolkit_version_observation": matrix.environments[
            0
        ].host.nvidia_container_toolkit_version.model_dump(mode="json"),
        "gate_status": gate_status,
        "gate_sequence": list(gate_sequence),
        "gate_evidence": gate_evidence,
        "claim_scope": claim_scope,
        "build_record_count": len(build_manifests),
        "stable_build_manifest_count": stable_build_manifest_count,
        "worker_build_result_count": worker_build_result_count,
        "tactic_selections": tactic_selections,
    }
    results = {
        "schema_version": "upgradeguard.dev/published-result-table/v1",
        "status": "passed",
        "failure_codes": [],
        **base_payload,
        "environment_lock": matrix.model_dump(mode="json"),
        "reference_environment_lock": reference_environment.model_dump(mode="json"),
        "worker_images": worker_images,
        "corpus_provenance": corpus_provenance,
        "corpus_artifacts": corpus_artifacts,
        "qualification_policy": qualification_policy,
        "build_manifests": build_manifests,
        "extended_typed_chains": extended_typed_chains,
        "tactic_selections": tactic_selections,
        "target_readiness": values["target_readiness"],
        "profiler_preflight": values["profiler_preflight"],
        "core_qualification": values["core_qualification"],
        "plugin_validation": values["plugin_validation"],
        "mobilenet_validation": values["mobilenet_validation"],
        "aa_pilot": values["aa_pilot"],
        "cuda_benchmark": values["cuda_benchmark"],
        "gpu_faults": values["gpu_faults"],
        "reduction_replay": values["reduction_replay"],
        "memory_seed": values["memory_seed"],
        "profile_validation": values["profile_validation"],
        "sanitizer_evidence": sanitizer,
        "sbom_validation": values["sbom_validation"],
        "profile_artifacts": {
            path.relative_to(state).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in required_files
            if path.is_relative_to(state / "profiles")
        },
        "dependency_audit_counts": dependency_audit_counts,
        "dependency_triage": dependency_triage,
        "reproduction": {
            "qualification_command": ["bash", "scripts/run_cuda_pm_qualification.sh"],
            "bundles": values["reduction_replay"]["clean_bundles"],
        },
    }
    results_path = state / "results.json"
    results_path.write_text(
        json.dumps(results, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_model_path = state / "report-model.json"
    report_model = ReportModel.model_validate(
        _report_model(
            generated_at=generated_at,
            matrix=matrix,
            state=state,
            gate_evidence=gate_evidence,
            source_commit=source_commit,
            reference_environment_lock_sha256=reference_environment.lock_sha256,
        )
    )
    report_model_path.write_text(
        report_model.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = state / "report.md"
    report_path.write_text(_markdown_report(base_payload, values), encoding="utf-8")
    artifacts = _inventory(state, output)
    payload = {
        "schema_version": "upgradeguard.dev/remote-evidence/v1",
        "status": "passed",
        **base_payload,
        "artifacts": artifacts,
        "generated_artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (results_path, report_model_path, report_path)
        },
        "external_corpus_artifacts": corpus_artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.state, arguments.output)


if __name__ == "__main__":
    main()
