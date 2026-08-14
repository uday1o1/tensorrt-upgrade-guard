"""Validate remote gates and publish a hash-addressed evidence index and report."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.qualification_state import MODE_STEPS, verify_marker
from scripts.validate_profiler_outputs import validate_summary
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.environment import MatrixLock
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
    for step in MODE_STEPS["full"]:
        if step in {"final-evidence", "terminal-cleanup"}:
            continue
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
        gates[step] = {"status": "passed", "marker_sha256": sha256_file(marker)}
    return gates


def _validate_sanitizer_evidence(state: Path) -> dict[str, Any]:
    seed_path = state / "sanitizers" / "sanitizer-seed.json"
    seed = _json_object(seed_path)
    diagnostic = state / "sanitizers" / "sanitizer-tail-oob.log"
    if (
        seed.get("expected") != "SANITIZER_FAILURE"
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
    patterns = (
        "target-readiness/**/build.json",
        "core-run/*/*/build-*.json",
        "plugin-runs/*/*/build.json",
        "mobilenet-runs/*/build.json",
        "memory-seed/*/build-*.json",
    )
    paths = sorted({path for pattern in patterns for path in state.glob(pattern)})
    if not paths:
        raise RuntimeError("qualification retained no engine build manifests")
    table: list[dict[str, Any]] = []
    for path in paths:
        value = _passed(path)
        engine = value.get("engine")
        command = value.get("command")
        if not isinstance(engine, dict) or not engine.get("sha256"):
            raise RuntimeError(f"engine build manifest lacks engine identity: {path}")
        if (
            not isinstance(command, list)
            or not command
            or not isinstance(value.get("command_sha256"), str)
            or value.get("command_sha256") != command_sha256(command)
            or value.get("strongly_typed") is not True
        ):
            raise RuntimeError(f"engine build manifest lacks exact strong-typed command: {path}")
        table.append(
            {
                "path": path.relative_to(state).as_posix(),
                "sha256": sha256_file(path),
                "command": command,
                "command_sha256": value.get("command_sha256"),
                "model": value.get("model"),
                "engine": engine,
                "strongly_typed": value.get("strongly_typed"),
                "timing_cache_state": value.get("timing_cache_state"),
                "builder_warnings": value.get("builder_warnings", []),
                "memory_diagnostics": value.get("memory_diagnostics"),
                "inspector": value.get("inspector"),
            }
        )
    return table


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


def _validate_clean_replays(value: dict[str, Any]) -> None:
    prepared = value.get("clean_bundles")
    replays = value.get("clean_replays")
    expected = {"G2": "NUMERICAL_REGRESSION", "G7": "PROFILE_REJECTED"}
    if not isinstance(prepared, dict) or not isinstance(replays, dict):
        raise RuntimeError("reduction evidence lacks prepared bundles or clean CLI replays")
    if set(prepared) != set(expected) or set(replays) != set(expected):
        raise RuntimeError("reduction evidence must retain exactly G2 and G7 clean replays")
    for seed, failure_code in expected.items():
        bundle = prepared[seed]
        replay = replays[seed]
        if not isinstance(bundle, dict) or not isinstance(replay, dict):
            raise RuntimeError(f"{seed} clean replay evidence is malformed")
        if (
            replay.get("status") != "passed"
            or replay.get("expected_failure_code") != failure_code
            or replay.get("bundle_manifest_sha256") != bundle.get("bundle_manifest_sha256")
            or not replay.get("step_results")
        ):
            raise RuntimeError(f"{seed} did not clean-replay its expected failure")


def _inventory(state: Path, output: Path) -> dict[str, dict[str, int | str]]:
    excluded = {state / name for name in GENERATED_NAMES}
    excluded.add(output)
    excluded.add(state / "logs" / "final-evidence.log")
    files = [
        path
        for path in state.rglob("*")
        if path.is_file()
        and path not in excluded
        and path.relative_to(state).parts[0] not in {"stale", "diagnostics"}
    ]
    if not files:
        raise RuntimeError("remote qualification retained no evidence artifacts")
    artifacts = {
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
    }


def _markdown_report(payload: dict[str, Any], values: dict[str, dict[str, Any]]) -> str:
    gate_lines = "\n".join(
        f"| `{name}` | `{status}` |" for name, status in sorted(payload["gate_status"].items())
    )
    environment_lines = "\n".join(
        f"- `{name}`: `{image}`" for name, image in payload["environment_images"].items()
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
                f"| `{case['precision']}` | `{case['shape_id']}` | `{gate}` | "
                f"`{result['elementwise_passed']}` | `{result['maximum_absolute_error']}` | "
                f"`{result['candidate_nonfinite_count']}` |"
            )
    performance_lines = []
    for case in performance:
        result = case["performance"]
        for shape, estimate in sorted(result["shapes"].items()):
            performance_lines.append(
                f"| `{case['precision']}` | `{shape}` | `{estimate['accepted_pairs']}` | "
                f"`{estimate['point']}` | `{estimate['one_sided_lower']}` | "
                f"`{estimate['one_sided_upper']}` | `{estimate['outcome']}` |"
            )
        estimate = result["aggregate"]
        performance_lines.append(
            f"| `{case['precision']}` | `weighted` | `{estimate['accepted_pairs']}` | "
            f"`{estimate['point']}` | `{estimate['one_sided_lower']}` | "
            f"`{estimate['one_sided_upper']}` | `{estimate['outcome']}` |"
        )
    memory_lines = []
    for case in memory:
        for source in ("engine_bytes", "device_memory"):
            result = case["memory"][source]
            memory_lines.append(
                f"| `{case['precision']}` | `{source}` | `{result['baseline_bytes']}` | "
                f"`{list(result['candidate_bytes'])}` | `{result['allowance_bytes']}` | "
                f"`{result['outcome']}` |"
            )
    toolkit = payload["toolkit_version_observation"]
    numerical_table = "\n".join(numerical_lines)
    performance_table = "\n".join(performance_lines)
    memory_table = "\n".join(memory_lines)
    return (
        "# TensorRT UpgradeGuard qualification report\n\n"
        "## Observed facts\n\n"
        f"The source commit is `{payload['source_git_commit']}`.\n"
        f"The selected GPU is `{payload['gpu_name']}` with UUID `{payload['gpu_uuid']}`.\n"
        f"The observed driver is `{payload['driver']}`.\n"
        f"The environment-lock hash is `{payload['matrix_lock_sha256']}`.\n\n"
        f"Host toolkit-version provenance is `{toolkit['status']}`.\n"
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
        f"{numerical_table}\n\n"
        f"The core retained `{len(correctness)}` three-way correctness cases and all "
        "repetition-level hashes in `results.json`.\n\n"
        "## Unprofiled performance and memory evidence\n\n"
        "| Precision | Shape | Accepted pairs | Point | Lower 95% one-sided | "
        "Upper 95% one-sided | Outcome |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | --- |\n"
        f"{performance_table}\n\n"
        "| Precision | Memory source | Baseline bytes | Candidate bytes | Allowance bytes | "
        "Outcome |\n"
        "| --- | --- | ---: | --- | ---: | --- |\n"
        f"{memory_table}\n\n"
        "Accepted and rejected raw blocks, hardware observations, workload weights, builder "
        "memory, execution-context memory, coarse process GPU memory, and host RSS are in "
        "`results.json`.\n"
        "Profiled measurements are diagnostic and are excluded from the primary performance "
        "decision.\n\n"
        "## Build, tactic, reduction, sanitizer, and profiling evidence\n\n"
        f"`results.json` contains `{payload['build_manifest_count']}` exact strongly typed build "
        "manifests with commands, warnings, cold or warm timing-cache state, engine identity, "
        "memory diagnostics, and inspector references.\n"
        "The G2 and G7 bundles rebuild their engine from hash-verified source in clean output "
        "directories through the public CLI.\n"
        "The quarantined G4 defect fails with an out-of-bounds global-access diagnostic while "
        "memcheck, racecheck, initcheck, and synccheck controls report zero errors.\n"
        "The focused Nsight Systems and Nsight Compute artifacts name the optimized "
        "`residualRmsNormFloat4` tactic and remain diagnostic.\n"
        "Each retained artifact is content-addressed in `evidence.json`.\n\n"
        "## Policy decision\n\n"
        "Every full-mode acceptance marker and semantic validator passed its checked-in policy.\n"
        "A failed, inconclusive, infrastructure-invalid, missing, malformed, or tampered gate "
        "prevents this report from being generated.\n\n"
        "The decision applies to the complete locked stacks, not TensorRT in isolation.\n\n"
        "## Inferences\n\n"
        "The profile evidence is consistent with the optimized CUDA tactic explaining the "
        "measured focused-kernel behavior.\n"
        "This is an inference from the selected-kernel profile, not a universal causal claim.\n\n"
        "## Threats to validity and non-goals\n\n"
        f"{payload['claim_scope']}\n"
        "The measurements cover one selected GPU and the exact locked software stacks.\n"
        "They are not universal TensorRT, CUDA, model-quality, energy, or production-SLO "
        "claims.\n"
        "Dependency-clean claims cover the host lock and Python packages added by the worker "
        "Dockerfile only.\n"
        "Preinstalled NGC Python packages, Debian packages, and proprietary NVIDIA packages "
        "are inventoried in SPDX documents but are not claimed vulnerability-free.\n\n"
        "## Reproduction and trust\n\n"
        "Run `bash scripts/run_cuda_pm_qualification.sh` to resume the full qualification.\n"
        "Use each verified reproduction bundle and its typed CLI path from an empty directory.\n"
        "Review source hashes before granting source trust.\n"
        "Do not execute unreviewed bundled scripts, engines, containers, or profiler artifacts.\n"
    )


def generate(state: Path, output: Path) -> None:
    state = state.resolve(strict=True)
    output = output.resolve()
    project = Path(__file__).resolve().parents[1]
    matrix_path = state / "matrix.lock.json"
    matrix = MatrixLock.model_validate_json(matrix_path.read_text(encoding="utf-8"))
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise RuntimeError("remote matrix lock self-hash differs")
    source_commit = (state / "source.commit").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or state.name != source_commit:
        raise RuntimeError("state directory and source commit identity differ")
    gate_evidence = _validate_step_markers(
        state,
        project,
        source_commit=source_commit,
        gpu_uuid=matrix.gpu_uuid,
    )
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
    _validate_cuda_benchmark(values["cuda_benchmark"])
    _validate_clean_replays(values["reduction_replay"])
    sanitizer = _validate_sanitizer_evidence(state)
    for name in ("G1", "G6", "G7"):
        value = _json_object(state / "gpu-faults" / f"{name}.json")
        if not value.get("detected") or value.get("control") != "passed":
            raise RuntimeError(f"{name} or its control did not pass")
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
    }
    dependency_triage = _passed(state / "supply-chain" / "triage.json")
    build_manifests = _build_manifest_table(state)
    corpus_provenance = _json_object(state / "corpora.json")
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
    gate_status = {name: evidence["status"] for name, evidence in gate_evidence.items()}
    claim_scope = (
        "Results apply to the complete locked stacks and selected GPU. "
        "They do not isolate TensorRT as the sole cause of a difference."
    )
    base_payload = {
        "generated_at": generated_at,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "source_git_commit": source_commit,
        "environment_images": environment_images,
        "gpu_name": matrix.environments[0].probe.gpu.name,
        "driver": matrix.environments[0].probe.observed_driver,
        "toolkit_version_observation": matrix.environments[
            0
        ].host.nvidia_container_toolkit_version.model_dump(mode="json"),
        "gate_status": gate_status,
        "gate_evidence": gate_evidence,
        "claim_scope": claim_scope,
        "build_manifest_count": len(build_manifests),
    }
    results = {
        "schema_version": "upgradeguard.dev/published-result-table/v1",
        "status": "passed",
        **base_payload,
        "environment_lock": matrix.model_dump(mode="json"),
        "worker_images": worker_images,
        "corpus_provenance": corpus_provenance,
        "corpus_artifacts": corpus_artifacts,
        "qualification_policy": qualification_policy,
        "build_manifests": build_manifests,
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
