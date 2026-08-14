"""Validate remote gates and publish a hash-addressed evidence index and report."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.report.model import ReportModel

GENERATED_NAMES = frozenset({"evidence.json", "report-model.json", "report.md", "results.json"})


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


def _validate_spdx(path: Path, *, expected_image: str | None = None) -> None:
    value = _json_object(path)
    packages = value.get("packages")
    if value.get("spdxVersion") != "SPDX-2.3" or not isinstance(packages, list) or not packages:
        raise RuntimeError(f"SBOM is not a populated SPDX 2.3 document: {path}")
    if expected_image is not None and expected_image not in str(value.get("documentComment", "")):
        raise RuntimeError(f"worker SBOM is not bound to its locked image: {path}")


def _validate_pip_audit(path: Path) -> None:
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


def _validate_profile_summary(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "residualRmsNorm" not in text or "TensorRT Release" in text:
        raise RuntimeError(
            f"profile summary lacks the selected kernel or contains a banner: {path}"
        )


def _validate_cuda_benchmark(value: dict[str, Any]) -> None:
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("CUDA benchmark retained no cases")
    for case in cases:
        pairs = case.get("pairs") if isinstance(case, dict) else None
        if not isinstance(pairs, list) or len(pairs) != 20:
            raise RuntimeError("CUDA benchmark requires 20 paired blocks per case")
        expected_orders = [
            "scalar_then_optimized" if index % 2 == 0 else "optimized_then_scalar"
            for index in range(20)
        ]
        observed_orders = [pair.get("order") for pair in pairs if isinstance(pair, dict)]
        if observed_orders != expected_orders:
            raise RuntimeError("CUDA benchmark did not alternate scalar and optimized order")


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
    files = [path for path in state.rglob("*") if path.is_file() and path not in excluded]
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
    *, generated_at: str, matrix: MatrixLock, state: Path, artifact_paths: list[Path]
) -> dict[str, Any]:
    environments = {environment.id: environment for environment in matrix.environments}
    references = [
        {
            "path": path.relative_to(state).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "media_type": "application/json",
        }
        for path in artifact_paths
    ]
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
        "result_count": len(artifact_paths),
        "passed_count": len(artifact_paths),
        "failed_count": 0,
        "unsupported_count": 0,
        "infrastructure_invalid_count": 0,
        "inconclusive_count": 0,
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
    return (
        "# TensorRT UpgradeGuard qualification report\n\n"
        "## Observed facts\n\n"
        f"The source commit is `{payload['source_git_commit']}`.\n"
        f"The selected GPU UUID is `{payload['gpu_uuid']}`.\n"
        f"The environment-lock hash is `{payload['matrix_lock_sha256']}`.\n\n"
        f"{environment_lines}\n\n"
        "| Gate | Status |\n"
        "| --- | --- |\n"
        f"{gate_lines}\n\n"
        "## Core numerical and determinism evidence\n\n"
        f"The core retained `{len(correctness)}` three-way correctness cases with "
        "repetition-level determinism evidence.\n"
        "Exact numerical summaries, output hashes, nonfinite counts, and failure indexes "
        "are in `results.json`.\n\n"
        "## Unprofiled performance and memory evidence\n\n"
        f"The core retained `{len(performance)}` precision-specific paired performance tables "
        f"and `{len(memory)}` precision-specific memory tables.\n"
        "Accepted and rejected raw blocks, intervals, engine bytes, and engine-reported "
        "device-memory values are in `results.json`.\n"
        "Profiled measurements are diagnostic and are excluded from the primary performance "
        "decision.\n\n"
        "## Reduction, sanitizer, and profiling evidence\n\n"
        "The indexed artifacts include clean replay evidence, fault controls, Compute Sanitizer "
        "output, Nsight Systems data, and focused Nsight Compute data.\n"
        "Each retained artifact is content-addressed in `evidence.json`.\n\n"
        "## Policy decision\n\n"
        "Every required gate represented here passed its locked checked-in policy.\n"
        "A failed, inconclusive, infrastructure-invalid, missing, malformed, or tampered gate "
        "prevents this report from being generated.\n\n"
        "## Threats to validity and non-goals\n\n"
        f"{payload['claim_scope']}\n"
        "The measurements cover one selected GPU and the exact locked software stacks.\n"
        "They are not universal TensorRT, CUDA, model-quality, energy, or production-SLO "
        "claims.\n\n"
        "## Reproduction and trust\n\n"
        "Use the verified reproduction bundle and its typed CLI path from an empty directory.\n"
        "Review source hashes before granting source trust.\n"
        "Do not execute unreviewed bundled scripts, engines, containers, or profiler artifacts.\n"
    )


def generate(state: Path, output: Path) -> None:
    state = state.resolve(strict=True)
    output = output.resolve()
    required_json = {
        "core_qualification": state / "core-run" / "qualification-summary.json",
        "plugin_validation": state / "plugin-runs" / "validation.json",
        "mobilenet_validation": state / "mobilenet-runs" / "validation.json",
        "aa_pilot": state / "aa" / "validation.json",
        "cuda_benchmark": state / "plugin" / "candidate" / "plugin-benchmark.json",
        "gpu_faults": state / "faults" / "validation.json",
        "reduction_replay": state / "reductions" / "validation.json",
        "memory_seed": state / "memory-seed" / "validation.json",
    }
    values = {name: _passed(path) for name, path in required_json.items()}
    _validate_cuda_benchmark(values["cuda_benchmark"])
    _validate_clean_replays(values["reduction_replay"])
    sanitizer_path = state / "plugin" / "candidate" / "sanitizer-seed.json"
    sanitizer = _json_object(sanitizer_path)
    if sanitizer.get("expected") != "SANITIZER_FAILURE" or sanitizer.get("control") != "passed":
        raise RuntimeError("sanitizer seed or clean control evidence is invalid")
    for name in ("G1", "G6", "G7"):
        value = _json_object(state / "faults" / f"{name}.json")
        if not value.get("detected") or value.get("control") != "passed":
            raise RuntimeError(f"{name} or its control did not pass")
    matrix_path = state / "matrix.lock.json"
    matrix = MatrixLock.model_validate_json(matrix_path.read_text(encoding="utf-8"))
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise RuntimeError("remote matrix lock self-hash differs")
    source_commit = (state / "source.commit").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or state.name != source_commit:
        raise RuntimeError("state directory and source commit identity differ")

    environment_images = {
        environment.id: environment.worker_image.canonical_reference
        for environment in matrix.environments
    }
    required_files = [
        state / "gpu-preflight.csv",
        state / "plugin" / "candidate" / "residual-rmsnorm-timeline.nsys-rep",
        state / "plugin" / "candidate" / "residual-rmsnorm-timeline.sqlite",
        state / "plugin" / "candidate" / "nsys-kernel-summary.csv",
        state / "plugin" / "candidate" / "residual-rmsnorm-kernel.ncu-rep",
        state / "plugin" / "candidate" / "ncu-kernel-summary.csv",
        state / "plugin" / "baseline" / "build" / "compile_commands.json",
        state / "plugin" / "candidate" / "build" / "compile_commands.json",
        state / "plugin" / "candidate" / "build" / "libupgrade_guard_residual_rmsnorm.so",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required remote artifact is absent or empty: {path}")
    for path in (
        state / "plugin" / "candidate" / "nsys-kernel-summary.csv",
        state / "plugin" / "candidate" / "ncu-kernel-summary.csv",
    ):
        _validate_profile_summary(path)
    sbom_paths = {
        "baseline": state / "sbom" / "baseline.spdx.json",
        "candidate": state / "sbom" / "candidate.spdx.json",
    }
    for name, path in sbom_paths.items():
        _validate_spdx(path, expected_image=environment_images[name])
    _validate_spdx(state / "sbom" / "host.spdx.json")
    _validate_pip_audit(state / "supply-chain" / "pip-audit.json")
    _validate_pip_audit(state / "supply-chain" / "worker-pip-audit.json")

    generated_at = datetime.now(UTC).isoformat()
    gate_status = {name: value["status"] for name, value in values.items()}
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
        "gate_status": gate_status,
        "claim_scope": claim_scope,
    }
    results = {
        "schema_version": "upgradeguard.dev/published-result-table/v1",
        "status": "passed",
        **base_payload,
        "core_qualification": values["core_qualification"],
        "plugin_validation": values["plugin_validation"],
        "mobilenet_validation": values["mobilenet_validation"],
        "aa_pilot": values["aa_pilot"],
        "cuda_benchmark": values["cuda_benchmark"],
        "gpu_faults": values["gpu_faults"],
        "reduction_replay": values["reduction_replay"],
        "memory_seed": values["memory_seed"],
        "sanitizer_seed": sanitizer,
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
            artifact_paths=list(required_json.values()),
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
