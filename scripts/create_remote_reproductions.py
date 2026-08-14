"""Create reduced seeded evidence and a clean source-bearing GPU bundle."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from upgrade_guard.contracts.base import canonical_json_bytes, sha256_bytes
from upgrade_guard.contracts.common import FailureRecord, Phase, PrecisionMode
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode
from upgrade_guard.reduce.session import reduce_failure_directory
from upgrade_guard.reproduce.bundle import BundleExport, export_bundle
from upgrade_guard.reproduce.run import prepare_replay
from upgrade_guard.reproduce.verify import materialize_verified_bundle, verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--plugin-corpus", type=Path, required=True)
    arguments = parser.parse_args()
    state = arguments.state.resolve(strict=True)
    project = arguments.project.resolve(strict=True)
    corpus = arguments.plugin_corpus.resolve(strict=True)
    root = state / "reductions"
    root.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in (state / "faults" / "gpu-fault-samples.jsonl").read_text().splitlines()
        if line
    ]
    if len(records) != 20:
        raise RuntimeError("remote reduction requires 20 seeded GPU observations")
    signature = sha256_bytes(canonical_json_bytes(records))
    numerical = _numerical(root, records, signature)
    performance = _performance(root, records, signature)
    profile = _profile(root, signature)
    matrix = MatrixLock.model_validate_json(
        (state / "matrix.lock.json").read_text(encoding="utf-8")
    )
    bundles = {
        "G2": _g2_bundle(root, state, project, corpus, matrix, signature),
        "G7": _g7_bundle(root, state, project, matrix, signature),
    }
    clean_bundles: dict[str, dict[str, object]] = {}
    for seed, bundle in bundles.items():
        clean = root / f"{seed}-clean-bundle"
        verified = (
            verify_bundle(clean) if clean.exists() else materialize_verified_bundle(bundle, clean)
        )
        plan = prepare_replay(clean, trust_source_code=True, trust_included_engine=False)
        if plan.selected_gpu_uuid != matrix.gpu_uuid or not plan.source_paths:
            raise RuntimeError("clean replay plan does not preserve the locked GPU and sources")
        clean_bundles[seed] = {
            "bundle_manifest_sha256": verified.manifest.manifest_sha256,
            "bundle_id": plan.bundle_id,
            "source_paths": plan.source_paths,
            "clean_directory": str(clean),
        }
    result = {
        "schema_version": "upgradeguard.dev/reduction-replay/v1",
        "status": "prepared",
        "signature_sha256": signature,
        "numerical": numerical,
        "performance": performance,
        "profile": profile,
        "selected_gpu_uuid": plan.selected_gpu_uuid,
        "clean_bundles": clean_bundles,
    }
    (root / "prepared.json").write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request(failure_code: str, signature: str, predicate: dict[str, object]) -> dict[str, object]:
    return {
        "api_version": "upgradeguard.dev/v1alpha1",
        "kind": "ReductionRequest",
        "failure_code": failure_code,
        "signature_sha256": signature,
        "confirmation_count": 2,
        "maximum_trials": 100,
        "maximum_seconds": 120,
        "predicate": predicate,
    }


def _numerical(root: Path, records: list[dict[str, object]], signature: str) -> dict[str, object]:
    source = root / "G2-failure"
    source.mkdir(exist_ok=True)
    reference = np.asarray([record["G2"]["reference"] for record in records], dtype=np.float32)  # type: ignore[index]
    candidate = np.asarray([record["G2"]["observed"] for record in records], dtype=np.float32)  # type: ignore[index]
    np.save(source / "reference.npy", reference, allow_pickle=False)
    np.save(source / "candidate.npy", candidate, allow_pickle=False)
    request = _request(
        "NUMERICAL_REGRESSION",
        signature,
        {
            "kind": "numerical",
            "output_name": "residual_rmsnorm",
            "reference_path": "reference.npy",
            "candidate_path": "candidate.npy",
            "atol": 1e-4,
            "rtol": 1e-3,
        },
    )
    (source / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    destination = root / "G2-reduced"
    result = (
        json.loads((destination / "reduction-result.json").read_text())
        if destination.exists()
        else reduce_failure_directory(source, destination)
    )
    if (destination / "candidate.npy").stat().st_size >= (source / "candidate.npy").stat().st_size:
        raise RuntimeError("G2 numerical reduction did not produce a smaller candidate")
    return result


def _performance(root: Path, records: list[dict[str, object]], signature: str) -> dict[str, object]:
    source = root / "G5-failure"
    source.mkdir(exist_ok=True)
    baseline = [1.0] * len(records)
    candidate = [float(record["G5"]["ratio"]) for record in records]  # type: ignore[index]
    (source / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (source / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    request = _request(
        "PERFORMANCE_REGRESSION",
        signature,
        {
            "kind": "performance",
            "baseline_path": "baseline.json",
            "candidate_path": "candidate.json",
            "allowance": 0.10,
            "bootstrap_seed": 20260813,
            "bootstrap_replicates": 5000,
            "minimum_pairs": 20,
        },
    )
    (source / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    destination = root / "G5-reduced"
    return (
        json.loads((destination / "reduction-result.json").read_text())
        if destination.exists()
        else reduce_failure_directory(source, destination)
    )


def _profile(root: Path, signature: str) -> dict[str, object]:
    source = root / "G7-failure"
    source.mkdir(exist_ok=True)
    request = _request(
        "PROFILE_REJECTED",
        signature,
        {
            "kind": "profile",
            "input_name": "tokens",
            "observed_shape": [9, 8, 256],
            "minimum_shape": [1, 8, 256],
            "maximum_shape": [8, 512, 256],
        },
    )
    (source / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    destination = root / "G7-reduced"
    return (
        json.loads((destination / "reduction-result.json").read_text())
        if destination.exists()
        else reduce_failure_directory(source, destination)
    )


def _g2_bundle(
    root: Path,
    state: Path,
    project: Path,
    corpus: Path,
    matrix: MatrixLock,
    signature: str,
) -> Path:
    bundle = root / "G2-bundle"
    if bundle.exists():
        return bundle
    environment_root = root / "bundle-inputs"
    environment_root.mkdir(exist_ok=True)
    baseline_path = environment_root / "baseline.json"
    candidate_path = environment_root / "candidate.json"
    baseline_path.write_text(matrix.environments[0].model_dump_json(indent=2), encoding="utf-8")
    candidate_path.write_text(matrix.environments[1].model_dump_json(indent=2), encoding="utf-8")
    commands = environment_root / "commands.json"
    build_command = (
        "cmake",
        "-S",
        "/opt/upgrade-guard",
        "-B",
        "/output/build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DUPGRADE_GUARD_BUILD_TESTS=OFF",
        "-DUPGRADE_GUARD_BUILD_FAULTS=ON",
    )
    commands.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/replay-recipe/v1",
                "expected_failure_code": "NUMERICAL_REGRESSION",
                "steps": [
                    {"id": "configure", "command": list(build_command)},
                    {
                        "id": "compile",
                        "command": [
                            "cmake",
                            "--build",
                            "/output/build",
                            "--target",
                            "upgrade_guard_residual_rmsnorm",
                            "upgrade_guard_gpu_faults",
                        ],
                    },
                    {
                        "id": "build-engine",
                        "command": [
                            "python3",
                            "-m",
                            "upgrade_guard.worker.build_engine",
                            "--model",
                            "/corpus/model.onnx",
                            "--profile",
                            "/corpus/profile.json",
                            "--engine",
                            "/output/engine.plan",
                            "--inspector",
                            "/output/inspector.json",
                            "--timing-cache",
                            "/output/timing.cache",
                            "--result",
                            "/output/build-engine.json",
                            "--plugin",
                            "/output/build/libupgrade_guard_residual_rmsnorm.so",
                        ],
                        "result_file": "build-engine.json",
                        "expected_result_status": "passed",
                    },
                    {
                        "id": "clean-control",
                        "command": [
                            "python3",
                            "-m",
                            "upgrade_guard.worker.run_correctness",
                            "--engine",
                            "/output/engine.plan",
                            "--input",
                            "x=/corpus/inputs/000-x.npy",
                            "--input",
                            "residual=/corpus/inputs/001-residual.npy",
                            "--input",
                            "gamma=/corpus/inputs/002-gamma.npy",
                            "--output",
                            "/output/control-outputs",
                            "--result",
                            "/output/control.json",
                            "--repetitions",
                            "20",
                            "--plugin",
                            "/output/build/libupgrade_guard_residual_rmsnorm.so",
                        ],
                        "result_file": "control.json",
                        "expected_result_status": "passed",
                    },
                    {
                        "id": "seeded-failure",
                        "command": ["/output/build/upgrade_guard_gpu_faults"],
                        "stdout_json_equals": {"G2.detected": True, "G2.control": "passed"},
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_paths = [
        project / "CMakeLists.txt",
        *(project / "cmake").glob("*.cmake"),
        *(project / "cpp" / "kernels").glob("*"),
        *(project / "cpp" / "plugin").glob("*"),
        *(project / "cpp" / "faults").glob("*"),
        *(project / "src" / "upgrade_guard").rglob("*.py"),
    ]
    source_files = {
        path.relative_to(project).as_posix(): path
        for path in sorted(source_paths)
        if path.is_file()
    }
    expected = FailureRecord(
        code=FailureCode.NUMERICAL_REGRESSION,
        phase=Phase.CORRECTNESS,
        environment_id="candidate",
        model_id="residual-rmsnorm-fp32",
        precision=PrecisionMode.FP32,
        shape_id="tail-random-h259",
        input_fixture_id="tail-random-h259",
        output_name="residual_rmsnorm",
        gate="candidate_to_reference",
        observed="residual omitted by quarantined G2 kernel",
        threshold="absolute error exceeds 0.1",
        evidence=(),
    )
    profile = environment_root / "plugin-profile.json"
    profile.write_text(
        json.dumps(
            {
                "x": {"min": [1, 1, 7], "opt": [2, 17, 256], "max": [8, 512, 259]},
                "residual": {
                    "min": [1, 1, 7],
                    "opt": [2, 17, 256],
                    "max": [8, 512, 259],
                },
                "gamma": {"min": [7], "opt": [256], "max": [259]},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    export_bundle(
        BundleExport(
            id=f"G2-{signature.removeprefix('sha256:')[:12]}",
            created_at=datetime.now(UTC),
            baseline_environment=baseline_path,
            candidate_environment=candidate_path,
            qualification=state / "full.yaml",
            model=corpus / "residual-rmsnorm-fp32.onnx",
            inputs=(
                corpus / "fp32" / "tail-random-h259" / "x.npy",
                corpus / "fp32" / "tail-random-h259" / "residual.npy",
                corpus / "fp32" / "tail-random-h259" / "gamma.npy",
            ),
            expected_failure=expected,
            extra_files={"commands/replay.json": commands, "profile.json": profile},
            source_files=source_files,
            worker_image_manifest_digest=matrix.environments[1].worker_image.manifest_digest,
            selected_gpu_uuid=matrix.gpu_uuid,
            source_build_command=build_command,
        ),
        bundle,
    )
    return bundle


def _g7_bundle(
    root: Path,
    state: Path,
    project: Path,
    matrix: MatrixLock,
    signature: str,
) -> Path:
    bundle = root / "G7-bundle"
    if bundle.exists():
        return bundle
    environment_root = root / "bundle-inputs"
    environment_root.mkdir(exist_ok=True)
    baseline_path = environment_root / "baseline.json"
    candidate_path = environment_root / "candidate.json"
    baseline_path.write_text(matrix.environments[0].model_dump_json(indent=2), encoding="utf-8")
    candidate_path.write_text(matrix.environments[1].model_dump_json(indent=2), encoding="utf-8")
    commands = environment_root / "G7-commands.json"
    profile = environment_root / "transformer-profile.json"
    profile.write_text(
        json.dumps(
            {
                "tokens": {
                    "min": [1, 8, 256],
                    "opt": [4, 128, 256],
                    "max": [8, 512, 256],
                },
                "mask": {
                    "min": [1, 1, 1, 8],
                    "opt": [4, 1, 1, 128],
                    "max": [8, 1, 1, 512],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    build_command = (
        "python3",
        "-m",
        "upgrade_guard.worker.build_engine",
        "--model",
        "/corpus/model.onnx",
        "--profile",
        "/corpus/profile.json",
        "--engine",
        "/output/engine.plan",
        "--inspector",
        "/output/inspector.json",
        "--timing-cache",
        "/output/timing.cache",
        "--result",
        "/output/build-engine.json",
    )
    commands.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/replay-recipe/v1",
                "expected_failure_code": "PROFILE_REJECTED",
                "steps": [
                    {
                        "id": "build-engine",
                        "command": list(build_command),
                        "result_file": "build-engine.json",
                        "expected_result_status": "passed",
                    },
                    {
                        "id": "clean-control",
                        "command": [
                            "python3",
                            "-m",
                            "upgrade_guard.worker.run_correctness",
                            "--engine",
                            "/output/engine.plan",
                            "--input",
                            "tokens=/corpus/control/tokens.npy",
                            "--input",
                            "mask=/corpus/control/mask.npy",
                            "--output",
                            "/output/control-outputs",
                            "--result",
                            "/output/control.json",
                            "--repetitions",
                            "20",
                        ],
                        "result_file": "control.json",
                        "expected_result_status": "passed",
                    },
                    {
                        "id": "seeded-failure",
                        "command": [
                            "python3",
                            "-m",
                            "upgrade_guard.worker.run_correctness",
                            "--engine",
                            "/output/engine.plan",
                            "--input",
                            "tokens=/corpus/inputs/000-tokens.npy",
                            "--input",
                            "mask=/corpus/inputs/001-mask.npy",
                            "--output",
                            "/output/failure-outputs",
                            "--result",
                            "/output/failure.json",
                            "--repetitions",
                            "20",
                        ],
                        "accepted_returncodes": [1],
                        "result_file": "failure.json",
                        "expected_result_status": "failed",
                        "result_message_contains": "input shape was rejected",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_files = {
        path.relative_to(project).as_posix(): path
        for path in sorted((project / "src" / "upgrade_guard").rglob("*.py"))
        if path.is_file()
    }
    expected = FailureRecord(
        code=FailureCode.PROFILE_REJECTED,
        phase=Phase.CORRECTNESS,
        environment_id="candidate",
        model_id="tiny-transformer-fp32",
        precision=PrecisionMode.FP32,
        shape_id="b9_s8",
        input_fixture_id="b9_s8",
        output_name=None,
        gate="optimization_profile",
        observed="tokens shape [9, 8, 256] exceeds the locked profile",
        threshold="maximum tokens shape [8, 512, 256]",
        evidence=(),
    )
    core = project / ".upgrade-guard" / "corpora" / "v1-core"
    export_bundle(
        BundleExport(
            id=f"G7-{signature.removeprefix('sha256:')[:12]}",
            created_at=datetime.now(UTC),
            baseline_environment=baseline_path,
            candidate_environment=candidate_path,
            qualification=state / "full.yaml",
            model=core / "models" / "tiny-transformer-fp32.onnx",
            inputs=(state / "faults" / "g7" / "tokens.npy", state / "faults" / "g7" / "mask.npy"),
            expected_failure=expected,
            extra_files={
                "commands/replay.json": commands,
                "profile.json": profile,
                "control/tokens.npy": core
                / "inputs"
                / "tiny-transformer-fp32"
                / "b8_s512"
                / "tokens.npy",
                "control/mask.npy": core
                / "inputs"
                / "tiny-transformer-fp32"
                / "b8_s512"
                / "mask.npy",
            },
            source_files=source_files,
            worker_image_manifest_digest=matrix.environments[1].worker_image.manifest_digest,
            selected_gpu_uuid=matrix.gpu_uuid,
            source_build_command=build_command,
        ),
        bundle,
    )
    return bundle


if __name__ == "__main__":
    main()
