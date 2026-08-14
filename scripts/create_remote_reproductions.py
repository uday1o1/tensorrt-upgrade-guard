"""Create reduced seeded evidence and a clean source-bearing GPU bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from upgrade_guard.compare.performance import AcceptedPair, GateOutcome, paired_ratio_gate
from upgrade_guard.contracts.base import canonical_json_bytes, sha256_bytes, sha256_file
from upgrade_guard.contracts.bundle import canonical_cmake_cuda_architecture
from upgrade_guard.contracts.common import FailureRecord, Phase, PrecisionMode
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.contracts.qualification import QualificationSpec, ReductionBudget
from upgrade_guard.errors import FailureCode
from upgrade_guard.reduce.candidate import G2ReductionCandidate, G7ReductionCandidate
from upgrade_guard.reduce.general import ReductionLimits
from upgrade_guard.reduce.remote import run_remote_reductions
from upgrade_guard.reduce.session import reduce_failure_directory
from upgrade_guard.reproduce.bundle import BundleExport, export_bundle
from upgrade_guard.reproduce.run import prepare_replay
from upgrade_guard.reproduce.verify import materialize_verified_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--core-corpus", type=Path, required=True)
    parser.add_argument("--plugin-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    state = arguments.state.resolve(strict=True)
    project = arguments.project.resolve(strict=True)
    core_corpus = arguments.core_corpus.resolve(strict=True)
    plugin_corpus = arguments.plugin_corpus.resolve(strict=True)
    output = (arguments.output or state / "reductions" / "prepared").resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("refusing to overwrite prepared reduction evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepared.", dir=output.parent))
    try:
        records = [
            json.loads(line)
            for line in (state / "gpu-faults" / "gpu-fault-samples.jsonl").read_text().splitlines()
            if line
        ]
        if len(records) != 24:
            raise RuntimeError("remote reduction requires 24 seeded GPU observations")
        signature = sha256_bytes(canonical_json_bytes(records))
        budget = _locked_reduction_budget(state / "full.yaml")
        limits = ReductionLimits(
            maximum_trials=budget.maximum_trials,
            maximum_seconds=float(budget.maximum_seconds),
            confirmation_count=budget.confirmation_count,
        )
        performance = _performance(staging, records, signature, budget)
        matrix = MatrixLock.model_validate_json(
            (state / "matrix.lock.json").read_text(encoding="utf-8")
        )
        reduction_work = output.parent / "candidate-reduction-work"
        remote = run_remote_reductions(
            project=project,
            state=state,
            core_corpus=core_corpus,
            plugin_corpus=plugin_corpus,
            matrix=matrix,
            signature_sha256=signature,
            output=reduction_work,
            limits=limits,
        )
        shutil.copytree(reduction_work, staging / "candidate-reductions")
        bundles = {
            "G2": _g2_bundle(
                staging,
                state,
                project,
                plugin_corpus,
                matrix,
                signature,
                remote.g2,
            ),
            "G7": _g7_bundle(
                staging,
                state,
                project,
                core_corpus,
                matrix,
                signature,
                remote.g7,
            ),
        }
        clean_bundles: dict[str, dict[str, object]] = {}
        original_gpu_uuid: str | None = None
        for seed, bundle in bundles.items():
            clean = staging / f"{seed}-clean-bundle"
            verified = materialize_verified_bundle(bundle, clean)
            plan = prepare_replay(clean, trust_source_code=True, trust_included_engine=False)
            if plan.original_gpu_uuid != matrix.gpu_uuid or not plan.source_paths:
                raise RuntimeError("clean replay plan does not preserve GPU provenance and sources")
            if plan.selected_replay_gpu_uuid is not None:
                raise RuntimeError("bundle preparation cannot preselect the replay GPU")
            original_gpu_uuid = plan.original_gpu_uuid
            clean_bundles[seed] = {
                "bundle_manifest_sha256": verified.manifest.manifest_sha256,
                "bundle_id": plan.bundle_id,
                "source_paths": plan.source_paths,
                "clean_directory": f"{seed}-clean-bundle",
            }
        result = {
            "schema_version": "upgradeguard.dev/reduction-replay/v1",
            "status": "prepared",
            "signature_sha256": signature,
            "reduction_budget": budget.model_dump(mode="json"),
            "numerical": remote.g2_session.model_dump(mode="json"),
            "performance": performance,
            "profile": remote.g7_session.model_dump(mode="json"),
            "original_gpu_uuid": original_gpu_uuid,
            "clean_bundles": clean_bundles,
        }
        (staging / "prepared.json").write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _request(
    failure_code: str,
    signature: str,
    predicate: dict[str, object],
    budget: ReductionBudget,
) -> dict[str, object]:
    return {
        "api_version": "upgradeguard.dev/v1alpha1",
        "kind": "ReductionRequest",
        "failure_code": failure_code,
        "signature_sha256": signature,
        "confirmation_count": budget.confirmation_count,
        "maximum_trials": budget.maximum_trials,
        "maximum_seconds": budget.maximum_seconds,
        "predicate": predicate,
    }


def _performance(
    root: Path,
    records: list[dict[str, object]],
    signature: str,
    budget: ReductionBudget,
) -> dict[str, object]:
    source = root / "G5-failure"
    source.mkdir(exist_ok=True)
    baseline = [float(record["G5"]["baseline_ms"]) for record in records]  # type: ignore[index]
    candidate = [float(record["G5"]["candidate_ms"]) for record in records]  # type: ignore[index]
    (source / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (source / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    request = _request(
        "PERFORMANCE_REGRESSION",
        signature,
        {
            "kind": "performance",
            "baseline_path": "baseline.json",
            "candidate_path": "candidate.json",
            "allowance": 0.03,
            "bootstrap_seed": 20260813,
            "bootstrap_replicates": 5000,
            "minimum_pairs": 20,
        },
        budget,
    )
    (source / "reduction-request.json").write_text(json.dumps(request), encoding="utf-8")
    destination = root / "G5-reduced"
    reduction = (
        json.loads((destination / "reduction-result.json").read_text())
        if destination.exists()
        else reduce_failure_directory(source, destination)
    )
    replay = _replay_performance_reduction(root, destination, request, reduction)
    return {**reduction, "clean_replay": replay}


def _replay_performance_reduction(
    root: Path,
    reduction: Path,
    request: dict[str, object],
    result: dict[str, object],
) -> dict[str, object]:
    """Re-execute the reduced G5 predicate from a newly empty directory."""

    replay_root = root / "G5-clean-replay"
    replay_path = replay_root / "replay-result.json"
    if replay_path.is_file() and not replay_path.is_symlink():
        retained = json.loads(replay_path.read_text(encoding="utf-8"))
        if not isinstance(retained, dict):
            raise RuntimeError("retained G5 clean replay is malformed")
        _validate_performance_replay(reduction, request, result, retained)
        return retained
    if replay_root.exists() or replay_root.is_symlink():
        raise RuntimeError("partial G5 clean replay requires a fresh preparation attempt")
    replay_root.mkdir()
    pairs_path = reduction / "reduced-pairs.json"
    raw_pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    if not isinstance(raw_pairs, list):
        raise RuntimeError("reduced G5 pairs are malformed")
    pairs = tuple(
        AcceptedPair(float(item["baseline_ms"]), float(item["candidate_ms"]))
        for item in raw_pairs
        if isinstance(item, dict)
    )
    original_pairs = result.get("original_pairs")
    confirmation_count = request.get("confirmation_count")
    if (
        not isinstance(original_pairs, int)
        or isinstance(original_pairs, bool)
        or not isinstance(confirmation_count, int)
        or isinstance(confirmation_count, bool)
    ):
        raise RuntimeError("G5 reduction counts are malformed")
    if len(pairs) != len(raw_pairs) or not (20 <= len(pairs) < original_pairs):
        raise RuntimeError("G5 reduction did not produce a smaller valid paired artifact")
    predicate = request.get("predicate")
    if not isinstance(predicate, dict):
        raise RuntimeError("G5 reduction predicate is malformed")
    observations = []
    for _ in range(confirmation_count):
        estimate = paired_ratio_gate(
            pairs,
            allowance=float(predicate["allowance"]),
            seed=int(predicate["bootstrap_seed"]),
            replicates=int(predicate["bootstrap_replicates"]),
            minimum_pairs=int(predicate["minimum_pairs"]),
        )
        observations.append(
            {
                "accepted_pairs": estimate.accepted_pairs,
                "point": estimate.point,
                "one_sided_lower": estimate.one_sided_lower,
                "one_sided_upper": estimate.one_sided_upper,
                "outcome": estimate.outcome.value,
            }
        )
        if estimate.outcome is not GateOutcome.REGRESSION:
            raise RuntimeError("G5 clean replay did not preserve PERFORMANCE_REGRESSION")
    replay = {
        "schema_version": "upgradeguard.dev/performance-replay/v1",
        "status": "passed",
        "fresh_directory": True,
        "expected_failure_code": FailureCode.PERFORMANCE_REGRESSION.value,
        "observed_failure_code": FailureCode.PERFORMANCE_REGRESSION.value,
        "expected_signature_sha256": request["signature_sha256"],
        "observed_signature_sha256": request["signature_sha256"],
        "request_sha256": sha256_file(root / "G5-failure" / "reduction-request.json"),
        "reduced_pairs_path": "../G5-reduced/reduced-pairs.json",
        "reduced_pairs_sha256": sha256_file(pairs_path),
        "reduced_pairs_bytes": pairs_path.stat().st_size,
        "confirmation_count": request["confirmation_count"],
        "maximum_trials": request["maximum_trials"],
        "maximum_seconds": request["maximum_seconds"],
        "observations": observations,
    }
    replay_path.write_text(
        json.dumps(replay, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _validate_performance_replay(reduction, request, result, replay)
    return replay


def _validate_performance_replay(
    reduction: Path,
    request: dict[str, object],
    result: dict[str, object],
    replay: dict[str, object],
) -> None:
    pairs_path = reduction / "reduced-pairs.json"
    if (
        replay.get("status") != "passed"
        or replay.get("fresh_directory") is not True
        or replay.get("expected_failure_code") != FailureCode.PERFORMANCE_REGRESSION.value
        or replay.get("observed_failure_code") != replay.get("expected_failure_code")
        or replay.get("expected_signature_sha256") != request.get("signature_sha256")
        or replay.get("observed_signature_sha256") != request.get("signature_sha256")
        or replay.get("reduced_pairs_sha256") != sha256_file(pairs_path)
        or replay.get("reduced_pairs_bytes") != pairs_path.stat().st_size
        or replay.get("confirmation_count") != request.get("confirmation_count")
        or replay.get("maximum_trials") != request.get("maximum_trials")
        or replay.get("maximum_seconds") != request.get("maximum_seconds")
        or result.get("reduced_pairs_sha256") != replay.get("reduced_pairs_sha256")
    ):
        raise RuntimeError("G5 clean replay identity differs from its reduction")


def _locked_reduction_budget(path: Path) -> ReductionBudget:
    """Read the one matrix-derived qualification budget used by every reducer."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        specification = QualificationSpec.model_validate(value)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise RuntimeError("locked qualification reduction budget is invalid") from error
    return specification.reduction_budget


def _g2_bundle(
    root: Path,
    state: Path,
    project: Path,
    corpus: Path,
    matrix: MatrixLock,
    signature: str,
    candidate: G2ReductionCandidate,
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
    original_compute_capability, cmake_cuda_architecture = _locked_cuda_architecture(matrix)
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
        f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architecture}",
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
                        "command": [
                            "python3",
                            "-m",
                            "upgrade_guard.reduce.g2_replay",
                            "--executable",
                            "/output/build/upgrade_guard_gpu_faults",
                            "--rows",
                            str(candidate.rows),
                            "--hidden",
                            str(candidate.hidden),
                            "--x-value",
                            format(candidate.x_value, ".9g"),
                            "--residual-value",
                            format(candidate.residual_value, ".9g"),
                            "--gamma-value",
                            format(candidate.gamma_value, ".9g"),
                        ],
                        "stdout_json_equals": {
                            "status": "failed",
                            "observation.G2.detected": True,
                            "observation.G2.control": "passed",
                        },
                        "expected_failure_code": "NUMERICAL_REGRESSION",
                        "failure_code_source": "stdout",
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
        shape_id=f"rows{candidate.rows}-h{candidate.hidden}",
        input_fixture_id="reduced-finite-constants",
        output_name="residual_rmsnorm",
        gate="candidate_to_reference",
        observed="residual omitted by quarantined G2 kernel",
        threshold="absolute error exceeds 0.1",
        evidence=(),
        signature_sha256=signature,
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
            extra_files={
                "commands/replay.json": commands,
                "profile.json": profile,
                "reduction/candidate.json": root / "candidate-reductions" / "G2-candidate.json",
                "reduction/session.json": root / "candidate-reductions" / "G2-session.json",
            },
            source_files=source_files,
            original_worker_image_manifest_digest=(
                matrix.environments[1].worker_image.manifest_digest
            ),
            original_gpu_uuid=matrix.gpu_uuid,
            base_image=matrix.environments[1].base_image.canonical_reference,
            base_image_manifest_digest=matrix.environments[1].base_image.manifest_digest,
            dockerfile=project / "containers" / "Dockerfile.worker",
            worker_lock=project / "containers" / "requirements-worker.txt",
            worker_build_arguments=(
                ("BASE_IMAGE", matrix.environments[1].base_image.canonical_reference),
                (
                    "BASE_MANIFEST_DIGEST",
                    matrix.environments[1].base_image.manifest_digest,
                ),
            ),
            minimum_compute_capability=(
                matrix.environments[1].compatibility.minimum_compute_capability
            ),
            minimum_driver=matrix.environments[1].compatibility.minimum_driver,
            minimum_vram_mib=_minimum_vram_mib(project),
            original_compute_capability=original_compute_capability,
            source_build_command=build_command,
        ),
        bundle,
    )
    return bundle


def _g7_bundle(
    root: Path,
    state: Path,
    project: Path,
    core: Path,
    matrix: MatrixLock,
    signature: str,
    candidate: G7ReductionCandidate,
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
                    "min": [
                        candidate.profile_min_batch,
                        candidate.profile_min_sequence,
                        candidate.hidden,
                    ],
                    "opt": [
                        candidate.profile_opt_batch,
                        candidate.profile_opt_sequence,
                        candidate.hidden,
                    ],
                    "max": [
                        candidate.profile_max_batch,
                        candidate.profile_max_sequence,
                        candidate.hidden,
                    ],
                },
                "mask": {
                    "min": [
                        candidate.profile_min_batch,
                        1,
                        1,
                        candidate.profile_min_sequence,
                    ],
                    "opt": [
                        candidate.profile_opt_batch,
                        1,
                        1,
                        candidate.profile_opt_sequence,
                    ],
                    "max": [
                        candidate.profile_max_batch,
                        1,
                        1,
                        candidate.profile_max_sequence,
                    ],
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
        "--workspace-bytes",
        str(candidate.workspace_bytes),
        "--optimization-level",
        str(candidate.optimization_level),
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
                        "expected_failure_code": "PROFILE_REJECTED",
                        "failure_code_source": "result_file",
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
        shape_id=f"b{candidate.batch}_s{candidate.sequence}",
        input_fixture_id=f"reduced-{candidate.input_mode}",
        output_name=None,
        gate="optimization_profile",
        observed=(
            f"tokens shape [{candidate.batch}, {candidate.sequence}, {candidate.hidden}] "
            "exceeds the reduced locked profile"
        ),
        threshold=(
            f"maximum tokens shape [{candidate.profile_max_batch}, "
            f"{candidate.profile_max_sequence}, {candidate.hidden}]"
        ),
        evidence=(),
        signature_sha256=signature,
    )
    reduced_tokens, reduced_mask, control_tokens, control_mask = _materialize_g7_bundle_inputs(
        environment_root, candidate
    )
    export_bundle(
        BundleExport(
            id=f"G7-{signature.removeprefix('sha256:')[:12]}",
            created_at=datetime.now(UTC),
            baseline_environment=baseline_path,
            candidate_environment=candidate_path,
            qualification=state / "full.yaml",
            model=candidate.model_path,
            inputs=(
                reduced_tokens,
                reduced_mask,
            ),
            expected_failure=expected,
            extra_files={
                "commands/replay.json": commands,
                "profile.json": profile,
                "control/tokens.npy": control_tokens,
                "control/mask.npy": control_mask,
                "reduction/candidate.json": root / "candidate-reductions" / "G7-candidate.json",
                "reduction/session.json": root / "candidate-reductions" / "G7-session.json",
            },
            source_files=source_files,
            original_worker_image_manifest_digest=(
                matrix.environments[1].worker_image.manifest_digest
            ),
            original_gpu_uuid=matrix.gpu_uuid,
            base_image=matrix.environments[1].base_image.canonical_reference,
            base_image_manifest_digest=matrix.environments[1].base_image.manifest_digest,
            dockerfile=project / "containers" / "Dockerfile.worker",
            worker_lock=project / "containers" / "requirements-worker.txt",
            worker_build_arguments=(
                ("BASE_IMAGE", matrix.environments[1].base_image.canonical_reference),
                (
                    "BASE_MANIFEST_DIGEST",
                    matrix.environments[1].base_image.manifest_digest,
                ),
            ),
            minimum_compute_capability=(
                matrix.environments[1].compatibility.minimum_compute_capability
            ),
            minimum_driver=matrix.environments[1].compatibility.minimum_driver,
            minimum_vram_mib=_minimum_vram_mib(project),
            source_build_command=build_command,
        ),
        bundle,
    )
    return bundle


def _materialize_g7_bundle_inputs(
    root: Path,
    candidate: G7ReductionCandidate,
) -> tuple[Path, Path, Path, Path]:
    inputs = root / "G7-reduced-inputs"
    inputs.mkdir(exist_ok=True)
    tokens = inputs / "tokens.npy"
    mask = inputs / "mask.npy"
    if candidate.input_mode == "original":
        assert candidate.tokens_path is not None and candidate.mask_path is not None
        shutil.copyfile(candidate.tokens_path, tokens)
        shutil.copyfile(candidate.mask_path, mask)
    else:
        value = 0.0 if candidate.input_mode == "zeros" else 1.0
        np.save(
            tokens,
            np.full((candidate.batch, candidate.sequence, candidate.hidden), value, np.float32),
            allow_pickle=False,
        )
        np.save(
            mask,
            np.full((candidate.batch, 1, 1, candidate.sequence), value, np.float32),
            allow_pickle=False,
        )
    control = root / "G7-reduced-control"
    control.mkdir(exist_ok=True)
    control_tokens = control / "tokens.npy"
    control_mask = control / "mask.npy"
    np.save(
        control_tokens,
        np.zeros(
            (
                candidate.profile_min_batch,
                candidate.profile_min_sequence,
                candidate.hidden,
            ),
            np.float32,
        ),
        allow_pickle=False,
    )
    np.save(
        control_mask,
        np.zeros(
            (
                candidate.profile_min_batch,
                1,
                1,
                candidate.profile_min_sequence,
            ),
            np.float32,
        ),
        allow_pickle=False,
    )
    return tokens, mask, control_tokens, control_mask


def _minimum_vram_mib(project: Path) -> int:
    """Read the locked compatibility policy used by the source-bearing replay."""

    policy = project / "src" / "upgrade_guard" / "matrix" / "compatibility-rules.json"
    try:
        value = json.loads(policy.read_text(encoding="utf-8"))["minimum_vram_mib"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("compatibility policy has no valid minimum_vram_mib") from error
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("compatibility policy minimum_vram_mib must be positive")
    return value


def _locked_cuda_architecture(matrix: MatrixLock) -> tuple[str, str]:
    """Derive the CMake architecture only from the candidate environment lock."""

    compute_capability = matrix.environments[1].probe.gpu.compute_capability
    return compute_capability, canonical_cmake_cuda_architecture(compute_capability)


if __name__ == "__main__":
    main()
