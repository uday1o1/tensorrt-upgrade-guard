"""Sequential baseline and candidate qualification orchestration."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from pydantic import ValidationError

from upgrade_guard.compare.determinism import summarize_determinism
from upgrade_guard.compare.memory import device_memory_gate, engine_size_gate
from upgrade_guard.compare.numerical import decide_three_way
from upgrade_guard.compare.performance import (
    AcceptedPair,
    GateOutcome,
    weighted_performance_gate,
)
from upgrade_guard.compare.validity import ValidityObservation, rejection_reasons
from upgrade_guard.containers.commands import CommandRunner, Runner, command_sha256
from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.common import PrecisionMode
from upgrade_guard.contracts.environment import EnvironmentLock, MatrixLock
from upgrade_guard.contracts.qualification import ConcreteShape, QualificationSpec
from upgrade_guard.corpus.reference import run_onnx_reference
from upgrade_guard.corpus.registry import CorpusLock
from upgrade_guard.errors import (
    FailureCode,
    InfrastructureError,
    InvalidInputError,
    UnsupportedEnvironmentError,
)
from upgrade_guard.worker.trtexec import benchmark_command, load_exported_times


@dataclass(frozen=True)
class QualificationOutcome:
    """Published end-to-end result identity."""

    directory: Path
    status: Literal["passed", "failed", "inconclusive", "infrastructure_invalid"]
    failure_codes: tuple[FailureCode, ...]


class QualificationRunner:
    """Run frozen artifacts in one baseline then one candidate worker at a time."""

    def __init__(self, runner: Runner | None = None, source_root: Path | None = None) -> None:
        self.runner = runner or CommandRunner()
        self.worker = DockerGpuWorker(self.runner)
        self.source_root = source_root or Path(__file__).resolve().parents[2]

    def run(self, specification_path: Path, destination: Path) -> QualificationOutcome:
        """Validate locks, run all declared cases, and publish complete evidence."""

        if destination.exists() or destination.is_symlink():
            raise InvalidInputError("refusing to overwrite qualification output")
        specification = _load_specification(specification_path)
        matrix_path = _resolve_authored_path(specification_path, specification.environment_lock)
        matrix = _load_model(matrix_path, MatrixLock, "environment lock")
        if matrix.computed_sha256() != matrix.lock_sha256:
            raise InvalidInputError("environment lock self-hash differs")
        environments = {environment.id: environment for environment in matrix.environments}
        if set(environments) != {
            specification.baseline_environment_id,
            specification.candidate_environment_id,
        }:
            raise InvalidInputError("qualification and environment lock IDs differ")
        if specification.hardware_validity.selected_gpu_uuid != matrix.gpu_uuid:
            raise InvalidInputError("qualification and environment lock GPU UUIDs differ")
        corpus_root = self.source_root / ".upgrade-guard" / "corpora" / specification.corpus_lock_id
        corpus = _load_model(corpus_root / "corpus.lock.json", CorpusLock, "corpus lock")
        if corpus.id != specification.corpus_lock_id:
            raise InvalidInputError("qualification and corpus lock IDs differ")
        _verify_corpus(corpus_root, corpus)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            summary = self._execute(specification, matrix, environments, corpus_root, staging)
            _write_json(staging / "qualification-summary.json", summary)
            staging.replace(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        failure_codes = tuple(FailureCode(item) for item in summary["failure_codes"])
        return QualificationOutcome(destination, summary["status"], failure_codes)

    def _execute(
        self,
        specification: QualificationSpec,
        matrix: MatrixLock,
        environments: dict[str, EnvironmentLock],
        corpus_root: Path,
        staging: Path,
    ) -> dict[str, Any]:
        started = datetime.now(UTC)
        all_case_evidence: list[dict[str, Any]] = []
        failures: list[FailureCode] = []
        if tuple(specification.required_cases) != ("tiny-transformer",):
            raise UnsupportedEnvironmentError(
                "this qualification path currently requires the tiny-transformer case"
            )
        shape_map = {shape.id: shape for shape in specification.concrete_shapes}
        if set(shape_map) != set(specification.performance.shape_weights):
            raise InvalidInputError("concrete shapes and performance weights differ")
        profile = specification.optimization_profiles[0]
        for precision in specification.precision_modes:
            suffix = "fp32" if precision.value == "fp32" else "fp16"
            if precision.value == "qdq":
                raise UnsupportedEnvironmentError("Q/DQ transformer artifact is not declared")
            model_relative = Path("models") / f"tiny-transformer-{suffix}.onnx"
            builds: dict[str, list[dict[str, Any]]] = {}
            runs: dict[str, dict[str, dict[str, Any]]] = {}
            for environment_id in (
                specification.baseline_environment_id,
                specification.candidate_environment_id,
            ):
                environment = environments[environment_id]
                builds[environment_id] = self._build_confirmations(
                    specification,
                    matrix,
                    environment,
                    corpus_root,
                    staging,
                    model_relative,
                    profile.inputs,
                    suffix,
                )
                runs[environment_id] = {}
                for shape_id, shape in shape_map.items():
                    runs[environment_id][shape_id] = self._run_correctness(
                        specification,
                        matrix,
                        environment,
                        corpus_root,
                        staging,
                        suffix,
                        shape,
                    )
            for shape_id in sorted(shape_map):
                evidence, failure = _compare_correctness_case(
                    specification,
                    corpus_root,
                    staging,
                    suffix,
                    precision,
                    shape_id,
                    runs[specification.baseline_environment_id][shape_id],
                    runs[specification.candidate_environment_id][shape_id],
                )
                all_case_evidence.append(evidence)
                if failure is not None:
                    failures.append(failure)
            memory = _memory_evidence(
                builds[specification.baseline_environment_id],
                builds[specification.candidate_environment_id],
            )
            all_case_evidence.append({"precision": suffix, "memory": memory})
            for outcome in (memory["engine_bytes"]["outcome"], memory["device_memory"]["outcome"]):
                if outcome == GateOutcome.REGRESSION.value:
                    failures.append(FailureCode.MEMORY_REGRESSION)
                elif outcome == GateOutcome.INFRASTRUCTURE_INVALID.value:
                    failures.append(FailureCode.INFRASTRUCTURE_INVALID)
            performance = self._run_performance(
                specification,
                matrix,
                environments,
                staging,
                suffix,
                shape_map,
            )
            all_case_evidence.append({"precision": suffix, "performance": performance})
            performance_outcome = GateOutcome(performance["outcome"])
            if performance_outcome is GateOutcome.REGRESSION:
                failures.append(FailureCode.PERFORMANCE_REGRESSION)
            elif performance_outcome is GateOutcome.INCONCLUSIVE:
                failures.append(FailureCode.INCONCLUSIVE)
            elif performance_outcome is GateOutcome.INFRASTRUCTURE_INVALID:
                failures.append(FailureCode.INFRASTRUCTURE_INVALID)
        unique_failures = tuple(dict.fromkeys(failures))
        status = _status(unique_failures)
        return {
            "schema_version": "upgradeguard.dev/qualification-summary/v1",
            "status": status,
            "failure_codes": [item.value for item in unique_failures],
            "started_at": started.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "environment_lock_sha256": matrix.lock_sha256,
            "gpu_uuid": matrix.gpu_uuid,
            "stack_attribution": (
                "Observed changes belong to the complete locked baseline and candidate stacks. "
                "No result is attributed to TensorRT alone."
            ),
            "cases": all_case_evidence,
        }

    def _build_confirmations(
        self,
        specification: QualificationSpec,
        matrix: MatrixLock,
        environment: EnvironmentLock,
        corpus_root: Path,
        staging: Path,
        model_relative: Path,
        profile_inputs: dict[str, Any],
        precision: str,
    ) -> list[dict[str, Any]]:
        environment_output = staging / environment.id / precision
        environment_output.mkdir(parents=True, exist_ok=True)
        profile_path = environment_output / "profile.json"
        _write_json(
            profile_path,
            {
                name: {
                    "min": list(shape.minimum),
                    "opt": list(shape.optimum),
                    "max": list(shape.maximum),
                }
                for name, shape in profile_inputs.items()
            },
        )
        builds: list[dict[str, Any]] = []
        for index in range(specification.memory.confirmation_builds):
            engine = environment_output / f"engine-{index}.plan"
            inspector = environment_output / f"engine-{index}.inspector.json"
            result = environment_output / f"build-{index}.json"
            timing_cache = environment_output / "timing.cache"
            command = (
                "python3",
                "-m",
                "upgrade_guard.worker.build_engine",
                "--model",
                f"/corpus/{model_relative.as_posix()}",
                "--profile",
                f"/output/{environment.id}/{precision}/profile.json",
                "--engine",
                f"/output/{environment.id}/{precision}/engine-{index}.plan",
                "--inspector",
                f"/output/{environment.id}/{precision}/engine-{index}.inspector.json",
                "--timing-cache",
                f"/output/{environment.id}/{precision}/timing.cache",
                "--result",
                f"/output/{environment.id}/{precision}/build-{index}.json",
                "--workspace-bytes",
                str(specification.builder.workspace_limit_bytes),
                "--optimization-level",
                str(specification.builder.optimization_level),
                "--cache-state",
                "cold" if index == 0 else "warm",
            )
            self.worker.run(
                image=environment.worker_image.canonical_reference,
                gpu_uuid=matrix.gpu_uuid,
                mounts=WorkerMounts(self.source_root, corpus_root, staging),
                command=command,
                timeout_seconds=1800,
            )
            build = _read_json(result)
            if build.get("status") != "passed":
                raise InfrastructureError("worker engine build did not publish a passing manifest")
            build = _translate_container_paths(build, staging, corpus_root)
            build["command"] = list(command)
            build["command_sha256"] = command_sha256(command)
            build["environment_id"] = environment.id
            builds.append(build)
            if not engine.is_file() or not inspector.is_file() or not timing_cache.is_file():
                raise InfrastructureError("worker build omitted a required artifact")
        return builds

    def _run_correctness(
        self,
        specification: QualificationSpec,
        matrix: MatrixLock,
        environment: EnvironmentLock,
        corpus_root: Path,
        staging: Path,
        precision: str,
        shape: ConcreteShape,
    ) -> dict[str, Any]:
        base = staging / environment.id / precision
        result = base / shape.id / "correctness.json"
        input_directory = f"/corpus/inputs/tiny-transformer-{precision}/{shape.id}"
        command = (
            "python3",
            "-m",
            "upgrade_guard.worker.run_correctness",
            "--engine",
            f"/output/{environment.id}/{precision}/engine-0.plan",
            "--input",
            f"tokens={input_directory}/tokens.npy",
            "--input",
            f"mask={input_directory}/mask.npy",
            "--output",
            f"/output/{environment.id}/{precision}/{shape.id}/outputs",
            "--result",
            f"/output/{environment.id}/{precision}/{shape.id}/correctness.json",
            "--repetitions",
            str(specification.determinism.repetitions),
        )
        self.worker.run(
            image=environment.worker_image.canonical_reference,
            gpu_uuid=matrix.gpu_uuid,
            mounts=WorkerMounts(self.source_root, corpus_root, staging),
            command=command,
            timeout_seconds=900,
        )
        run = _read_json(result)
        if run.get("status") != "passed":
            raise InfrastructureError("worker correctness run did not publish a passing result")
        run = cast(dict[str, Any], _translate_container_paths(run, staging, corpus_root))
        run["command"] = list(command)
        run["command_sha256"] = command_sha256(command)
        run["environment_id"] = environment.id
        return run

    def _run_performance(
        self,
        specification: QualificationSpec,
        matrix: MatrixLock,
        environments: dict[str, EnvironmentLock],
        staging: Path,
        precision: str,
        shape_map: dict[str, ConcreteShape],
    ) -> dict[str, Any]:
        rng = random.Random(specification.performance.bootstrap_seed)  # noqa: S311
        pairs_by_shape: dict[str, tuple[AcceptedPair, ...]] = {}
        raw_blocks: dict[str, list[dict[str, Any]]] = {}
        for shape_id, shape in sorted(shape_map.items()):
            pairs: list[AcceptedPair] = []
            raw_blocks[shape_id] = []
            maximum_attempts = specification.performance.minimum_accepted_pairs * 3
            attempt = 0
            while (
                len(pairs) < specification.performance.minimum_accepted_pairs
                and attempt < maximum_attempts
            ):
                order = [
                    specification.baseline_environment_id,
                    specification.candidate_environment_id,
                ]
                if rng.getrandbits(1):
                    order.reverse()
                medians: dict[str, float] = {}
                pair_records: list[dict[str, Any]] = []
                for order_index, environment_id in enumerate(order):
                    environment = environments[environment_id]
                    idle: dict[str, Any] = {}
                    idle_reasons: tuple[str, ...] = ("gpu_idle_wait_not_started",)
                    idle_attempts: list[dict[str, Any]] = []
                    for idle_attempt in range(60):
                        idle, idle_reasons = _observe_validity(
                            self.runner,
                            matrix.gpu_uuid,
                            specification,
                        )
                        idle_attempts.append(
                            {
                                "attempt": idle_attempt,
                                "observed": idle,
                                "rejection_reasons": list(idle_reasons),
                            }
                        )
                        if not idle_reasons:
                            break
                        time.sleep(1)
                    if idle_reasons:
                        pair_records.append(
                            {
                                "pair_attempt": attempt,
                                "order_in_pair": order_index,
                                "environment_id": environment_id,
                                "accepted": False,
                                "rejection_reasons": list(idle_reasons),
                                "idle": idle,
                                "idle_attempts": idle_attempts,
                                "profiled": False,
                            }
                        )
                        break
                    precondition_command = benchmark_command(
                        trtexec_path=environment.probe.trtexec.path or "trtexec",
                        supported_options=environment.probe.trtexec.options,
                        engine=f"/output/{environment_id}/{precision}/engine-0.plan",
                        shapes=shape.inputs,
                        export_times=(
                            f"/output/{environment_id}/{precision}/{shape_id}/"
                            f"precondition-attempt-{attempt:02d}.json"
                        ),
                        warmup_milliseconds=specification.performance.warmup_milliseconds,
                        measurement_milliseconds=1000,
                    )
                    self._run_benchmark_worker(
                        specification,
                        matrix,
                        environment,
                        staging,
                        precondition_command,
                        measurement_milliseconds=1000,
                    )
                    before, before_reasons = _observe_validity(
                        self.runner,
                        matrix.gpu_uuid,
                        specification,
                        require_idle=False,
                    )
                    if before_reasons:
                        pair_records.append(
                            {
                                "pair_attempt": attempt,
                                "order_in_pair": order_index,
                                "environment_id": environment_id,
                                "accepted": False,
                                "rejection_reasons": list(before_reasons),
                                "idle": idle,
                                "idle_attempts": idle_attempts,
                                "before": before,
                                "precondition_command": list(precondition_command),
                                "precondition_command_sha256": command_sha256(precondition_command),
                                "profiled": False,
                            }
                        )
                        break
                    times_path = (
                        staging
                        / environment_id
                        / precision
                        / shape_id
                        / f"times-attempt-{attempt:02d}.json"
                    )
                    command = benchmark_command(
                        trtexec_path=environment.probe.trtexec.path or "trtexec",
                        supported_options=environment.probe.trtexec.options,
                        engine=f"/output/{environment_id}/{precision}/engine-0.plan",
                        shapes=shape.inputs,
                        export_times=(
                            f"/output/{environment_id}/{precision}/{shape_id}/"
                            f"times-attempt-{attempt:02d}.json"
                        ),
                        warmup_milliseconds=specification.performance.warmup_milliseconds,
                        measurement_milliseconds=specification.performance.measurement_milliseconds,
                    )
                    self._run_benchmark_worker(
                        specification,
                        matrix,
                        environment,
                        staging,
                        command,
                        measurement_milliseconds=(
                            specification.performance.measurement_milliseconds
                        ),
                    )
                    timing = load_exported_times(times_path)
                    after, after_reasons = _observe_validity(
                        self.runner,
                        matrix.gpu_uuid,
                        specification,
                        require_idle=False,
                    )
                    after_reasons = (
                        *after_reasons,
                        *_block_variation_reasons(before, after, specification),
                    )
                    medians[environment_id] = timing.median_milliseconds
                    pair_records.append(
                        {
                            "pair_attempt": attempt,
                            "order_in_pair": order_index,
                            "environment_id": environment_id,
                            "command": list(command),
                            "command_sha256": command_sha256(command),
                            "accepted": not after_reasons,
                            "rejection_reasons": list(after_reasons),
                            "idle": idle,
                            "idle_attempts": idle_attempts,
                            "before": before,
                            "after": after,
                            "precondition_command": list(precondition_command),
                            "precondition_command_sha256": command_sha256(precondition_command),
                            **asdict(timing),
                            "profiled": False,
                        }
                    )
                    if after_reasons:
                        break
                pair_valid = len(pair_records) == 2 and all(
                    bool(record["accepted"]) for record in pair_records
                )
                if pair_valid:
                    accepted_index = len(pairs)
                    for record in pair_records:
                        record["pair_index"] = accepted_index
                    pairs.append(
                        AcceptedPair(
                            medians[specification.baseline_environment_id],
                            medians[specification.candidate_environment_id],
                        )
                    )
                else:
                    for record in pair_records:
                        if record["accepted"]:
                            record["accepted"] = False
                            record["rejection_reasons"] = ["paired_block_invalid"]
                raw_blocks[shape_id].extend(pair_records)
                attempt += 1
            pairs_by_shape[shape_id] = tuple(pairs)
        gate = weighted_performance_gate(
            pairs_by_shape,
            specification.performance.shape_weights,
            specification.performance.shape_allowances,
            aggregate_allowance=specification.performance.practical_allowance,
            seed=specification.performance.bootstrap_seed,
            replicates=specification.performance.bootstrap_replicates,
            minimum_pairs=specification.performance.minimum_accepted_pairs,
        )
        return {
            "outcome": gate.outcome.value,
            "shapes": {name: _jsonable(asdict(value)) for name, value in gate.shapes.items()},
            "aggregate": _jsonable(asdict(gate.aggregate)),
            "raw_blocks": raw_blocks,
            "profiled": False,
        }

    def _run_benchmark_worker(
        self,
        specification: QualificationSpec,
        matrix: MatrixLock,
        environment: EnvironmentLock,
        staging: Path,
        command: tuple[str, ...],
        *,
        measurement_milliseconds: int,
    ) -> None:
        self.worker.run(
            image=environment.worker_image.canonical_reference,
            gpu_uuid=matrix.gpu_uuid,
            mounts=WorkerMounts(
                self.source_root,
                self.source_root / ".upgrade-guard" / "corpora" / specification.corpus_lock_id,
                staging,
            ),
            command=command,
            timeout_seconds=max(120, measurement_milliseconds / 1000 + 60),
        )


def compare_stored_run(directory: Path) -> dict[str, Any]:
    """Validate and return the stored end-to-end summary without recomputation drift."""

    summary = _read_json(directory / "qualification-summary.json")
    if summary.get("schema_version") != "upgradeguard.dev/qualification-summary/v1":
        raise InvalidInputError("run summary schema version is unsupported")
    if summary.get("status") not in {
        "passed",
        "failed",
        "inconclusive",
        "infrastructure_invalid",
    }:
        raise InvalidInputError("run summary status is invalid")
    return summary


def _compare_correctness_case(
    specification: QualificationSpec,
    corpus_root: Path,
    output_root: Path,
    precision: str,
    precision_mode: PrecisionMode,
    shape_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], FailureCode | None]:
    input_root = corpus_root / "inputs" / f"tiny-transformer-{precision}" / shape_id
    model = corpus_root / "models" / f"tiny-transformer-{precision}.onnx"
    reference = run_onnx_reference(
        model,
        {
            "tokens": np.load(input_root / "tokens.npy", allow_pickle=False),
            "mask": np.load(input_root / "mask.npy", allow_pickle=False),
        },
    )
    baseline_paths = _output_paths(baseline, "output", output_root)
    candidate_paths = _output_paths(candidate, "output", output_root)
    baseline_values = tuple(np.load(path, allow_pickle=False) for path in baseline_paths)
    candidate_values = tuple(np.load(path, allow_pickle=False) for path in candidate_paths)
    baseline_determinism = summarize_determinism(
        baseline_values,
        tuple(baseline["input_sha256"].values()),
        specification.determinism.tolerance,
    )
    candidate_determinism = summarize_determinism(
        candidate_values,
        tuple(candidate["input_sha256"].values()),
        specification.determinism.tolerance,
    )
    numerical_policy = specification.numerical_policy(precision_mode)
    decision = decide_three_way(
        "output",
        reference[0].values,
        baseline_values[0],
        candidate_values[0],
        baseline_policy=numerical_policy.baseline_to_reference,
        candidate_policy=numerical_policy.candidate_to_reference,
        drift_policy=numerical_policy.candidate_to_baseline,
    )
    failure = decision.failure_code
    if failure is None and (
        not baseline_determinism.tolerance_stable
        or not candidate_determinism.tolerance_stable
        or (specification.determinism.require_bitwise and not candidate_determinism.bitwise_stable)
    ):
        failure = FailureCode.NONDETERMINISM_REGRESSION
    return (
        {
            "model": "tiny-transformer",
            "precision": precision,
            "shape_id": shape_id,
            "reference": {
                "name": reference[0].name,
                "dtype": reference[0].dtype,
                "shape": reference[0].shape,
                "sha256": reference[0].sha256,
            },
            "baseline_to_reference": decision.baseline_to_reference.model_dump(mode="json"),
            "candidate_to_reference": decision.candidate_to_reference.model_dump(mode="json"),
            "candidate_to_baseline": decision.candidate_to_baseline.model_dump(mode="json"),
            "baseline_determinism": baseline_determinism.model_dump(mode="json"),
            "candidate_determinism": candidate_determinism.model_dump(mode="json"),
            "failure_code": failure.value if failure else None,
        },
        failure,
    )


def _output_paths(run: dict[str, Any], output_name: str, output_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for repetition in run["repetitions"]:
        matching = [item for item in repetition["outputs"] if item["name"] == output_name]
        if len(matching) != 1:
            raise InvalidInputError("worker output schema changed between repetitions")
        path = Path(matching[0]["path"]).resolve(strict=True)
        if not path.is_relative_to(output_root.resolve()) or path.suffix != ".npy":
            raise InvalidInputError("worker output path escaped the run directory")
        if sha256_file(path) != matching[0]["sha256"]:
            raise InvalidInputError("worker output hash differs from stored evidence")
        paths.append(path)
    return tuple(paths)


def _memory_evidence(
    baseline_builds: list[dict[str, Any]], candidate_builds: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_engine = tuple(int(item["engine"]["bytes"]) for item in baseline_builds)
    candidate_engine = tuple(int(item["engine"]["bytes"]) for item in candidate_builds)
    baseline_device = tuple(int(item["engine"]["device_memory_bytes"]) for item in baseline_builds)
    candidate_device = tuple(
        int(item["engine"]["device_memory_bytes"]) for item in candidate_builds
    )
    return {
        "engine_bytes": _jsonable(asdict(engine_size_gate(baseline_engine, candidate_engine))),
        "device_memory": _jsonable(asdict(device_memory_gate(baseline_device, candidate_device))),
    }


def _load_specification(path: Path) -> QualificationSpec:
    try:
        return QualificationSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as error:
        raise InvalidInputError("qualification specification is invalid") from error


def _load_model(path: Path, model: Any, label: str) -> Any:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise InvalidInputError(f"{label} is invalid", details={"path": str(path)}) from error


def _resolve_authored_path(source: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (source.parent / candidate).resolve()


def _verify_corpus(root: Path, lock: CorpusLock) -> None:
    expected = {artifact.path: artifact for artifact in lock.artifacts}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "corpus.lock.json"
    }
    if observed != set(expected):
        raise InvalidInputError("materialized corpus inventory differs from lock")
    for relative, artifact in expected.items():
        path = root / relative
        if path.stat().st_size != artifact.bytes or sha256_file(path) != artifact.sha256:
            raise InvalidInputError("materialized corpus artifact differs from lock")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InfrastructureError("worker omitted valid machine JSON") from error
    if not isinstance(value, dict):
        raise InfrastructureError("worker machine result must be an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, GateOutcome):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _translate_container_paths(value: Any, output_root: Path, corpus_root: Path) -> Any:
    if isinstance(value, str):
        if value.startswith("/output/"):
            return str(output_root / value.removeprefix("/output/"))
        if value.startswith("/corpus/"):
            return str(corpus_root / value.removeprefix("/corpus/"))
        return value
    if isinstance(value, dict):
        return {
            key: _translate_container_paths(item, output_root, corpus_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_translate_container_paths(item, output_root, corpus_root) for item in value]
    return value


def _status(failures: tuple[FailureCode, ...]) -> str:
    if FailureCode.INFRASTRUCTURE_INVALID in failures:
        return "infrastructure_invalid"
    if FailureCode.INCONCLUSIVE in failures:
        return "inconclusive"
    if failures:
        return "failed"
    return "passed"


def _observe_validity(
    runner: Runner,
    gpu_uuid: str,
    specification: QualificationSpec,
    *,
    require_idle: bool = True,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    query = runner.run(
        (
            "nvidia-smi",
            f"--id={gpu_uuid}",
            "--query-gpu=uuid,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        timeout_seconds=15,
    )
    if query.returncode != 0:
        return ({"query_error": query.stderr.strip()}, ("gpu_observation_failed",))
    fields = [field.strip() for field in query.stdout.strip().split(",")]
    if len(fields) != 7:
        return ({"raw": query.stdout.strip()}, ("gpu_observation_malformed",))
    process_query = runner.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ),
        timeout_seconds=15,
    )
    processes = tuple(
        line.strip()
        for line in process_query.stdout.splitlines()
        if line.strip().startswith(gpu_uuid + ",")
    )
    temperature = _optional_float(fields[1])
    graphics_clock = _optional_int(fields[2])
    memory_clock = _optional_int(fields[3])
    power = _optional_float(fields[4])
    power_limit = _optional_float(fields[5])
    utilization = _optional_float(fields[6])
    observed: dict[str, Any] = {
        "gpu_uuid": fields[0],
        "temperature_celsius": temperature,
        "graphics_clock_mhz": graphics_clock,
        "memory_clock_mhz": memory_clock,
        "power_watts": power,
        "power_limit_watts": power_limit,
        "utilization_percent": utilization,
        "host_load_1m": _host_load_1m(),
        "competing_compute_processes": processes,
    }
    observation = ValidityObservation(
        gpu_uuid=fields[0],
        expected_gpu_uuid=gpu_uuid,
        temperature_celsius=temperature,
        maximum_temperature_celsius=(specification.hardware_validity.maximum_temperature_celsius),
        utilization_percent_before=utilization,
        maximum_idle_utilization_percent=(
            specification.hardware_validity.maximum_gpu_utilization_before_block
            if require_idle
            else 100.0
        ),
        competing_compute_processes=(
            processes if specification.hardware_validity.reject_competing_compute_processes else ()
        ),
        graphics_clock_mhz=graphics_clock,
        minimum_graphics_clock_mhz=None,
    )
    reasons = list(rejection_reasons(observation))
    if process_query.returncode != 0:
        reasons.append("process_observation_failed")
    return observed, tuple(reasons)


def _block_variation_reasons(
    before: dict[str, Any], after: dict[str, Any], specification: QualificationSpec
) -> tuple[str, ...]:
    reasons: list[str] = []
    for field, allowance, label in (
        (
            "graphics_clock_mhz",
            specification.hardware_validity.maximum_clock_variation_ratio,
            "graphics_clock_variation_exceeded",
        ),
        (
            "power_watts",
            specification.hardware_validity.maximum_power_variation_ratio,
            "power_variation_exceeded",
        ),
    ):
        first = before.get(field)
        second = after.get(field)
        if not isinstance(first, int | float) or not isinstance(second, int | float) or first <= 0:
            reasons.append(f"{field}_observation_missing")
        elif abs(float(second) - float(first)) / float(first) > allowance:
            reasons.append(label)
    if specification.hardware_validity.require_stable_power_limit:
        first_limit = before.get("power_limit_watts")
        second_limit = after.get("power_limit_watts")
        if first_limit is None or second_limit is None:
            reasons.append("power_limit_observation_missing")
        elif first_limit != second_limit:
            reasons.append("power_limit_changed")
    return tuple(reasons)


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _host_load_1m() -> float | None:
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None
