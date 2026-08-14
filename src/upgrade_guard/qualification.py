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
from typing import Any, Literal, Protocol, cast

import numpy as np
import onnx
import yaml
from pydantic import ValidationError

from upgrade_guard.classify import status_for_failure
from upgrade_guard.compare.determinism import summarize_determinism
from upgrade_guard.compare.memory import confirmed_memory_gate
from upgrade_guard.compare.numerical import ThreeWayPrecedenceError, decide_three_way
from upgrade_guard.compare.performance import (
    AcceptedPair,
    GateOutcome,
    weighted_performance_gate,
)
from upgrade_guard.compare.validity import ValidityObservation, rejection_reasons
from upgrade_guard.containers.commands import CommandRunner, Runner, command_sha256
from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from upgrade_guard.contracts.build import (
    BuildManifestAdapterContext,
    WorkerBuildResult,
    adapt_worker_build,
)
from upgrade_guard.contracts.case import (
    CaseManifest,
    ReferenceCapability,
    SourceAttribution,
    adapt_case_manifest,
)
from upgrade_guard.contracts.common import (
    ArtifactReference,
    FailureRecord,
    Phase,
    PrecisionMode,
    TensorContract,
)
from upgrade_guard.contracts.environment import EnvironmentLock, MatrixLock
from upgrade_guard.contracts.qualification import ConcreteShape, QualificationSpec
from upgrade_guard.contracts.results import (
    HardwareObservation,
    RunResult,
    RunResultAdapterContext,
    WorkerCorrectnessResult,
    adapt_worker_run,
)
from upgrade_guard.corpus.registry import CorpusLock
from upgrade_guard.errors import (
    FailureCode,
    InfrastructureError,
    InvalidInputError,
    UnsupportedEnvironmentError,
)
from upgrade_guard.matrix.lock import MatrixLocker
from upgrade_guard.worker.trtexec import (
    benchmark_command,
    freeze_raw_inputs,
    load_exported_times,
)


class LockVerifier(Protocol):
    """Live environment-lock verification boundary."""

    def verify(self, expected: MatrixLock) -> MatrixLock: ...


type TensorDtype = Literal[
    "float16",
    "float32",
    "float64",
    "int8",
    "int32",
    "int64",
    "bool",
]


@dataclass(frozen=True)
class QualificationOutcome:
    """Published end-to-end result identity."""

    directory: Path
    status: Literal[
        "passed",
        "failed",
        "unsupported",
        "inconclusive",
        "infrastructure_invalid",
    ]
    failure_codes: tuple[FailureCode, ...]


class _TypedWorkerFailureError(RuntimeError):
    """A conclusive worker failure already mapped to the stable taxonomy."""

    def __init__(self, failure: FailureRecord, evidence: dict[str, Any]) -> None:
        super().__init__(failure.observed)
        self.failure = failure
        self.evidence = evidence


class QualificationRunner:
    """Run frozen artifacts in one baseline then one candidate worker at a time."""

    def __init__(
        self,
        runner: Runner | None = None,
        source_root: Path | None = None,
        lock_verifier: LockVerifier | None = None,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.worker = DockerGpuWorker(self.runner)
        self._source_root_override = source_root
        self.source_root = Path()
        self.lock_verifier = lock_verifier or MatrixLocker(runner=self.runner)

    def run(self, specification_path: Path, destination: Path) -> QualificationOutcome:
        """Validate locks, run all declared cases, and publish complete evidence."""

        if destination.exists() or destination.is_symlink():
            raise InvalidInputError("refusing to overwrite qualification output")
        self.source_root = _qualification_project_root(
            specification_path,
            self._source_root_override,
        )
        specification = _load_specification(specification_path)
        matrix_path = _resolve_authored_path(specification_path, specification.environment_lock)
        matrix = _load_model(matrix_path, MatrixLock, "environment lock")
        if matrix.computed_sha256() != matrix.lock_sha256:
            raise InvalidInputError("environment lock self-hash differs")
        self.lock_verifier.verify(matrix)
        environments = {environment.id: environment for environment in matrix.environments}
        if set(environments) != {
            specification.baseline_environment_id,
            specification.candidate_environment_id,
        }:
            raise InvalidInputError("qualification and environment lock IDs differ")
        if specification.hardware_validity.selected_gpu_uuid != matrix.gpu_uuid:
            raise InvalidInputError("qualification and environment lock GPU UUIDs differ")
        corpus_root = _corpus_root(self.source_root, specification)
        corpus = _load_model(corpus_root / "corpus.lock.json", CorpusLock, "corpus lock")
        if corpus.id != specification.corpus_lock_id:
            raise InvalidInputError("qualification and corpus lock IDs differ")
        from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock

        reference_path = _resolve_authored_path(
            specification_path,
            specification.reference_environment_lock,
        )
        reference_environment = _load_model(
            reference_path,
            ReferenceEnvironmentLock,
            "reference environment lock",
        )
        if reference_environment.computed_sha256() != reference_environment.lock_sha256:
            raise InvalidInputError("reference environment lock self-hash differs")
        if corpus.reference_environment_sha256 != reference_environment.lock_sha256:
            raise InvalidInputError("corpus and reference environment lock identities differ")
        _verify_corpus(corpus_root, corpus)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        started = datetime.now(UTC)
        try:
            summary = self._execute(specification, matrix, environments, corpus_root, staging)
            _write_json(staging / "qualification-summary.json", summary)
            staging.replace(destination)
        except _TypedWorkerFailureError as error:
            summary = {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": _status((error.failure.code,)),
                "failure_codes": [error.failure.code.value],
                "started_at": started.isoformat(),
                "ended_at": datetime.now(UTC).isoformat(),
                "environment_lock_sha256": matrix.lock_sha256,
                "gpu_uuid": matrix.gpu_uuid,
                "stack_attribution": (
                    "Observed changes belong to the complete locked baseline and candidate stacks. "
                    "No result is attributed to TensorRT alone."
                ),
                "cases": [error.evidence],
                "failures": [error.failure.model_dump(mode="json")],
            }
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
        failure_records: list[FailureRecord] = []
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
            case_manifests = _materialize_case_manifests(
                specification,
                corpus_root,
                staging,
                precision,
                suffix,
                shape_map,
            )
            build_case_sha256 = case_manifests[sorted(case_manifests)[0]].manifest_sha256
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
                    build_case_sha256,
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
                        builds[environment_id][0],
                        case_manifests[shape_id].manifest_sha256,
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
                evidence["case_manifest"] = {
                    "path": (
                        Path("case-manifests") / f"tiny-transformer-{suffix}-{shape_id}.json"
                    ).as_posix(),
                    "sha256": case_manifests[shape_id].manifest_sha256,
                }
                if failure is not None:
                    failures.append(failure)
                    typed = evidence.get("typed_run_results")
                    if isinstance(typed, dict):
                        for environment_id in (
                            specification.baseline_environment_id,
                            specification.candidate_environment_id,
                        ):
                            run = typed.get(environment_id)
                            if not isinstance(run, dict) or run.get("failure") is None:
                                continue
                            failure_records.append(FailureRecord.model_validate(run["failure"]))
            memory = _memory_evidence(
                specification,
                builds[specification.baseline_environment_id],
                builds[specification.candidate_environment_id],
                runs[specification.baseline_environment_id],
                runs[specification.candidate_environment_id],
            )
            all_case_evidence.append({"precision": suffix, "memory": memory})
            memory_path = staging / f"memory-{suffix}.json"
            _write_json(memory_path, memory)
            for metric, outcome in (
                ("engine_bytes", memory["engine_bytes"]["outcome"]),
                ("device_memory", memory["device_memory"]["outcome"]),
            ):
                if outcome == GateOutcome.REGRESSION.value:
                    failures.append(FailureCode.MEMORY_REGRESSION)
                    failure_records.append(
                        _aggregate_failure_record(
                            code=FailureCode.MEMORY_REGRESSION,
                            phase=Phase.MEMORY,
                            environment_id=specification.candidate_environment_id,
                            precision=precision,
                            gate=f"memory.{metric}",
                            observed=json.dumps(memory[metric], allow_nan=False, sort_keys=True),
                            threshold=(
                                "the locked absolute and relative memory allowances must pass "
                                "across three confirmed builds"
                            ),
                            evidence_path=memory_path,
                            output_root=staging,
                        )
                    )
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
            performance_path = staging / f"performance-{suffix}.json"
            _write_json(performance_path, performance)
            performance_outcome = GateOutcome(performance["outcome"])
            if performance_outcome is GateOutcome.REGRESSION:
                failures.append(FailureCode.PERFORMANCE_REGRESSION)
                failure_records.append(
                    _aggregate_failure_record(
                        code=FailureCode.PERFORMANCE_REGRESSION,
                        phase=Phase.PERFORMANCE,
                        environment_id=specification.candidate_environment_id,
                        precision=precision,
                        gate="performance.weighted_paired_bootstrap",
                        observed=json.dumps(
                            {
                                "aggregate": performance["aggregate"],
                                "shapes": performance["shapes"],
                            },
                            allow_nan=False,
                            sort_keys=True,
                        ),
                        threshold=(
                            "the locked one-sided paired-bootstrap allowance must pass for "
                            "every weighted workload shape and the aggregate"
                        ),
                        evidence_path=performance_path,
                        output_root=staging,
                    )
                )
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
            "failures": [item.model_dump(mode="json") for item in failure_records],
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
        case_manifest_sha256: str,
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
                accepted_returncodes=(0, 1),
            )
            worker_build = _load_worker_result(result, WorkerBuildResult, "build")
            _verify_worker_command(worker_build.command, worker_build.command_sha256, command)
            failure = _worker_failure_record(
                worker_build.failure_code,
                worker_build.message,
                phase=Phase.BUILD,
                environment_id=environment.id,
                precision=precision,
                shape_id=None,
                result_path=result,
                output_root=staging,
            )
            manifest = adapt_worker_build(
                worker_build,
                BuildManifestAdapterContext(
                    id=f"{environment.id}-{precision}-build-{index}",
                    case_manifest_sha256=case_manifest_sha256,
                    environment_lock_sha256=matrix.lock_sha256,
                    failure=failure,
                ),
            )
            manifest_path = environment_output / f"build-{index}.manifest.json"
            _write_json(manifest_path, manifest.model_dump(mode="json"))
            record = {
                "environment_id": environment.id,
                "build_manifest": manifest.model_dump(mode="json"),
                "worker": _translate_container_paths(
                    worker_build.model_dump(mode="json"), staging, corpus_root
                ),
            }
            if failure is not None:
                raise _TypedWorkerFailureError(failure, record)
            builds.append(record)
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
        build: dict[str, Any],
        case_manifest_sha256: str,
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
        started_at = datetime.now(UTC)
        self.worker.run(
            image=environment.worker_image.canonical_reference,
            gpu_uuid=matrix.gpu_uuid,
            mounts=WorkerMounts(self.source_root, corpus_root, staging),
            command=command,
            timeout_seconds=900,
            accepted_returncodes=(0, 1),
        )
        ended_at = datetime.now(UTC)
        worker_run = _load_worker_result(result, WorkerCorrectnessResult, "correctness")
        _verify_worker_command(worker_run.command, worker_run.command_sha256, command)
        promoted_failure_code = _contextual_correctness_failure_code(
            worker_run,
            baseline=environment.id == specification.baseline_environment_id,
        )
        failure = _worker_failure_record(
            promoted_failure_code,
            worker_run.message,
            phase=Phase.CORRECTNESS,
            environment_id=environment.id,
            precision=precision,
            shape_id=shape.id,
            result_path=result,
            output_root=staging,
        )
        build_manifest = build["build_manifest"]
        translated = cast(
            dict[str, Any],
            _translate_container_paths(worker_run.model_dump(mode="json"), staging, corpus_root),
        )
        tolerance_stable = _worker_tolerance_stable(
            translated,
            staging,
            specification.determinism.tolerance,
        )
        hardware = HardwareObservation(
            gpu_uuid=matrix.gpu_uuid,
            driver=environment.probe.observed_driver,
            environment_lock_sha256=matrix.lock_sha256,
            valid=True,
            invalid_reasons=(),
        )
        promoted_worker_run = worker_run.model_copy(update={"failure_code": promoted_failure_code})
        run_manifest = adapt_worker_run(
            promoted_worker_run,
            RunResultAdapterContext(
                id=f"{environment.id}-{precision}-{shape.id}",
                case_manifest_sha256=case_manifest_sha256,
                build_manifest_sha256=sha256_bytes(canonical_json_bytes(build_manifest)),
                environment_lock_sha256=matrix.lock_sha256,
                hardware_sha256=sha256_bytes(
                    canonical_json_bytes(hardware.model_dump(mode="json"))
                ),
                hardware=hardware,
                started_at=started_at,
                ended_at=ended_at,
                serialized_engine_bytes=int(build_manifest["engine_bytes"]),
                engine_device_memory_bytes=int(build_manifest["engine_device_memory_bytes"]),
                determinism_tolerance_stable=tolerance_stable,
                failure=failure,
            ),
        )
        run_manifest_path = base / shape.id / "run-result.json"
        _write_json(run_manifest_path, run_manifest.model_dump(mode="json"))
        record = {
            "environment_id": environment.id,
            "run_result": run_manifest.model_dump(mode="json"),
            "worker": translated,
        }
        if failure is not None:
            raise _TypedWorkerFailureError(failure, record)
        return record

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
            source_inputs = {
                name: (
                    _corpus_root(self.source_root, specification)
                    / "inputs"
                    / f"tiny-transformer-{precision}"
                    / shape_id
                    / f"{name}.npy"
                )
                for name in shape.inputs
            }
            frozen_inputs = freeze_raw_inputs(
                source_inputs,
                staging / "benchmark-inputs" / precision / shape_id,
                f"/output/benchmark-inputs/{precision}/{shape_id}",
            )
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
                        inputs=frozen_inputs,
                    )
                    self._run_benchmark_worker(
                        specification,
                        matrix,
                        environment,
                        staging,
                        precondition_command.command,
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
                                "precondition": precondition_command.evidence(),
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
                        inputs=frozen_inputs,
                    )
                    self._run_benchmark_worker(
                        specification,
                        matrix,
                        environment,
                        staging,
                        command.command,
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
                            **command.evidence(),
                            "accepted": not after_reasons,
                            "rejection_reasons": list(after_reasons),
                            "idle": idle,
                            "idle_attempts": idle_attempts,
                            "before": before,
                            "after": after,
                            "precondition": precondition_command.evidence(),
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
                _corpus_root(self.source_root, specification),
                staging,
            ),
            command=command,
            timeout_seconds=max(120, measurement_milliseconds / 1000 + 60),
        )


def compare_stored_run(directory: Path) -> dict[str, Any]:
    """Validate and return the stored end-to-end summary without recomputation drift."""

    direct_summary = directory / "qualification-summary.json"
    public_summary = directory / "core-run" / "qualification-summary.json"
    published_result = directory / "results.json"
    direct_present = direct_summary.exists() or direct_summary.is_symlink()
    public_summary_present = public_summary.exists() or public_summary.is_symlink()
    published_result_present = published_result.exists() or published_result.is_symlink()
    if direct_present and (public_summary_present or published_result_present):
        raise InvalidInputError("qualification run layout is ambiguous")
    if direct_present:
        _require_regular_compare_artifact(direct_summary, "direct core summary")
        return _validate_stored_summary(_read_json(direct_summary))
    if public_summary_present or published_result_present:
        if not public_summary_present or not published_result_present:
            raise InvalidInputError(
                "public qualification output is incomplete; compare core-run explicitly "
                "for a legacy core-only result"
            )
        from upgrade_guard.publication import PublicationValidationError, validate_publication

        try:
            decision = validate_publication(directory)
        except PublicationValidationError as error:
            raise InvalidInputError("public qualification publication is invalid") from error
        return decision.results
    raise InvalidInputError("directory is not a recognized qualification run layout")


def _validate_stored_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete or internally contradictory core qualification summaries."""

    if summary.get("schema_version") != "upgradeguard.dev/qualification-summary/v1":
        raise InvalidInputError("run summary schema version is unsupported")
    if summary.get("status") not in {
        "passed",
        "failed",
        "inconclusive",
        "infrastructure_invalid",
    }:
        raise InvalidInputError("run summary status is invalid")
    failure_codes = _validated_failure_codes(summary.get("failure_codes"))
    if summary["status"] != _status(failure_codes):
        raise InvalidInputError("run summary status and failure codes differ")
    return summary


def _validated_failure_codes(value: object) -> tuple[FailureCode, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidInputError("run summary failure codes are invalid")
    try:
        failure_codes = tuple(FailureCode(item) for item in value)
    except ValueError as error:
        raise InvalidInputError("run summary failure codes are invalid") from error
    if len(failure_codes) != len(set(failure_codes)):
        raise InvalidInputError("run summary failure codes contain duplicates")
    return failure_codes


def _require_regular_compare_artifact(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise InvalidInputError(f"{label} must be a regular file")


def _corpus_root(source_root: Path, specification: QualificationSpec) -> Path:
    authored = specification.corpus_root
    if authored is None:
        path = source_root / ".upgrade-guard" / "corpora" / specification.corpus_lock_id
    else:
        relative = Path(authored)
        if relative.is_absolute() or ".." in relative.parts:
            raise InvalidInputError("qualification corpus root must be project-relative")
        path = source_root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise InvalidInputError("qualification corpus root is unavailable") from error
    source = source_root.resolve(strict=True)
    if not resolved.is_relative_to(source) or path.is_symlink() or not resolved.is_dir():
        raise InvalidInputError("qualification corpus root escaped the source tree")
    return resolved


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
    baseline_record = baseline
    candidate_record = candidate
    baseline = baseline_record["worker"]
    candidate = candidate_record["worker"]
    integrity_failure = _input_integrity_failure(baseline, candidate)
    if integrity_failure is not None:
        if baseline_record.get("run_result") and candidate_record.get("run_result"):
            return _typed_input_integrity_failure(
                baseline_record=baseline_record,
                candidate_record=candidate_record,
                precision=precision,
                shape_id=shape_id,
                output_root=output_root,
                failure_code=integrity_failure,
            )
        return (
            {
                "model": "tiny-transformer",
                "precision": precision,
                "shape_id": shape_id,
                "input_integrity": {
                    "baseline": baseline.get("input_integrity_stable"),
                    "candidate": candidate.get("input_integrity_stable"),
                },
                "failure_code": integrity_failure.value,
            },
            integrity_failure,
        )
    reference_metadata = _read_json_array(
        corpus_root / "reference" / f"tiny-transformer-{precision}-{shape_id}.json"
    )
    if not isinstance(reference_metadata, list) or len(reference_metadata) != 1:
        raise InvalidInputError("stored reference output schema is invalid")
    reference_contract = reference_metadata[0]
    reference_value = np.load(
        corpus_root
        / "reference"
        / f"tiny-transformer-{precision}-{shape_id}-{reference_contract['name']}.npy",
        allow_pickle=False,
    )
    if (
        str(reference_value.dtype) != reference_contract["dtype"]
        or list(reference_value.shape) != list(reference_contract["shape"])
        or sha256_bytes(reference_value.tobytes(order="C")) != reference_contract["sha256"]
        or reference_contract.get("repetitions") != 2
        or reference_contract.get("bitwise_deterministic") is not True
    ):
        raise InvalidInputError("stored reference output differs from its locked evidence")
    try:
        baseline_paths = _output_paths(baseline, "output", output_root)
    except InvalidInputError as error:
        if error.message != "worker output schema changed between repetitions":
            raise
        return _typed_correctness_precedence_failure(
            baseline_record=baseline_record,
            candidate_record=candidate_record,
            precision=precision,
            shape_id=shape_id,
            output_root=output_root,
            reference_contract=reference_contract,
            failure_code=FailureCode.CORPUS_INVALID,
            failed_gates=("baseline_output_schema",),
        )
    try:
        candidate_paths = _output_paths(candidate, "output", output_root)
    except InvalidInputError as error:
        if error.message != "worker output schema changed between repetitions":
            raise
        return _typed_correctness_precedence_failure(
            baseline_record=baseline_record,
            candidate_record=candidate_record,
            precision=precision,
            shape_id=shape_id,
            output_root=output_root,
            reference_contract=reference_contract,
            failure_code=FailureCode.OUTPUT_SCHEMA_CHANGED,
            failed_gates=("candidate_output_schema",),
        )
    baseline_values = tuple(np.load(path, allow_pickle=False) for path in baseline_paths)
    candidate_values = tuple(np.load(path, allow_pickle=False) for path in candidate_paths)
    baseline_determinism = summarize_determinism(
        baseline_values,
        (),
        specification.determinism.tolerance,
        input_hashes_stable=_repetition_input_hashes_stable(baseline),
    )
    candidate_determinism = summarize_determinism(
        candidate_values,
        (),
        specification.determinism.tolerance,
        input_hashes_stable=_repetition_input_hashes_stable(candidate),
    )
    comparison_index = next(
        (index for index, value in enumerate(baseline_values) if not np.all(np.isfinite(value))),
        next(
            (
                index
                for index, value in enumerate(candidate_values)
                if not np.all(np.isfinite(value))
            ),
            0,
        ),
    )
    numerical_policy = specification.numerical_policy(precision_mode)
    try:
        decision = decide_three_way(
            "output",
            reference_value,
            baseline_values[comparison_index],
            candidate_values[comparison_index],
            policy=numerical_policy,
        )
    except ThreeWayPrecedenceError as error:
        return _typed_correctness_precedence_failure(
            baseline_record=baseline_record,
            candidate_record=candidate_record,
            precision=precision,
            shape_id=shape_id,
            output_root=output_root,
            reference_contract=reference_contract,
            failure_code=error.failure_code,
            failed_gates=error.failed_gates,
            baseline_determinism=baseline_determinism.model_dump(mode="json"),
            candidate_determinism=candidate_determinism.model_dump(mode="json"),
        )
    failure = decision.failure_code
    baseline_failure = (
        FailureCode.CORPUS_INVALID
        if failure is FailureCode.CORPUS_INVALID
        or not baseline_determinism.tolerance_stable
        or (specification.determinism.require_bitwise and not baseline_determinism.bitwise_stable)
        else None
    )
    candidate_failure: FailureCode | None = None
    if failure in {
        FailureCode.NONFINITE_OUTPUT,
        FailureCode.NUMERICAL_REGRESSION,
        FailureCode.OUTPUT_SCHEMA_CHANGED,
    }:
        candidate_failure = failure
    elif failure is None and (
        not candidate_determinism.tolerance_stable
        or (specification.determinism.require_bitwise and not candidate_determinism.bitwise_stable)
    ):
        candidate_failure = FailureCode.NONDETERMINISM_REGRESSION
    if failure is None and baseline_failure is not None:
        failure = baseline_failure
    if failure is None and candidate_failure is not None:
        failure = candidate_failure
    baseline_run_result = _finalize_run_result(
        baseline_record,
        numerical=(decision.baseline_to_reference,),
        failure_code=baseline_failure,
        precision=precision,
        shape_id=shape_id,
        output_root=output_root,
    )
    candidate_run_result = _finalize_run_result(
        candidate_record,
        numerical=(decision.candidate_to_reference, decision.candidate_to_baseline),
        failure_code=candidate_failure,
        precision=precision,
        shape_id=shape_id,
        output_root=output_root,
    )
    return (
        {
            "model": "tiny-transformer",
            "precision": precision,
            "shape_id": shape_id,
            "reference": {
                "name": reference_contract["name"],
                "dtype": reference_contract["dtype"],
                "shape": reference_contract["shape"],
                "sha256": reference_contract["sha256"],
                "repetitions": reference_contract["repetitions"],
                "bitwise_deterministic": reference_contract["bitwise_deterministic"],
            },
            "baseline_to_reference": decision.baseline_to_reference.model_dump(mode="json"),
            "candidate_to_reference": decision.candidate_to_reference.model_dump(mode="json"),
            "candidate_to_baseline": decision.candidate_to_baseline.model_dump(mode="json"),
            "comparison_repetition": comparison_index,
            "baseline_determinism": baseline_determinism.model_dump(mode="json"),
            "candidate_determinism": candidate_determinism.model_dump(mode="json"),
            "worker_commands": {
                "baseline": {
                    "command": baseline["command"],
                    "command_sha256": baseline["command_sha256"],
                },
                "candidate": {
                    "command": candidate["command"],
                    "command_sha256": candidate["command_sha256"],
                },
            },
            "worker_memory_diagnostics": {
                "baseline": baseline.get("memory_diagnostics"),
                "candidate": candidate.get("memory_diagnostics"),
            },
            "typed_run_results": {
                "baseline": baseline_run_result,
                "candidate": candidate_run_result,
            },
            "failure_code": failure.value if failure else None,
        },
        failure,
    )


def _typed_correctness_precedence_failure(
    *,
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any],
    precision: str,
    shape_id: str,
    output_root: Path,
    reference_contract: dict[str, Any],
    failure_code: FailureCode,
    failed_gates: tuple[str, ...],
    baseline_determinism: dict[str, Any] | None = None,
    candidate_determinism: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], FailureCode]:
    """Retain exact typed run results when schema precedence precludes numerical metrics."""

    baseline_failure = failure_code if failure_code is FailureCode.CORPUS_INVALID else None
    candidate_failure = failure_code if baseline_failure is None else None
    baseline_run_result = _finalize_run_result(
        baseline_record,
        numerical=(),
        failure_code=baseline_failure,
        precision=precision,
        shape_id=shape_id,
        output_root=output_root,
    )
    candidate_run_result = _finalize_run_result(
        candidate_record,
        numerical=(),
        failure_code=candidate_failure,
        precision=precision,
        shape_id=shape_id,
        output_root=output_root,
    )
    baseline_worker = baseline_record["worker"]
    candidate_worker = candidate_record["worker"]
    evidence: dict[str, Any] = {
        "model": "tiny-transformer",
        "precision": precision,
        "shape_id": shape_id,
        "reference": {
            "name": reference_contract["name"],
            "dtype": reference_contract["dtype"],
            "shape": reference_contract["shape"],
            "sha256": reference_contract["sha256"],
            "repetitions": reference_contract["repetitions"],
            "bitwise_deterministic": reference_contract["bitwise_deterministic"],
        },
        "failed_gates": list(failed_gates),
        "worker_commands": {
            "baseline": {
                "command": baseline_worker["command"],
                "command_sha256": baseline_worker["command_sha256"],
            },
            "candidate": {
                "command": candidate_worker["command"],
                "command_sha256": candidate_worker["command_sha256"],
            },
        },
        "worker_memory_diagnostics": {
            "baseline": baseline_worker.get("memory_diagnostics"),
            "candidate": candidate_worker.get("memory_diagnostics"),
        },
        "typed_run_results": {
            "baseline": baseline_run_result,
            "candidate": candidate_run_result,
        },
        "failure_code": failure_code.value,
    }
    if baseline_determinism is not None:
        evidence["baseline_determinism"] = baseline_determinism
    if candidate_determinism is not None:
        evidence["candidate_determinism"] = candidate_determinism
    return evidence, failure_code


def _typed_input_integrity_failure(
    *,
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any],
    precision: str,
    shape_id: str,
    output_root: Path,
    failure_code: FailureCode,
) -> tuple[dict[str, Any], FailureCode]:
    """Promote input-integrity precedence into the same stable RunResult chain."""

    baseline_failure = failure_code if failure_code is FailureCode.CORPUS_INVALID else None
    candidate_failure = failure_code if baseline_failure is None else None
    baseline_run = _finalize_run_result(
        baseline_record,
        numerical=(),
        failure_code=baseline_failure,
        precision=precision,
        shape_id=shape_id,
        output_root=output_root,
    )
    candidate_run = _finalize_run_result(
        candidate_record,
        numerical=(),
        failure_code=candidate_failure,
        precision=precision,
        shape_id=shape_id,
        output_root=output_root,
    )
    return (
        {
            "model": "tiny-transformer",
            "precision": precision,
            "shape_id": shape_id,
            "input_integrity": {
                "baseline": baseline_record["worker"].get("input_integrity_stable"),
                "candidate": candidate_record["worker"].get("input_integrity_stable"),
            },
            "failed_gates": [
                "baseline_input_integrity"
                if baseline_failure is not None
                else "candidate_input_integrity"
            ],
            "typed_run_results": {
                "baseline": baseline_run,
                "candidate": candidate_run,
            },
            "failure_code": failure_code.value,
        },
        failure_code,
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
    specification: QualificationSpec,
    baseline_builds: list[dict[str, Any]],
    candidate_builds: list[dict[str, Any]],
    baseline_runs: dict[str, dict[str, Any]],
    candidate_runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_engine = tuple(int(item["build_manifest"]["engine_bytes"]) for item in baseline_builds)
    candidate_engine = tuple(
        int(item["build_manifest"]["engine_bytes"]) for item in candidate_builds
    )
    baseline_device = tuple(
        int(item["build_manifest"]["engine_device_memory_bytes"]) for item in baseline_builds
    )
    candidate_device = tuple(
        int(item["build_manifest"]["engine_device_memory_bytes"]) for item in candidate_builds
    )
    return {
        "effective_policy": specification.memory.model_dump(mode="json"),
        "engine_bytes": _jsonable(
            asdict(
                confirmed_memory_gate(
                    baseline_engine,
                    candidate_engine,
                    fixed_allowance_bytes=specification.memory.engine_bytes_absolute,
                    proportional_allowance=specification.memory.engine_bytes_relative,
                )
            )
        ),
        "device_memory": _jsonable(
            asdict(
                confirmed_memory_gate(
                    baseline_device,
                    candidate_device,
                    fixed_allowance_bytes=specification.memory.device_memory_absolute,
                    proportional_allowance=specification.memory.device_memory_relative,
                )
            )
        ),
        "measurement_sources": {
            "baseline": _memory_sources(baseline_builds, baseline_runs),
            "candidate": _memory_sources(candidate_builds, candidate_runs),
        },
    }


def _aggregate_failure_record(
    *,
    code: FailureCode,
    phase: Phase,
    environment_id: str,
    precision: PrecisionMode,
    gate: str,
    observed: str,
    threshold: str,
    evidence_path: Path,
    output_root: Path,
) -> FailureRecord:
    """Bind an aggregate decision to its exact retained machine evidence."""

    reference = ArtifactReference(
        path=evidence_path.relative_to(output_root).as_posix(),
        sha256=sha256_file(evidence_path),
        bytes=evidence_path.stat().st_size,
        media_type="application/json",
    )
    predicate = {
        "code": code.value,
        "phase": phase.value,
        "environment_id": environment_id,
        "model_id": "tiny-transformer",
        "precision": precision.value,
        "gate": gate,
        "observed": observed,
        "threshold": threshold,
        "evidence": reference.sha256,
    }
    return FailureRecord(
        code=code,
        phase=phase,
        environment_id=environment_id,
        model_id="tiny-transformer",
        precision=precision,
        shape_id=None,
        input_fixture_id="deterministic-numeric",
        output_name=None,
        gate=gate,
        observed=observed,
        threshold=threshold,
        evidence=(reference,),
        signature_sha256=sha256_bytes(canonical_json_bytes(predicate)),
    )


def _memory_sources(
    builds: list[dict[str, Any]], runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "serialized_engine_bytes": [int(item["build_manifest"]["engine_bytes"]) for item in builds],
        "engine_reported_device_memory_bytes": [
            int(item["build_manifest"]["engine_device_memory_bytes"]) for item in builds
        ],
        "builder_diagnostics": [item["worker"].get("memory_diagnostics") for item in builds],
        "builds": [
            {
                "environment_id": item["environment_id"],
                "manifest": item["build_manifest"],
            }
            for item in builds
        ],
        "execution_context_diagnostics_by_shape": {
            shape: run["worker"].get("memory_diagnostics") for shape, run in sorted(runs.items())
        },
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


def _qualification_project_root(specification_path: Path, explicit: Path | None) -> Path:
    """Resolve a complete checkout instead of assuming an editable package layout."""

    if explicit is not None:
        candidate = explicit.resolve(strict=True)
        if not candidate.is_dir() or candidate.is_symlink():
            raise InvalidInputError("qualification project root must be a real directory")
        return candidate
    specification = specification_path.resolve(strict=True)
    for candidate in (specification.parent, *specification.parents):
        if (candidate / "BUILD_PLAN.md").is_file() and (candidate / "src").is_dir():
            return candidate
    raise InvalidInputError(
        "could not locate a complete qualification project; pass an explicit project root"
    )


def _verify_corpus(root: Path, lock: CorpusLock) -> None:
    excluded = {root / "corpus.lock.json"}
    materializer = root / "materializer.json"
    if materializer.is_file() and not materializer.is_symlink():
        value = _read_json(materializer)
        identity = value.get("materializer_sha256")
        if (
            not isinstance(identity, str)
            or not identity.startswith("sha256:")
            or root.name != identity.removeprefix("sha256:")
        ):
            raise InvalidInputError("corpus materializer identity differs from its path")
        excluded.add(materializer)
    expected = {artifact.path: artifact for artifact in lock.artifacts}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    }
    if observed != set(expected):
        raise InvalidInputError("materialized corpus inventory differs from lock")
    for relative, artifact in expected.items():
        path = root / relative
        if path.stat().st_size != artifact.bytes or sha256_file(path) != artifact.sha256:
            raise InvalidInputError("materialized corpus artifact differs from lock")


def _load_worker_result[TWorker: StrictModel](
    path: Path, model: type[TWorker], phase: str
) -> TWorker:
    """Load strict worker JSON, reserving infrastructure for malformed transport."""

    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise InfrastructureError(f"worker {phase} result is invalid") from error


def _verify_worker_command(
    observed: tuple[str, ...], observed_sha256: str, expected: tuple[str, ...]
) -> None:
    if observed != expected or observed_sha256 != command_sha256(expected):
        raise InfrastructureError("worker command evidence differs from the host invocation")


def _worker_failure_record(
    code: FailureCode | None,
    message: str | None,
    *,
    phase: Phase,
    environment_id: str,
    precision: str,
    shape_id: str | None,
    result_path: Path,
    output_root: Path,
) -> FailureRecord | None:
    if code is None:
        if message is not None:
            raise InfrastructureError("passing worker result unexpectedly contains an error")
        return None
    if message is None:
        raise InfrastructureError("failed worker result omitted its typed message")
    try:
        relative = result_path.resolve(strict=True).relative_to(output_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise InfrastructureError("worker failure evidence escaped the run directory") from error
    evidence = ArtifactReference(
        path=relative.as_posix(),
        sha256=sha256_file(result_path),
        bytes=result_path.stat().st_size,
        media_type="application/json",
    )
    predicate = {
        "code": code.value,
        "phase": phase.value,
        "environment_id": environment_id,
        "precision": precision,
        "shape_id": shape_id,
        "gate": f"worker_{phase.value}",
        "observed": message,
        "evidence_sha256": evidence.sha256,
    }
    return FailureRecord(
        code=code,
        phase=phase,
        environment_id=environment_id,
        model_id="tiny-transformer",
        precision=(PrecisionMode.FP32 if precision == "fp32" else PrecisionMode.EXPLICIT_FP16),
        shape_id=shape_id,
        input_fixture_id="deterministic-numeric" if shape_id is not None else None,
        output_name=None,
        gate=f"worker_{phase.value}",
        observed=message,
        threshold="the typed worker phase must complete successfully",
        evidence=(evidence,),
        signature_sha256=sha256_bytes(canonical_json_bytes(predicate)),
    )


def _contextual_correctness_failure_code(
    worker: WorkerCorrectnessResult,
    *,
    baseline: bool,
) -> FailureCode | None:
    """Promote baseline execution-invalid evidence as an unusable corpus observation."""

    if not baseline:
        return worker.failure_code
    if worker.input_integrity_stable is False or worker.failure_code in {
        FailureCode.EXECUTION_FAILED,
        FailureCode.NONFINITE_OUTPUT,
    }:
        return FailureCode.CORPUS_INVALID
    return worker.failure_code


def _materialize_case_manifests(
    specification: QualificationSpec,
    corpus_root: Path,
    staging: Path,
    precision: PrecisionMode,
    suffix: str,
    shape_map: dict[str, ConcreteShape],
) -> dict[str, CaseManifest]:
    """Retain one self-hashed typed case manifest for every concrete shape."""

    try:
        corpus_lock = CorpusLock.model_validate_json(
            (corpus_root / "corpus.lock.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise InvalidInputError("corpus lock is invalid for case manifests") from error
    artifacts = {artifact.path: artifact for artifact in corpus_lock.artifacts}
    model_relative = f"models/tiny-transformer-{suffix}.onnx"
    model_artifact = artifacts.get(model_relative)
    if model_artifact is None:
        raise InvalidInputError("case model is absent from the corpus lock")
    model_path = corpus_root / model_relative
    try:
        model = onnx.load_model(model_path, load_external_data=False)
    except (OSError, ValueError) as error:
        raise InvalidInputError("case model cannot be inspected") from error
    if len(model.opset_import) != 1:
        raise InvalidInputError("case model must declare one default ONNX opset")
    output_root = staging / "case-manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, CaseManifest] = {}
    for shape_id, shape in sorted(shape_map.items()):
        input_artifacts: list[ArtifactReference] = []
        input_contracts: list[TensorContract] = []
        input_dtypes: dict[str, str] = {}
        for name in sorted(shape.inputs):
            relative = f"inputs/tiny-transformer-{suffix}/{shape_id}/{name}.npy"
            artifact = artifacts.get(relative)
            if artifact is None:
                raise InvalidInputError(f"case input is absent from the corpus lock: {relative}")
            value = np.load(corpus_root / relative, allow_pickle=False)
            dtype = _tensor_dtype(str(value.dtype))
            input_contracts.append(TensorContract(name=name, dtype=dtype, shape=tuple(value.shape)))
            input_dtypes[name] = dtype
            input_artifacts.append(
                ArtifactReference(
                    path=artifact.path,
                    sha256=artifact.sha256,
                    bytes=artifact.bytes,
                    media_type=artifact.media_type,
                )
            )
        metadata_relative = f"reference/tiny-transformer-{suffix}-{shape_id}.json"
        try:
            metadata = json.loads((corpus_root / metadata_relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidInputError("case reference metadata is invalid") from error
        if not isinstance(metadata, list) or not metadata:
            raise InvalidInputError("case reference metadata has no outputs")
        output_contracts = tuple(
            TensorContract(
                name=str(item["name"]),
                dtype=_tensor_dtype(str(item["dtype"])),
                shape=tuple(int(dimension) for dimension in item["shape"]),
            )
            for item in metadata
        )
        zero = "sha256:" + ("0" * 64)
        manifest = CaseManifest(
            api_version="upgradeguard.dev/v1alpha1",
            kind="CaseManifest",
            id=f"tiny-transformer-{suffix}-{shape_id}",
            model_id=f"tiny-transformer-{suffix}",
            source=SourceAttribution(
                name="Project-owned deterministic tiny transformer",
                source_url="https://github.com/uday1o1/tensorrt-upgrade-guard",
                source_revision=model_artifact.sha256,
                license_name="Apache-2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
                redistribution_allowed=True,
            ),
            model=ArtifactReference(
                path=model_artifact.path,
                sha256=model_artifact.sha256,
                bytes=model_artifact.bytes,
                media_type=model_artifact.media_type,
            ),
            opset=int(model.opset_import[0].version),
            ir_version=int(model.ir_version),
            exporter_environment_sha256=corpus_lock.reference_environment_sha256,
            precision=precision,
            profile_id=specification.optimization_profiles[0].id,
            shape_id=shape_id,
            inputs=tuple(input_contracts),
            input_fixtures=tuple(input_artifacts),
            outputs=output_contracts,
            reference_runner="onnxruntime_cpu",
            reference_environment_sha256=corpus_lock.reference_environment_sha256,
            reference_capability=ReferenceCapability(
                supported=True,
                execution_provider="CPUExecutionProvider",
                observed_input_dtypes=input_dtypes,
                observed_output_dtypes={item.name: item.dtype for item in output_contracts},
            ),
            numerical=specification.numerical_policy(precision),
            determinism=specification.determinism,
            workload_weight=specification.performance.shape_weights[shape_id],
            semantic_policy={"comparison": "elementwise"},
            manifest_sha256=zero,
        )
        manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
        manifest = adapt_case_manifest(manifest)
        _write_json(
            output_root / f"tiny-transformer-{suffix}-{shape_id}.json",
            manifest.model_dump(mode="json"),
        )
        manifests[shape_id] = manifest
    return manifests


def _tensor_dtype(value: str) -> TensorDtype:
    allowed = {"float16", "float32", "float64", "int8", "int32", "int64", "bool"}
    if value not in allowed:
        raise InvalidInputError(f"case tensor dtype is unsupported: {value}")
    return cast(TensorDtype, value)


def _worker_tolerance_stable(run: dict[str, Any], output_root: Path, tolerance: Any) -> bool:
    repetitions = run.get("repetitions")
    if not isinstance(repetitions, list) or not repetitions:
        return False
    first_outputs = repetitions[0].get("outputs")
    if not isinstance(first_outputs, list) or not first_outputs:
        return False
    names = [item.get("name") for item in first_outputs if isinstance(item, dict)]
    if not all(isinstance(name, str) for name in names) or len(names) != len(set(names)):
        return False
    input_hashes_stable = _repetition_input_hashes_stable(run)
    return all(
        summarize_determinism(
            tuple(
                np.load(path, allow_pickle=False) for path in _output_paths(run, name, output_root)
            ),
            (),
            tolerance,
            input_hashes_stable=input_hashes_stable,
        ).tolerance_stable
        for name in cast(list[str], names)
    )


def _repetition_input_hashes_stable(run: dict[str, Any]) -> bool:
    """Compare each named input with its own frozen identity for every repetition."""

    expected = run.get("input_sha256")
    repetitions = run.get("repetitions")
    if not isinstance(expected, dict) or not expected or not isinstance(repetitions, list):
        return False
    for repetition in repetitions:
        if not isinstance(repetition, dict) or not isinstance(repetition.get("inputs"), list):
            return False
        inputs = repetition["inputs"]
        if len(inputs) != len(expected):
            return False
        observed: dict[str, str] = {}
        for item in inputs:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                return False
            name = item["name"]
            if name in observed or name not in expected or item.get("stable") is not True:
                return False
            if item.get("source_sha256") != expected[name] or item.get(
                "host_value_sha256"
            ) != item.get("device_value_sha256"):
                return False
            observed[name] = expected[name]
        if observed != expected:
            return False
    return bool(repetitions)


def _input_integrity_failure(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> FailureCode | None:
    """Apply input-integrity precedence before numerical or determinism gates."""

    if baseline.get("input_integrity_stable") is not True:
        return FailureCode.CORPUS_INVALID
    if candidate.get("input_integrity_stable") is not True:
        return FailureCode.EXECUTION_FAILED
    return None


def _finalize_run_result(
    record: dict[str, Any],
    *,
    numerical: tuple[Any, ...],
    failure_code: FailureCode | None,
    precision: str,
    shape_id: str,
    output_root: Path,
) -> dict[str, Any]:
    """Attach comparison evidence and final status to the adapted worker run."""

    payload = dict(record["run_result"])
    payload["numerical"] = [item.model_dump(mode="json") for item in numerical]
    if payload.get("determinism") is not None:
        payload["determinism"]["nonfinite_observed"] = any(
            item.reference_nonfinite_count or item.candidate_nonfinite_count for item in numerical
        )
    failure = None
    if failure_code is not None:
        evidence = tuple(
            ArtifactReference.model_validate(item) for item in payload["output_artifacts"]
        )
        phase = (
            Phase.DETERMINISM
            if failure_code is FailureCode.NONDETERMINISM_REGRESSION
            else Phase.CORRECTNESS
        )
        predicate = {
            "code": failure_code.value,
            "phase": phase.value,
            "environment_id": record["environment_id"],
            "precision": precision,
            "shape_id": shape_id,
            "evidence": [item.sha256 for item in evidence],
        }
        failure = FailureRecord(
            code=failure_code,
            phase=phase,
            environment_id=record["environment_id"],
            model_id="tiny-transformer",
            precision=(PrecisionMode.FP32 if precision == "fp32" else PrecisionMode.EXPLICIT_FP16),
            shape_id=shape_id,
            input_fixture_id="deterministic-numeric",
            output_name="output",
            gate=(
                "determinism"
                if failure_code is FailureCode.NONDETERMINISM_REGRESSION
                else "output_schema"
                if failure_code is FailureCode.OUTPUT_SCHEMA_CHANGED
                else "finite_outputs"
                if failure_code is FailureCode.NONFINITE_OUTPUT
                else "three_way_numerical"
            ),
            observed=failure_code.value,
            threshold="the locked case policy must pass",
            evidence=evidence,
            signature_sha256=sha256_bytes(canonical_json_bytes(predicate)),
        )
    payload["status"] = status_for_failure(failure_code).value
    payload["failure"] = failure.model_dump(mode="json") if failure is not None else None
    finalized = RunResult.model_validate(payload)
    dumped = finalized.model_dump(mode="json")
    record["run_result"] = dumped
    path = output_root / record["environment_id"] / precision / shape_id / "run-result.json"
    _write_json(path, dumped)
    return dumped


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InfrastructureError("worker omitted valid machine JSON") from error
    if not isinstance(value, dict):
        raise InfrastructureError("worker machine result must be an object")
    return value


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidInputError("stored reference evidence is invalid") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise InvalidInputError("stored reference evidence must be an object array")
    return cast(list[dict[str, Any]], value)


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
