"""Promote extended-suite worker evidence into stable typed artifact chains."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from upgrade_guard.classify import status_for_failure
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import (
    canonical_json_bytes,
    model_sha256,
    sha256_bytes,
    sha256_file,
)
from upgrade_guard.contracts.build import (
    BuildManifestAdapterContext,
    WorkerBuildResult,
    adapt_worker_build,
)
from upgrade_guard.contracts.case import CaseManifest, ReferenceCapability, adapt_case_manifest
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord, PrecisionMode
from upgrade_guard.contracts.environment import EnvironmentLock, MatrixLock
from upgrade_guard.contracts.extended import (
    ExtendedCorpusCase,
    ExtendedCorpusManifest,
    ExtendedCorpusModel,
    ExtendedInvocationManifest,
    PluginBuildProvenance,
)
from upgrade_guard.contracts.qualification import QualificationSpec
from upgrade_guard.contracts.results import (
    HardwareObservation,
    NumericalSummary,
    RunResult,
    RunResultAdapterContext,
    WorkerCorrectnessResult,
    adapt_worker_run,
)
from upgrade_guard.worker.common import write_json_atomic

ExtendedSuite = Literal["plugin", "mobilenet"]
_ZERO_SHA256 = "sha256:" + "0" * 64
_PLUGIN_SOURCES = (
    "CMakeLists.txt",
    "cpp/plugin/register_plugin.cpp",
    "cpp/plugin/residual_rmsnorm_creator.cpp",
    "cpp/plugin/residual_rmsnorm_plugin.cpp",
    "cpp/plugin/residual_rmsnorm_plugin.hpp",
    "cpp/kernels/residual_rmsnorm_launch.hpp",
    "cpp/kernels/residual_rmsnorm_naive.cu",
    "cpp/kernels/residual_rmsnorm_optimized.cu",
)


@dataclass(frozen=True)
class ExtendedPromotionContext:
    """Verified immutable inputs used to promote one extended suite."""

    suite: ExtendedSuite
    corpus_root: Path
    runs_root: Path
    specification: QualificationSpec
    matrix: MatrixLock
    corpus: ExtendedCorpusManifest
    invocation: ExtendedInvocationManifest
    invocation_artifact: ArtifactReference

    def environment(self, environment_id: str) -> EnvironmentLock:
        matches = tuple(item for item in self.matrix.environments if item.id == environment_id)
        if len(matches) != 1:
            raise RuntimeError(f"unknown extended environment: {environment_id}")
        return matches[0]


def prepare_extended_promotion(
    *,
    suite: ExtendedSuite,
    project_root: Path,
    corpus_root: Path,
    runs_root: Path,
    specification_path: Path,
    matrix_lock_path: Path,
    source_commit_file: Path,
    plugin_build_root: Path | None = None,
    plugin_build_log: Path | None = None,
) -> ExtendedPromotionContext:
    """Verify frozen inputs and atomically retain one invocation manifest."""

    project = project_root.resolve(strict=True)
    corpus = corpus_root.resolve(strict=True)
    runs = runs_root.resolve(strict=True)
    specification = QualificationSpec.model_validate(_yaml_object(specification_path))
    matrix = MatrixLock.model_validate(_json_object(matrix_lock_path))
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise RuntimeError("extended matrix lock self-hash differs")
    environment_ids = tuple(item.id for item in matrix.environments)
    expected_ids = (
        specification.baseline_environment_id,
        specification.candidate_environment_id,
    )
    if environment_ids != expected_ids:
        raise RuntimeError("extended matrix environment order differs from specification")
    source_commit = source_commit_file.read_text(encoding="utf-8").strip()
    lock_name = f"{suite}-corpus.lock.json"
    corpus_lock_path = corpus / lock_name
    corpus_lock = _json_object(corpus_lock_path)
    extended = corpus_lock.get("extended_manifest")
    if not isinstance(extended, dict):
        raise RuntimeError("extended corpus lock has no typed manifest identity")
    manifest_path = _safe_relative_file(corpus, extended.get("path"))
    if extended.get("sha256") != sha256_file(manifest_path):
        raise RuntimeError("extended corpus manifest file hash differs")
    corpus_manifest = ExtendedCorpusManifest.model_validate(_json_object(manifest_path))
    if (
        corpus_manifest.suite != suite
        or corpus_manifest.computed_sha256() != corpus_manifest.manifest_sha256
        or extended.get("manifest_sha256") != corpus_manifest.manifest_sha256
    ):
        raise RuntimeError("extended corpus manifest identity differs")
    _verify_corpus_artifacts(corpus, corpus_manifest)
    plugin_builds: tuple[PluginBuildProvenance, ...] = ()
    if suite == "plugin":
        if plugin_build_root is None or plugin_build_log is None:
            raise RuntimeError("plugin promotion requires exact compile provenance")
        plugin_builds = tuple(
            _plugin_build_provenance(
                project=project,
                runs=runs,
                build_root=plugin_build_root,
                build_log=plugin_build_log,
                environment=environment,
            )
            for environment in matrix.environments
        )
    invocation = ExtendedInvocationManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ExtendedInvocationManifest",
        suite=suite,
        source_git_commit=source_commit,
        matrix_lock_sha256=matrix.lock_sha256,
        specification_sha256=sha256_file(specification_path),
        corpus_lock_sha256=sha256_file(corpus_lock_path),
        corpus_manifest_sha256=corpus_manifest.manifest_sha256,
        environment_ids=environment_ids,
        plugin_builds=plugin_builds,
        manifest_sha256=_ZERO_SHA256,
    )
    invocation = invocation.model_copy(update={"manifest_sha256": invocation.computed_sha256()})
    invocation_path = runs / "invocation-manifest.json"
    write_json_atomic(invocation_path, invocation.model_dump(mode="json"))
    invocation_artifact = _artifact(runs, invocation_path, "application/json")
    return ExtendedPromotionContext(
        suite=suite,
        corpus_root=corpus,
        runs_root=runs,
        specification=specification,
        matrix=matrix,
        corpus=corpus_manifest,
        invocation=invocation,
        invocation_artifact=invocation_artifact,
    )


def promote_extended_case(
    context: ExtendedPromotionContext,
    *,
    environment_id: str,
    precision: PrecisionMode,
    case_id: str,
    build_result_path: Path,
    correctness_result_path: Path,
    output_root: Path,
    numerical: tuple[NumericalSummary, ...],
    determinism_tolerance_stable: bool,
    failure: FailureRecord | None,
) -> dict[str, object]:
    """Atomically write a self-valid case, build, and run chain."""

    case, model = _case_model(context.corpus, precision, case_id)
    policy = context.specification.numerical_policy(precision)
    manifest = CaseManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="CaseManifest",
        id=case.id,
        model_id=model.model_id,
        source=model.source,
        model=model.artifact,
        opset=model.opset,
        ir_version=model.ir_version,
        exporter_environment_sha256=context.corpus.reference_environment_sha256,
        precision=precision,
        profile_id=case.profile_id,
        shape_id=case.shape_id,
        inputs=case.inputs,
        input_fixtures=case.input_fixtures,
        outputs=case.outputs,
        reference_runner=model.reference_runner,
        reference_environment_sha256=context.corpus.reference_environment_sha256,
        reference_capability=ReferenceCapability(
            supported=True,
            execution_provider=(
                "CPUExecutionProvider"
                if model.reference_runner == "onnxruntime_cpu"
                else "project_formula"
            ),
            observed_input_dtypes={item.name: item.dtype for item in case.inputs},
            observed_output_dtypes={item.name: item.dtype for item in case.outputs},
        ),
        numerical=policy,
        determinism=context.specification.determinism,
        workload_weight=case.workload_weight,
        semantic_policy=model.semantic_policy,
        manifest_sha256=_ZERO_SHA256,
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
    manifest = adapt_case_manifest(manifest)
    destination = output_root.resolve()
    if not destination.is_relative_to(context.runs_root) or output_root.is_symlink():
        raise RuntimeError("extended stable artifact destination escaped runs root")
    output_root.mkdir(parents=True, exist_ok=True)
    case_path = output_root / "case-manifest.json"
    write_json_atomic(case_path, manifest.model_dump(mode="json"))

    worker_build = WorkerBuildResult.model_validate(_json_object(build_result_path))
    if worker_build.status != "passed" or worker_build.model is None or worker_build.engine is None:
        raise RuntimeError("extended promotion requires a passing worker build")
    if (
        worker_build.command_sha256 != command_sha256(worker_build.command)
        or worker_build.model.sha256 != model.artifact.sha256
    ):
        raise RuntimeError("extended worker build identity differs from frozen inputs")
    plugin = next(
        (
            item
            for item in context.invocation.plugin_builds
            if item.environment_id == environment_id
        ),
        None,
    )
    build = adapt_worker_build(
        worker_build,
        BuildManifestAdapterContext(
            id=f"{context.suite}-{environment_id}-{case.id}-build",
            case_manifest_sha256=manifest.manifest_sha256,
            environment_lock_sha256=context.matrix.lock_sha256,
            plugin_source_sha256=plugin.source_inventory_sha256 if plugin else None,
            plugin_binary=plugin.binary if plugin else None,
            plugin_compile_command=plugin.build_command if plugin else None,
            plugin_build_log=plugin.build_log if plugin else None,
        ),
    )
    build_path = output_root / "build.manifest.json"
    write_json_atomic(build_path, build.model_dump(mode="json"))
    build_sha256 = sha256_file(build_path)

    worker_run = WorkerCorrectnessResult.model_validate(_json_object(correctness_result_path))
    if worker_run.status == "passed":
        if worker_run.engine_sha256 != worker_build.engine.sha256:
            raise RuntimeError("extended promotion run belongs to a different engine")
    elif failure is None or worker_run.failure_code is not failure.code:
        raise RuntimeError("extended failed run and host failure classification differ")
    if worker_run.command_sha256 != command_sha256(worker_run.command):
        raise RuntimeError("extended worker run command identity differs")
    environment = context.environment(environment_id)
    hardware = HardwareObservation(
        gpu_uuid=environment.probe.gpu.uuid,
        driver=environment.probe.observed_driver,
        environment_lock_sha256=context.matrix.lock_sha256,
        valid=True,
        invalid_reasons=(),
    )
    run = adapt_worker_run(
        worker_run,
        RunResultAdapterContext(
            id=f"{context.suite}-{environment_id}-{case.id}-run",
            case_manifest_sha256=manifest.manifest_sha256,
            build_manifest_sha256=build_sha256,
            environment_lock_sha256=context.matrix.lock_sha256,
            hardware_sha256=model_sha256(hardware),
            hardware=hardware,
            started_at=datetime.fromtimestamp(worker_run.started_unix_seconds, tz=UTC),
            ended_at=datetime.fromtimestamp(worker_run.ended_unix_seconds, tz=UTC),
            serialized_engine_bytes=worker_build.engine.bytes,
            engine_device_memory_bytes=worker_build.engine.device_memory_bytes,
            determinism_tolerance_stable=determinism_tolerance_stable,
            numerical=numerical,
            failure=failure if worker_run.status == "failed" else None,
        ),
    )
    if run.determinism is not None:
        run = RunResult.model_validate(
            run.model_dump(mode="python", exclude={"determinism"})
            | {
                "determinism": run.determinism.model_copy(
                    update={
                        "nonfinite_observed": any(
                            summary.reference_nonfinite_count or summary.candidate_nonfinite_count
                            for summary in numerical
                        )
                    }
                )
            }
        )
    if failure is not None:
        run = RunResult.model_validate(
            run.model_dump(
                mode="python",
                exclude={"status", "failure"},
            )
            | {"status": status_for_failure(failure.code), "failure": failure}
        )
    run_path = output_root / "run-result.json"
    write_json_atomic(run_path, run.model_dump(mode="json"))
    return {
        "case_manifest": _artifact(context.runs_root, case_path, "application/json").model_dump(
            mode="json"
        ),
        "case_manifest_sha256": manifest.manifest_sha256,
        "build_manifest": _artifact(context.runs_root, build_path, "application/json").model_dump(
            mode="json"
        ),
        "run_result": _artifact(context.runs_root, run_path, "application/json").model_dump(
            mode="json"
        ),
    }


def promote_extended_build_failure(
    context: ExtendedPromotionContext,
    *,
    environment_id: str,
    precision: PrecisionMode,
    case_id: str,
    build_result_path: Path,
    output_root: Path,
    failure: FailureRecord,
) -> dict[str, object]:
    """Promote a strict failed worker build without inventing a run artifact."""

    case, model = _case_model(context.corpus, precision, case_id)
    policy = context.specification.numerical_policy(precision)
    manifest = CaseManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="CaseManifest",
        id=case.id,
        model_id=model.model_id,
        source=model.source,
        model=model.artifact,
        opset=model.opset,
        ir_version=model.ir_version,
        exporter_environment_sha256=context.corpus.reference_environment_sha256,
        precision=precision,
        profile_id=case.profile_id,
        shape_id=case.shape_id,
        inputs=case.inputs,
        input_fixtures=case.input_fixtures,
        outputs=case.outputs,
        reference_runner=model.reference_runner,
        reference_environment_sha256=context.corpus.reference_environment_sha256,
        reference_capability=ReferenceCapability(
            supported=True,
            execution_provider=(
                "CPUExecutionProvider"
                if model.reference_runner == "onnxruntime_cpu"
                else "project_formula"
            ),
            observed_input_dtypes={item.name: item.dtype for item in case.inputs},
            observed_output_dtypes={item.name: item.dtype for item in case.outputs},
        ),
        numerical=policy,
        determinism=context.specification.determinism,
        workload_weight=case.workload_weight,
        semantic_policy=model.semantic_policy,
        manifest_sha256=_ZERO_SHA256,
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
    manifest = adapt_case_manifest(manifest)
    destination = output_root.resolve()
    if not destination.is_relative_to(context.runs_root) or output_root.is_symlink():
        raise RuntimeError("extended failed-build destination escaped runs root")
    output_root.mkdir(parents=True, exist_ok=True)
    case_path = output_root / "case-manifest.json"
    write_json_atomic(case_path, manifest.model_dump(mode="json"))

    worker_build = WorkerBuildResult.model_validate(_json_object(build_result_path))
    if worker_build.status != "failed" or worker_build.failure_code is not failure.code:
        raise RuntimeError("extended failed build and host failure classification differ")
    if worker_build.command_sha256 != command_sha256(worker_build.command):
        raise RuntimeError("extended failed build command identity differs")
    if worker_build.model is not None and worker_build.model.sha256 != model.artifact.sha256:
        raise RuntimeError("extended failed build model differs from frozen inputs")
    plugin = next(
        (
            item
            for item in context.invocation.plugin_builds
            if item.environment_id == environment_id
        ),
        None,
    )
    build = adapt_worker_build(
        worker_build,
        BuildManifestAdapterContext(
            id=f"{context.suite}-{environment_id}-{case.id}-build",
            case_manifest_sha256=manifest.manifest_sha256,
            environment_lock_sha256=context.matrix.lock_sha256,
            plugin_source_sha256=plugin.source_inventory_sha256 if plugin else None,
            plugin_binary=plugin.binary if plugin else None,
            plugin_compile_command=plugin.build_command if plugin else None,
            plugin_build_log=plugin.build_log if plugin else None,
            failure=failure,
        ),
    )
    build_path = output_root / "build.manifest.json"
    write_json_atomic(build_path, build.model_dump(mode="json"))
    return {
        "case_manifest": _artifact(context.runs_root, case_path, "application/json").model_dump(
            mode="json"
        ),
        "case_manifest_sha256": manifest.manifest_sha256,
        "build_manifest": _artifact(context.runs_root, build_path, "application/json").model_dump(
            mode="json"
        ),
        "run_result": None,
    }


def _case_model(
    corpus: ExtendedCorpusManifest,
    precision: PrecisionMode,
    case_id: str,
) -> tuple[ExtendedCorpusCase, ExtendedCorpusModel]:
    cases = tuple(
        case for case in corpus.cases if case.precision is precision and case.id == case_id
    )
    if len(cases) != 1:
        raise RuntimeError(f"extended corpus case is missing or ambiguous: {precision}/{case_id}")
    models = tuple(
        model
        for model in corpus.models
        if model.precision is precision and model.model_id == cases[0].model_id
    )
    if len(models) != 1:
        raise RuntimeError("extended corpus model binding is missing or ambiguous")
    return cases[0], models[0]


def _plugin_build_provenance(
    *,
    project: Path,
    runs: Path,
    build_root: Path,
    build_log: Path,
    environment: EnvironmentLock,
) -> PluginBuildProvenance:
    provenance = runs / "provenance" / environment.id
    sources: list[ArtifactReference] = []
    for relative in _PLUGIN_SOURCES:
        source = _safe_relative_file(project, relative)
        destination = provenance / "source" / relative
        _copy_atomic(source, destination)
        sources.append(_artifact(runs, destination, _media_type(destination)))
    compile_source = _safe_relative_file(
        build_root.resolve(strict=True), f"{environment.id}/build/compile_commands.json"
    )
    compile_destination = provenance / "compile_commands.json"
    _copy_atomic(compile_source, compile_destination)
    log_destination = provenance / "plugin-compile-test.log"
    _copy_atomic(build_log.resolve(strict=True), log_destination)
    binary = _safe_relative_file(runs, f"{environment.id}/libupgrade_guard_residual_rmsnorm.so")
    architecture = environment.probe.gpu.compute_capability.replace(".", "")
    configure = (
        "cmake",
        "-S",
        "/opt/upgrade-guard",
        "-B",
        "/output/build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DUPGRADE_GUARD_BUILD_TESTS=ON",
        "-DUPGRADE_GUARD_BUILD_FAULTS=ON",
    )
    source_inventory = tuple(sources)
    return PluginBuildProvenance(
        environment_id=environment.id,
        source_inventory=source_inventory,
        source_inventory_sha256=sha256_bytes(
            canonical_json_bytes([item.model_dump(mode="json") for item in source_inventory])
        ),
        binary=_artifact(runs, binary, "application/x-sharedlib"),
        compile_commands=_artifact(runs, compile_destination, "application/json"),
        build_log=_artifact(runs, log_destination, "text/plain"),
        configure_command=configure,
        build_command=("cmake", "--build", "/output/build", "--parallel"),
        test_command=("ctest", "--test-dir", "/output/build", "--output-on-failure"),
    )


def _verify_corpus_artifacts(root: Path, manifest: ExtendedCorpusManifest) -> None:
    artifacts = [model.artifact for model in manifest.models]
    for case in manifest.cases:
        artifacts.extend(case.input_fixtures)
        artifacts.append(case.reference_output)
    for artifact in artifacts:
        path = _safe_relative_file(root, artifact.path)
        if sha256_file(path) != artifact.sha256 or path.stat().st_size != artifact.bytes:
            raise RuntimeError(f"extended corpus artifact identity differs: {artifact.path}")


def _artifact(root: Path, path: Path, media_type: str) -> ArtifactReference:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or path.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"extended artifact escaped its retention root: {path}")
    return ArtifactReference(
        path=resolved.relative_to(resolved_root).as_posix(),
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
        media_type=media_type,
    )


def _safe_relative_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError("extended artifact path is not a string")
    resolved_root = root.resolve(strict=True)
    path = resolved_root / relative
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or path.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"extended artifact path escaped its root: {relative}")
    return resolved


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def _yaml_object(path: Path) -> dict[str, object]:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"invalid YAML artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML artifact is not an object: {path}")
    return value


def _media_type(path: Path) -> str:
    return {
        ".cpp": "text/x-c++src",
        ".cu": "text/x-cuda",
        ".hpp": "text/x-c++hdr",
        ".json": "application/json",
        ".txt": "text/plain",
    }.get(path.suffix, "text/plain")
