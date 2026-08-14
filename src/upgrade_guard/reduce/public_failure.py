"""Strict public-failure routing, reduction, bundle, and replay publication."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256, sha256_file
from upgrade_guard.contracts.bundle import BundleManifest, canonical_cmake_cuda_architecture
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord, PrecisionMode
from upgrade_guard.contracts.environment import MatrixLock, Sha256Digest
from upgrade_guard.contracts.qualification import QualificationSpec, ReductionBudget
from upgrade_guard.contracts.results import RunResult
from upgrade_guard.errors import FailureCode, InfrastructureError, InvalidInputError
from upgrade_guard.reduce.domain import (
    DomainInput,
    DomainNumericalCandidate,
    DomainPlugin,
    run_domain_numerical_reduction,
)
from upgrade_guard.reduce.general import ReductionLimits
from upgrade_guard.reduce.public_replay import build_replay_predicate
from upgrade_guard.reduce.workflow import ReductionSessionManifest
from upgrade_guard.reproduce.builder import LocalDockerReplayImageBuilder
from upgrade_guard.reproduce.bundle import BundleExport, export_bundle
from upgrade_guard.reproduce.run import ReplayResult, execute_replay, observe_replay_target
from upgrade_guard.reproduce.verify import verify_bundle
from upgrade_guard.worker.common import write_json_atomic

SUPPORTED_PUBLIC_REDUCTION_CODES = (FailureCode.NUMERICAL_REGRESSION,)
DomainStep = Literal["core-qualification", "plugin-matrix", "mobilenet-matrix"]
_NOT_APPLICABLE_REASONS: dict[FailureCode, str] = {
    FailureCode.PLUGIN_COMPILE_FAILED: "no authored V1 compile-failure candidate reducer",
    FailureCode.ONNX_PARSE_FAILED: "no authored V1 parser-failure candidate reducer",
    FailureCode.ENGINE_BUILD_FAILED: "no authored V1 engine-build candidate reducer",
    FailureCode.ENGINE_DESERIALIZE_FAILED: "no authored V1 deserialization candidate reducer",
    FailureCode.PROFILE_REJECTED: "the authored profile reducer is limited to the G7 seeded gate",
    FailureCode.EXECUTION_FAILED: "no authored V1 generic execution-failure candidate reducer",
    FailureCode.OUTPUT_SCHEMA_CHANGED: "no authored V1 output-schema candidate reducer",
    FailureCode.NONFINITE_OUTPUT: "no authored V1 nonfinite-output candidate reducer",
    FailureCode.NONDETERMINISM_REGRESSION: "no authored V1 determinism candidate reducer",
    FailureCode.PERFORMANCE_REGRESSION: (
        "portable replay cannot recreate the locked paired hardware-validity schedule"
    ),
    FailureCode.MEMORY_REGRESSION: (
        "portable replay cannot recreate three locked memory confirmation builds"
    ),
    FailureCode.SANITIZER_FAILURE: "sanitizer failures are outside the three domain suites",
}


class DomainReductionRequest(StrictModel):
    """One genuine public failure bound to its exact candidate and locked budget."""

    schema_version: Literal["upgradeguard.dev/domain-reduction-request/v1"] = (
        "upgradeguard.dev/domain-reduction-request/v1"
    )
    source_step: DomainStep
    source_artifact_sha256: Sha256Digest
    matrix_lock_sha256: Sha256Digest
    failure: FailureRecord
    candidate_sha256: Sha256Digest
    reduction_budget: ReductionBudget
    request_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_request(self) -> DomainReductionRequest:
        if self.failure.code not in SUPPORTED_PUBLIC_REDUCTION_CODES:
            raise ValueError("domain reduction request uses an unsupported failure code")
        if self.request_sha256 != self.computed_sha256():
            raise ValueError("domain reduction request self-hash differs")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"request_sha256"})


class PublicFailureItem(StrictModel):
    """Terminal reduction or explicit non-applicable disposition for one predicate."""

    failure: FailureRecord
    disposition: Literal["reduced_replayed", "not_applicable"]
    reason: str | None = Field(default=None, min_length=1, max_length=4096)
    request: ArtifactReference | None = None
    request_sha256: Sha256Digest | None = None
    predicate_sha256: Sha256Digest | None = None
    session: ArtifactReference | None = None
    session_sha256: Sha256Digest | None = None
    final_candidate: ArtifactReference | None = None
    final_candidate_sha256: Sha256Digest | None = None
    bundle_manifest: ArtifactReference | None = None
    bundle_manifest_sha256: Sha256Digest | None = None
    replay_result: ArtifactReference | None = None
    replay_predicate: ArtifactReference | None = None
    expected_failure_code: FailureCode | None = None
    observed_failure_code: FailureCode | None = None
    expected_signature_sha256: Sha256Digest | None = None
    observed_signature_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> PublicFailureItem:
        links = (
            self.request,
            self.request_sha256,
            self.predicate_sha256,
            self.session,
            self.session_sha256,
            self.final_candidate,
            self.final_candidate_sha256,
            self.bundle_manifest,
            self.bundle_manifest_sha256,
            self.replay_result,
            self.replay_predicate,
            self.expected_failure_code,
            self.observed_failure_code,
            self.expected_signature_sha256,
            self.observed_signature_sha256,
        )
        if self.disposition == "not_applicable":
            if not self.reason or any(item is not None for item in links):
                raise ValueError("not-applicable failure disposition has reduction claims")
            if self.failure.code in SUPPORTED_PUBLIC_REDUCTION_CODES:
                raise ValueError("supported failures cannot be marked not applicable")
            return self
        if self.reason is not None or any(item is None for item in links):
            raise ValueError("reduced failure disposition lacks exact artifact linkage")
        if (
            self.expected_failure_code is not self.failure.code
            or self.observed_failure_code is not self.failure.code
            or self.expected_signature_sha256 != self.failure.signature_sha256
            or self.observed_signature_sha256 != self.failure.signature_sha256
        ):
            raise ValueError("reduced failure predicate identity differs")
        return self


class PublicFailureDisposition(StrictModel):
    """Complete terminal M6 resolution required before failed publication."""

    schema_version: Literal["upgradeguard.dev/public-failure-disposition/v1"] = (
        "upgradeguard.dev/public-failure-disposition/v1"
    )
    status: Literal["completed"] = "completed"
    source_step: DomainStep
    source_artifact: ArtifactReference
    supported_failure_codes: tuple[Literal["NUMERICAL_REGRESSION"]] = ("NUMERICAL_REGRESSION",)
    items: tuple[PublicFailureItem, ...] = Field(min_length=1)
    disposition_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> PublicFailureDisposition:
        signatures = tuple(item.failure.signature_sha256 for item in self.items)
        if len(signatures) != len(set(signatures)):
            raise ValueError("public failure predicates must be unique")
        if self.disposition_sha256 != self.computed_sha256():
            raise ValueError("public failure disposition self-hash differs")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"disposition_sha256"})


@dataclass(frozen=True, slots=True)
class NumericalMaterial:
    """Resolved genuine failure inputs used by reduction and portable replay."""

    candidate: DomainNumericalCandidate
    retained_baseline_output: Path
    retained_candidate_output: Path


def process_public_failure(
    *,
    state: Path,
    project: Path,
    source_step: DomainStep,
    core_corpus: Path,
    plugin_corpus: Path,
    mobilenet_corpus: Path,
    output: Path,
    registry: str = "127.0.0.1:5500",
) -> PublicFailureDisposition:
    """Resolve every typed domain failure before allowing failed publication."""

    state = state.resolve(strict=True)
    project = project.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / "disposition.json"
    if completed_path.is_file() and not completed_path.is_symlink():
        return validate_public_failure_disposition(output, state=state)
    source_relative = {
        "core-qualification": "core-run/qualification-summary.json",
        "plugin-matrix": "plugin-runs/validation.json",
        "mobilenet-matrix": "mobilenet-runs/validation.json",
    }.get(source_step)
    if source_relative is None:
        raise InvalidInputError("public failure source step is unsupported")
    source_path = state / source_relative
    failures = _source_failures(source_path, source_step)
    if any(item.code is FailureCode.CORPUS_INVALID for item in failures):
        raise InvalidInputError("CORPUS_INVALID cannot become a failed publication")
    matrix = MatrixLock.model_validate_json(
        (state / "matrix.lock.json").read_text(encoding="utf-8")
    )
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise InfrastructureError("public failure matrix lock self-hash differs")
    specification = _specification(state / "full.yaml")
    limits = ReductionLimits(
        maximum_trials=specification.reduction_budget.maximum_trials,
        maximum_seconds=float(specification.reduction_budget.maximum_seconds),
        confirmation_count=specification.reduction_budget.confirmation_count,
    )
    items: list[PublicFailureItem] = []
    for index, failure in enumerate(failures):
        if failure.code not in SUPPORTED_PUBLIC_REDUCTION_CODES:
            reason = _NOT_APPLICABLE_REASONS.get(failure.code)
            if reason is None:
                raise InfrastructureError(
                    f"no explicit public failure disposition exists for {failure.code.value}"
                )
            items.append(
                PublicFailureItem(
                    failure=failure,
                    disposition="not_applicable",
                    reason=reason,
                )
            )
            continue
        item_root = output / f"failure-{index:02d}"
        item_root.mkdir(exist_ok=True)
        _preserve_staging(item_root)
        material = _numerical_material(
            state=state,
            source_step=source_step,
            failure=failure,
            specification=specification,
            matrix=matrix,
            core_corpus=core_corpus.resolve(strict=True),
            plugin_corpus=plugin_corpus.resolve(strict=True),
            mobilenet_corpus=mobilenet_corpus.resolve(strict=True),
        )
        request_path = item_root / "request.json"
        unvalidated_request = DomainReductionRequest.model_construct(
            source_step=source_step,
            source_artifact_sha256=sha256_file(source_path),
            matrix_lock_sha256=matrix.lock_sha256,
            failure=failure,
            candidate_sha256=material.candidate.candidate_sha256(),
            reduction_budget=specification.reduction_budget,
            request_sha256="sha256:" + "0" * 64,
        )
        request = DomainReductionRequest.model_validate(
            unvalidated_request.model_copy(
                update={"request_sha256": unvalidated_request.computed_sha256()}
            )
        )
        if request_path.exists():
            retained = DomainReductionRequest.model_validate_json(
                request_path.read_text(encoding="utf-8")
            )
            if retained != request:
                raise InfrastructureError("retained domain reduction request identity differs")
        else:
            write_json_atomic(request_path, request.model_dump(mode="json"))
        reduction_root = item_root / "reduction"
        session_path = reduction_root / "session.json"
        candidate_path = reduction_root / "candidate.json"
        if session_path.is_file() and candidate_path.is_file():
            session = ReductionSessionManifest.model_validate_json(
                session_path.read_text(encoding="utf-8")
            )
            reduced = DomainNumericalCandidate.model_validate_json(
                candidate_path.read_text(encoding="utf-8")
            )
            if (
                session.status.value != "completed"
                or session.predicate.predicate_signature_sha256 != failure.signature_sha256
                or session.final_candidate_sha256 != reduced.candidate_sha256()
                or session.maximum_trials != limits.maximum_trials
                or session.maximum_seconds != limits.maximum_seconds
                or session.confirmation_count != limits.confirmation_count
            ):
                raise InfrastructureError("retained domain reduction session differs")
        else:
            if reduction_root.exists():
                _preserve_partial(reduction_root)
            result = run_domain_numerical_reduction(
                project=project,
                matrix=matrix,
                failure=failure,
                original=material.candidate,
                retained_baseline_output=material.retained_baseline_output,
                retained_candidate_output=material.retained_candidate_output,
                output=reduction_root,
                limits=limits,
            )
            session = result.session
            reduced = result.candidate
        bundle_root = item_root / "bundle"
        if bundle_root.exists():
            bundle = verify_bundle(bundle_root).manifest
            if (
                bundle.expected_failure != failure
                or bundle.id != f"domain-{failure.signature_sha256.removeprefix('sha256:')[:20]}"
            ):
                raise InfrastructureError("retained public failure bundle differs")
        else:
            bundle = _export_numerical_bundle(
                project=project,
                state=state,
                source_step=source_step,
                matrix=matrix,
                specification=specification,
                failure=failure,
                candidate=reduced,
                baseline_output=material.retained_baseline_output,
                destination=bundle_root,
            )
        replay_root = item_root / "clean-replay"
        if replay_root.exists():
            replay = _load_replay(replay_root / "replay-result.json", bundle.manifest_sha256)
        else:
            target = observe_replay_target(matrix.gpu_uuid)
            replay = execute_replay(
                bundle_root,
                replay_root,
                trust_source_code=True,
                trust_included_engine=False,
                replay_target=target,
                image_builder=LocalDockerReplayImageBuilder(registry),
            )
        if (
            replay.expected_failure_code is not failure.code
            or replay.observed_failure_code is not failure.code
        ):
            raise InfrastructureError("public failure clean replay code differs")
        replay_predicate_path = replay_root / "steps" / "three-way-failure.json"
        observed_signature = _replay_signature(replay_predicate_path)
        if observed_signature != failure.signature_sha256:
            raise InfrastructureError("public failure clean replay signature differs")
        items.append(
            PublicFailureItem(
                failure=failure,
                disposition="reduced_replayed",
                request=_artifact(output, request_path),
                request_sha256=request.request_sha256,
                predicate_sha256=session.predicate.predicate_sha256,
                session=_artifact(output, session_path),
                session_sha256=session.session_sha256,
                final_candidate=_artifact(output, candidate_path),
                final_candidate_sha256=reduced.candidate_sha256(),
                bundle_manifest=_artifact(output, bundle_root / "bundle.json"),
                bundle_manifest_sha256=bundle.manifest_sha256,
                replay_result=_artifact(output, replay_root / "replay-result.json"),
                replay_predicate=_artifact(output, replay_predicate_path),
                expected_failure_code=failure.code,
                observed_failure_code=replay.observed_failure_code,
                expected_signature_sha256=failure.signature_sha256,
                observed_signature_sha256=observed_signature,
            )
        )
    unvalidated_disposition = PublicFailureDisposition.model_construct(
        source_step=source_step,
        source_artifact=_artifact(state, source_path),
        items=tuple(items),
        disposition_sha256="sha256:" + "0" * 64,
    )
    disposition = PublicFailureDisposition.model_validate(
        unvalidated_disposition.model_copy(
            update={
                "disposition_sha256": unvalidated_disposition.computed_sha256(),
            }
        )
    )
    write_json_atomic(completed_path, disposition.model_dump(mode="json"))
    return validate_public_failure_disposition(output, state=state)


def validate_public_failure_disposition(
    root: Path,
    *,
    state: Path | None = None,
) -> PublicFailureDisposition:
    """Recompute every retained link before resume or failed publication."""

    root = root.resolve(strict=True)
    value = PublicFailureDisposition.model_validate_json(
        (root / "disposition.json").read_text(encoding="utf-8")
    )
    for item in value.items:
        for artifact in (
            item.request,
            item.session,
            item.final_candidate,
            item.bundle_manifest,
            item.replay_result,
            item.replay_predicate,
        ):
            if artifact is not None:
                _verify_artifact(root, artifact)
        if item.disposition == "reduced_replayed":
            assert item.request is not None and item.session is not None
            assert item.final_candidate is not None and item.bundle_manifest is not None
            assert item.replay_result is not None and item.replay_predicate is not None
            request = DomainReductionRequest.model_validate_json(
                (root / item.request.path).read_text(encoding="utf-8")
            )
            session = ReductionSessionManifest.model_validate_json(
                (root / item.session.path).read_text(encoding="utf-8")
            )
            candidate = DomainNumericalCandidate.model_validate_json(
                (root / item.final_candidate.path).read_text(encoding="utf-8")
            )
            replay = _load_replay(root / item.replay_result.path, item.bundle_manifest_sha256)
            observed_signature = _replay_signature(root / item.replay_predicate.path)
            bundle = verify_bundle((root / item.bundle_manifest.path).parent).manifest
            if (
                request.failure != item.failure
                or request.source_step != value.source_step
                or request.source_artifact_sha256 != value.source_artifact.sha256
                or request.request_sha256 != item.request_sha256
                or session.expected_failure_code is not item.failure.code
                or session.predicate_signature_sha256 != item.failure.signature_sha256
                or session.predicate.failure_code is not item.failure.code
                or session.original_candidate_sha256 != request.candidate_sha256
                or session.maximum_trials != request.reduction_budget.maximum_trials
                or session.maximum_seconds != float(request.reduction_budget.maximum_seconds)
                or session.confirmation_count != request.reduction_budget.confirmation_count
                or session.predicate.predicate_sha256 != item.predicate_sha256
                or session.session_sha256 != item.session_sha256
                or session.final_candidate_sha256 != candidate.candidate_sha256()
                or candidate.candidate_sha256() != item.final_candidate_sha256
                or bundle.expected_failure != item.failure
                or bundle.manifest_sha256 != item.bundle_manifest_sha256
                or replay.expected_failure_code is not item.failure.code
                or replay.observed_failure_code is not item.failure.code
                or observed_signature != item.failure.signature_sha256
                or item.observed_signature_sha256 != observed_signature
            ):
                raise InfrastructureError("public failure disposition linkage differs")
            if state is not None:
                matrix = MatrixLock.model_validate_json(
                    (state / "matrix.lock.json").read_text(encoding="utf-8")
                )
                if request.matrix_lock_sha256 != matrix.lock_sha256:
                    raise InfrastructureError("public failure request matrix binding differs")
    if state is not None:
        state_root = state.resolve(strict=True)
        _verify_artifact(state_root, value.source_artifact)
        retained_failures = _source_failures(
            state_root / value.source_artifact.path,
            value.source_step,
        )
        if tuple(item.failure for item in value.items) != retained_failures:
            raise InfrastructureError("public failure records differ from their source artifact")
    return value


def _source_failures(path: Path, source_step: DomainStep) -> tuple[FailureRecord, ...]:
    value = _json(path)
    if value.get("status") != "failed":
        raise InvalidInputError("public failure source is not a failed domain result")
    if source_step == "core-qualification":
        raw = value.get("failures")
        if not isinstance(raw, list) or not raw:
            raise InfrastructureError("core failure lacks decision-time FailureRecords")
        failures = tuple(FailureRecord.model_validate(item) for item in raw)
    else:
        raw = value.get("failure")
        if raw is None:
            raise InfrastructureError("extended failure lacks a FailureRecord")
        failures = (FailureRecord.model_validate(raw),)
    raw_codes = value.get("failure_codes")
    if source_step == "core-qualification" and (
        not isinstance(raw_codes, list) or not all(isinstance(item, str) for item in raw_codes)
    ):
        raise InfrastructureError("core failure codes are malformed")
    codes = (
        tuple(dict.fromkeys(FailureCode(item) for item in raw_codes))
        if isinstance(raw_codes, list)
        else ()
    )
    if source_step != "core-qualification":
        code = value.get("failure_code")
        codes = (FailureCode(code),) if isinstance(code, str) else ()
    if not codes or set(codes) != {item.code for item in failures}:
        raise InfrastructureError("domain failure codes and FailureRecords differ")
    return failures


def _numerical_material(
    *,
    state: Path,
    source_step: DomainStep,
    failure: FailureRecord,
    specification: QualificationSpec,
    matrix: MatrixLock,
    core_corpus: Path,
    plugin_corpus: Path,
    mobilenet_corpus: Path,
) -> NumericalMaterial:
    if failure.precision is None or failure.output_name is None:
        raise InfrastructureError("numerical failure lacks precision or output identity")
    precision = "fp32" if failure.precision is PrecisionMode.FP32 else "fp16"
    plugins: tuple[DomainPlugin, ...] = ()
    if source_step == "core-qualification":
        summary = _json(state / "core-run" / "qualification-summary.json")
        raw_cases = summary.get("cases")
        if not isinstance(raw_cases, list):
            raise InfrastructureError("core numerical cases are malformed")
        cases = [
            item
            for item in raw_cases
            if isinstance(item, dict)
            and item.get("failure_code") == failure.code.value
            and item.get("precision") == precision
            and item.get("shape_id") == failure.shape_id
        ]
        if len(cases) != 1 or failure.shape_id is None:
            raise InfrastructureError("core numerical failure lacks one exact case")
        typed = cases[0].get("typed_run_results")
        if not isinstance(typed, dict):
            raise InfrastructureError("core numerical failure lacks typed runs")
        baseline = RunResult.model_validate(typed[matrix.environments[0].id])
        candidate_run = RunResult.model_validate(typed[matrix.environments[1].id])
        root = state / "core-run"
        corpus = core_corpus
        model = corpus / "models" / f"tiny-transformer-{precision}.onnx"
        profile = root / matrix.environments[1].id / precision / "profile.json"
        input_root = corpus / "inputs" / f"tiny-transformer-{precision}" / failure.shape_id
        input_paths = tuple(input_root / f"{name}.npy" for name in ("tokens", "mask"))
        reference = (
            corpus
            / "reference"
            / f"tiny-transformer-{precision}-{failure.shape_id}-{failure.output_name}.npy"
        )
    else:
        root_name = "plugin-runs" if source_step == "plugin-matrix" else "mobilenet-runs"
        root = state / root_name
        validation = _json(root / "validation.json")
        raw_cases = validation.get("cases")
        if not isinstance(raw_cases, list):
            raise InfrastructureError("extended numerical cases are malformed")
        matching = [
            item
            for item in raw_cases
            if isinstance(item, dict) and item.get("failure_code") == failure.code.value
        ]
        if len(matching) != 1:
            raise InfrastructureError("extended numerical failure lacks one exact case")
        case = matching[0]
        stable = case.get("stable_artifacts")
        if not isinstance(stable, dict):
            raise InfrastructureError("extended numerical failure lacks stable chains")
        baseline = _stable_run(root, stable, matrix.environments[0].id)
        candidate_run = _stable_run(root, stable, matrix.environments[1].id)
        if source_step == "plugin-matrix":
            case_id = str(case["case"])
            corpus = plugin_corpus
            model = corpus / f"residual-rmsnorm-{precision}.onnx"
            profile = root / matrix.environments[1].id / precision / "profile.json"
            input_root = corpus / precision / case_id
            input_paths = tuple(input_root / f"{name}.npy" for name in ("x", "residual", "gamma"))
            reference = input_root / "expected.npy"
            plugins = tuple(
                DomainPlugin(
                    environment_id=environment.id,
                    path=(
                        state
                        / "plugin-build"
                        / environment.id
                        / "build"
                        / "libupgrade_guard_residual_rmsnorm.so"
                    ),
                    sha256=sha256_file(
                        state
                        / "plugin-build"
                        / environment.id
                        / "build"
                        / "libupgrade_guard_residual_rmsnorm.so"
                    ),
                )
                for environment in matrix.environments
            )
        else:
            case_id = str(case["case"])
            corpus = mobilenet_corpus
            model = corpus / "mobilenetv3-small-075-dynamic.onnx"
            profile = root / matrix.environments[1].id / "profile.json"
            input_root = corpus / "inputs" / case_id
            input_paths = (input_root / "x.npy",)
            reference = input_root / "expected.npy"
    input_names = (
        ("tokens", "mask")
        if source_step == "core-qualification"
        else ("x", "residual", "gamma")
        if source_step == "plugin-matrix"
        else ("x",)
    )
    inputs = tuple(
        DomainInput(
            name=name,
            path=path,
            sha256=sha256_file(path),
            shape=tuple(int(item) for item in np.load(path, allow_pickle=False).shape),
        )
        for name, path in zip(input_names, input_paths, strict=True)
    )
    baseline_output = _first_output(root, baseline, failure.output_name)
    candidate_output = _first_output(root, candidate_run, failure.output_name)
    material = DomainNumericalCandidate(
        model_path=model,
        model_sha256=sha256_file(model),
        profile_path=profile,
        profile_sha256=sha256_file(profile),
        inputs=inputs,
        reference_path=reference,
        reference_sha256=sha256_file(reference),
        output_name=failure.output_name,
        semantics="classification" if source_step == "mobilenet-matrix" else "tensor",
        policy=specification.numerical_policy(failure.precision),
        determinism=specification.determinism,
        workspace_bytes=specification.builder.workspace_limit_bytes,
        optimization_level=specification.builder.optimization_level,
        environment_history=(matrix.environments[0].id, matrix.environments[1].id),
        plugins=plugins,
    )
    material.verify_artifacts()
    return NumericalMaterial(material, baseline_output, candidate_output)


def _export_numerical_bundle(
    *,
    project: Path,
    state: Path,
    source_step: DomainStep,
    matrix: MatrixLock,
    specification: QualificationSpec,
    failure: FailureRecord,
    candidate: DomainNumericalCandidate,
    baseline_output: Path,
    destination: Path,
) -> BundleManifest:
    candidate_environment = matrix.environments[1]
    inputs_root = destination.parent / "bundle-inputs"
    inputs_root.mkdir(exist_ok=True)
    baseline_environment_path = inputs_root / "baseline.environment.json"
    candidate_environment_path = inputs_root / "candidate.environment.json"
    policy_path = inputs_root / "policy.json"
    failure_path = inputs_root / "failure-record.json"
    source_result_path = (
        state
        / {
            "core-qualification": "core-run/qualification-summary.json",
            "plugin-matrix": "plugin-runs/validation.json",
            "mobilenet-matrix": "mobilenet-runs/validation.json",
        }[source_step]
    )
    if source_result_path.stat().st_size > 4 * 1024**2:
        raise InfrastructureError("public failure source evidence exceeds the V1 bundle bound")
    failure_path.write_text(failure.model_dump_json(indent=2) + "\n", encoding="utf-8")
    source_failure_root = source_result_path.parent
    failure_evidence: dict[str, Path] = {}
    for artifact in failure.evidence:
        _verify_artifact(source_failure_root, artifact)
        if artifact.path in failure_evidence:
            raise InfrastructureError("public failure evidence paths are not unique")
        failure_evidence[artifact.path] = source_failure_root / artifact.path
    for path, environment in (
        (baseline_environment_path, matrix.environments[0]),
        (candidate_environment_path, candidate_environment),
    ):
        path.write_text(environment.model_dump_json(indent=2) + "\n", encoding="utf-8")
    policy_path.write_text(candidate.policy.model_dump_json(indent=2) + "\n", encoding="utf-8")
    architecture = canonical_cmake_cuda_architecture(
        candidate_environment.probe.gpu.compute_capability
    )
    configure = (
        "cmake",
        "-S",
        "/opt/upgrade-guard",
        "-B",
        "/output/build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DUPGRADE_GUARD_BUILD_TESTS=OFF",
        "-DUPGRADE_GUARD_BUILD_FAULTS=OFF",
        f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
    )
    plugin_arguments: tuple[str, ...] = ()
    steps: list[dict[str, object]] = [{"id": "configure", "command": list(configure)}]
    if candidate.plugins:
        steps.append(
            {
                "id": "compile-plugin",
                "command": [
                    "cmake",
                    "--build",
                    "/output/build",
                    "--target",
                    "upgrade_guard_residual_rmsnorm",
                ],
            }
        )
        plugin_arguments = ("--plugin", "/output/build/libupgrade_guard_residual_rmsnorm.so")
    build_command = [
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
        "/output/build.json",
        "--workspace-bytes",
        str(candidate.workspace_bytes),
        "--optimization-level",
        str(candidate.optimization_level),
        *plugin_arguments,
    ]
    steps.append(
        {
            "id": "build-engine",
            "command": build_command,
            "result_file": "build.json",
            "expected_result_status": "passed",
        }
    )
    correctness = [
        "python3",
        "-m",
        "upgrade_guard.worker.run_correctness",
        "--engine",
        "/output/engine.plan",
    ]
    for index, item in enumerate(candidate.inputs):
        correctness.extend(("--input", f"{item.name}=/corpus/inputs/{index:03d}-{item.path.name}"))
    correctness.extend(
        (
            "--output",
            "/output/outputs",
            "--result",
            "/output/correctness.json",
            "--repetitions",
            str(candidate.determinism.repetitions),
            *plugin_arguments,
        )
    )
    steps.append(
        {
            "id": "run-correctness",
            "command": correctness,
            "result_file": "correctness.json",
            "expected_result_status": "passed",
        }
    )
    if candidate.semantics == "classification":
        replay_indexes = candidate.classification_indexes
        if not replay_indexes:
            raise InfrastructureError("reduced classification candidate lacks output indexes")
    else:
        if candidate.comparison_flat_index is None:
            raise InfrastructureError("reduced numerical candidate lacks its output index")
        replay_indexes = (candidate.comparison_flat_index,)
    replay_predicate_path = inputs_root / "replay-predicate.json"
    replay_predicate = build_replay_predicate(
        failure_signature_sha256=failure.signature_sha256,
        output_name=candidate.output_name,
        semantics=candidate.semantics,
        indexes=replay_indexes,
        reference=candidate.reference_path,
        baseline=baseline_output,
        policy=policy_path,
    )
    replay_predicate_path.write_text(
        replay_predicate.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    steps.append(
        {
            "id": "three-way-failure",
            "command": [
                "python3",
                "-m",
                "upgrade_guard.reduce.public_replay",
                "--reference",
                "/corpus/reference.npy",
                "--baseline",
                "/corpus/baseline.npy",
                "--candidate",
                f"/output/outputs/{candidate.output_name}.repetition-00.npy",
                "--policy",
                "/corpus/policy.json",
                "--predicate",
                "/corpus/reduction/replay-predicate.json",
            ],
            "stdout_json_equals": {"signature_sha256": failure.signature_sha256},
            "expected_failure_code": failure.code.value,
            "failure_code_source": "stdout",
        }
    )
    recipe_path = inputs_root / "replay.json"
    recipe_path.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/replay-recipe/v1",
                "expected_failure_code": failure.code.value,
                "steps": steps,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sources = _source_files(project)
    return export_bundle(
        BundleExport(
            id=f"domain-{failure.signature_sha256.removeprefix('sha256:')[:20]}",
            created_at=datetime.now(UTC),
            baseline_environment=baseline_environment_path,
            candidate_environment=candidate_environment_path,
            qualification=state / "full.yaml",
            model=candidate.model_path,
            inputs=tuple(item.path for item in candidate.inputs),
            expected_failure=failure,
            extra_files={
                "commands/replay.json": recipe_path,
                "profile.json": candidate.profile_path,
                "reference.npy": candidate.reference_path,
                "baseline.npy": baseline_output,
                "policy.json": policy_path,
                "reduction/replay-predicate.json": replay_predicate_path,
                "logs/failure-record.json": failure_path,
                "logs/source-result.json": source_result_path,
                **failure_evidence,
            },
            source_files=sources,
            original_worker_image_manifest_digest=(
                candidate_environment.worker_image.manifest_digest
            ),
            original_gpu_uuid=matrix.gpu_uuid,
            base_image=candidate_environment.base_image.canonical_reference,
            base_image_manifest_digest=candidate_environment.base_image.manifest_digest,
            dockerfile=project / "containers" / "Dockerfile.worker",
            worker_lock=project / "containers" / "requirements-worker.txt",
            worker_build_arguments=(
                ("BASE_IMAGE", candidate_environment.base_image.canonical_reference),
                (
                    "BASE_MANIFEST_DIGEST",
                    candidate_environment.base_image.manifest_digest,
                ),
            ),
            minimum_compute_capability=(
                candidate_environment.compatibility.minimum_compute_capability
            ),
            minimum_driver=candidate_environment.compatibility.minimum_driver,
            minimum_vram_mib=_minimum_vram_mib(project),
            original_compute_capability=candidate_environment.probe.gpu.compute_capability,
            source_build_command=configure,
        ),
        destination,
    )


def _source_files(project: Path) -> dict[str, Path]:
    files: dict[str, Path] = {"CMakeLists.txt": project / "CMakeLists.txt"}
    reviewed_suffixes = {
        ".c",
        ".cc",
        ".cmake",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".py",
    }
    for root in (project / "src", project / "cpp", project / "cmake"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix.lower() in reviewed_suffixes
            ):
                files[path.relative_to(project).as_posix()] = path
    return files


def _minimum_vram_mib(project: Path) -> int:
    policy = project / "src" / "upgrade_guard" / "matrix" / "compatibility-rules.json"
    try:
        value = json.loads(policy.read_text(encoding="utf-8"))["minimum_vram_mib"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise InfrastructureError("compatibility policy lacks minimum_vram_mib") from error
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InfrastructureError("compatibility policy minimum_vram_mib is invalid")
    return value


def _stable_run(root: Path, stable: dict[str, object], environment_id: str) -> RunResult:
    chain = stable.get(environment_id)
    if not isinstance(chain, dict) or not isinstance(chain.get("run_result"), dict):
        raise InfrastructureError("extended stable run chain is malformed")
    relative = chain["run_result"].get("path")
    if not isinstance(relative, str):
        raise InfrastructureError("extended stable run path is malformed")
    return RunResult.model_validate_json((root / relative).read_text(encoding="utf-8"))


def _first_output(root: Path, run: RunResult, output_name: str) -> Path:
    names = tuple(item.name for item in run.output_schema)
    if output_name not in names or not run.output_artifacts:
        raise InfrastructureError("stable numerical run lacks its output")
    output_index = names.index(output_name)
    artifact = run.output_artifacts[output_index]
    path = root / artifact.path
    _verify_artifact(root, artifact)
    return path


def _specification(path: Path) -> QualificationSpec:
    try:
        return QualificationSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InfrastructureError("locked qualification specification is invalid") from error


def _artifact(root: Path, path: Path) -> ArtifactReference:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise InfrastructureError("public failure artifact escaped its root")
    return ArtifactReference(
        path=resolved.relative_to(root).as_posix(),
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
        media_type="application/json",
    )


def _verify_artifact(root: Path, artifact: ArtifactReference) -> None:
    path = root / artifact.path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise InfrastructureError("public failure artifact is unavailable") from error
    if (
        path.is_symlink()
        or not resolved.is_relative_to(root.resolve(strict=True))
        or not resolved.is_file()
        or resolved.stat().st_size != artifact.bytes
        or sha256_file(resolved) != artifact.sha256
    ):
        raise InfrastructureError("public failure artifact identity differs")


def _load_replay(path: Path, bundle_manifest_sha256: str | None) -> ReplayResult:
    try:
        raw = _json(path)
        steps = raw["step_results"]
        if not isinstance(steps, list) or not all(isinstance(item, str) for item in steps):
            raise ValueError("replay step results are invalid")
        value = ReplayResult(
            schema_version=str(raw["schema_version"]),
            status=str(raw["status"]),
            bundle_id=str(raw["bundle_id"]),
            bundle_manifest_sha256=str(raw["bundle_manifest_sha256"]),
            worker_image=str(raw["worker_image"]),
            worker_rebuild_recipe_sha256=str(raw["worker_rebuild_recipe_sha256"]),
            worker_build_log_sha256=str(raw["worker_build_log_sha256"]),
            worker_build_log=ArtifactReference.model_validate(raw["worker_build_log"]),
            original_gpu_uuid=str(raw["original_gpu_uuid"]),
            selected_gpu_uuid=str(raw["selected_gpu_uuid"]),
            expected_failure_code=FailureCode(str(raw["expected_failure_code"])),
            observed_failure_code=FailureCode(str(raw["observed_failure_code"])),
            step_results=tuple(steps),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise InfrastructureError("public failure replay result is invalid") from error
    if value.status != "passed" or value.bundle_manifest_sha256 != bundle_manifest_sha256:
        raise InfrastructureError("public failure replay result identity differs")
    _verify_artifact(path.parent, value.worker_build_log)
    if value.worker_build_log.sha256 != value.worker_build_log_sha256:
        raise InfrastructureError("public failure replay build-log hash differs")
    return value


def _replay_signature(path: Path) -> str:
    record = _json(path)
    stdout = record.get("stdout")
    if not isinstance(stdout, str):
        raise InfrastructureError("public failure replay predicate lacks stdout evidence")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise InfrastructureError("public failure replay predicate stdout is invalid") from error
    signature = value.get("signature_sha256") if isinstance(value, dict) else None
    if not isinstance(signature, str):
        raise InfrastructureError("public failure replay predicate lacks a signature")
    return signature


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InfrastructureError(f"required public failure JSON is invalid: {path}") from error
    if not isinstance(value, dict):
        raise InfrastructureError("required public failure JSON is not an object")
    return value


def _preserve_partial(path: Path) -> None:
    target = path.with_name(f"{path.name}.partial-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    if target.exists():
        raise InfrastructureError("public failure partial preservation target exists")
    path.replace(target)


def _preserve_staging(root: Path) -> None:
    """Move exact project staging directories aside before a resumable retry."""

    for path in sorted(root.iterdir()):
        if not path.name.startswith((".bundle.", ".clean-replay.")):
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise InfrastructureError("public failure staging path is unavailable") from error
        if not stat.S_ISDIR(mode):
            raise InfrastructureError("public failure staging path is unsafe")
        suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        target = root / f"stale-{path.name.removeprefix('.')}-{suffix}"
        if target.exists():
            raise InfrastructureError("public failure staging preservation target exists")
        path.replace(target)
