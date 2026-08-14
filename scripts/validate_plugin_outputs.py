"""Validate both plugin workers with one authored three-way numerical policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.three_way_validation import (
    ThreeWayValidationResult,
    failure_record,
    is_output_schema_failure,
    worker_evidence_failure_code,
    worker_output_tolerance_stable,
)
from upgrade_guard.classify import status_for_failure
from upgrade_guard.compare.numerical import (
    ThreeWayDecision,
    ThreeWayPrecedenceError,
    decide_three_way,
)
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.build import WorkerBuildResult
from upgrade_guard.contracts.common import FailureRecord, Phase, PrecisionMode
from upgrade_guard.contracts.qualification import QualificationSpec
from upgrade_guard.contracts.results import WorkerCorrectnessResult
from upgrade_guard.errors import FailureCode
from upgrade_guard.extended_artifacts import (
    ExtendedPromotionContext,
    prepare_extended_promotion,
    promote_extended_build_failure,
    promote_extended_case,
)
from upgrade_guard.worker.common import write_json_atomic
from upgrade_guard.worker.evidence import validate_repetitions


def _specification(path: Path) -> QualificationSpec:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return QualificationSpec.model_validate(value)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise RuntimeError("plugin validation specification is invalid") from error


def _decision_evidence(index: int, decision: ThreeWayDecision) -> dict[str, object]:
    return {
        "repetition": index,
        "passed": decision.passed,
        "failure_code": decision.failure_code.value if decision.failure_code else None,
        "failed_gates": list(decision.failed_gates),
        "baseline_to_reference": decision.baseline_to_reference.model_dump(mode="json"),
        "candidate_to_reference": decision.candidate_to_reference.model_dump(mode="json"),
        "candidate_to_baseline": decision.candidate_to_baseline.model_dump(mode="json"),
    }


def _outputs(root: Path, name: str, repetitions: int) -> tuple[np.ndarray[Any, Any], ...]:
    return tuple(
        np.load(root / "outputs" / f"{name}.repetition-{index:02d}.npy", allow_pickle=False)
        for index in range(repetitions)
    )


def _write_result(
    path: Path,
    *,
    specification_sha256: str,
    repetitions: int,
    cases: list[dict[str, object]],
    failure: FailureRecord | None,
    promotion: ExtendedPromotionContext,
) -> None:
    result = ThreeWayValidationResult(
        schema_version="upgradeguard.dev/plugin-validation/v2",
        status=status_for_failure(failure.code if failure else None),
        failure_code=failure.code if failure else None,
        failure=failure,
        specification_sha256=specification_sha256,
        invocation_manifest=promotion.invocation_artifact,
        invocation_manifest_sha256=promotion.invocation.manifest_sha256,
        repetitions=repetitions,
        cases=tuple(cases),
    )
    write_json_atomic(path, result.model_dump(mode="json"))


def _promote_retained_worker_failure(
    *,
    arguments: argparse.Namespace,
    promotion: ExtendedPromotionContext,
    specification_sha256: str,
    repetitions: int,
) -> None:
    """Promote the first strict failed worker result retained by the shell runner."""

    failures: list[tuple[str, Path, WorkerBuildResult | WorkerCorrectnessResult]] = []
    for kind, pattern, model in (
        ("build", "*/*/build.json", WorkerBuildResult),
        ("correctness", "*/*/*/correctness.json", WorkerCorrectnessResult),
    ):
        for path in sorted(arguments.runs.glob(pattern)):
            try:
                value = model.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if value.status == "failed":
                failures.append((kind, path, value))
    if not failures:
        return
    if len(failures) != 1:
        raise RuntimeError("plugin run retained more than one terminal worker failure")
    kind, result_path, worker = failures[0]
    if worker.command_sha256 != command_sha256(worker.command) or worker.failure_code is None:
        raise RuntimeError("plugin failed worker command or classification differs")
    relative = result_path.relative_to(arguments.runs)
    environment, precision = relative.parts[:2]
    precision_mode = PrecisionMode.FP32 if precision == "fp32" else PrecisionMode.EXPLICIT_FP16
    case_name = (
        relative.parts[2]
        if kind == "correctness"
        else next(
            path.name for path in sorted((arguments.corpus / precision).iterdir()) if path.is_dir()
        )
    )
    failure = failure_record(
        code=worker.failure_code,
        phase=Phase.BUILD if kind == "build" else Phase.CORRECTNESS,
        environment_id=environment,
        model_id="residual-rmsnorm-plugin",
        precision=precision_mode,
        case_id=case_name,
        output_name="output",
        gate="worker_build" if kind == "build" else "worker_execution",
        observed=f"strict worker {kind} result failed",
        threshold="the isolated worker must complete with typed passing evidence",
        runs_root=arguments.runs,
        evidence_paths=(result_path,),
    )
    output_root = arguments.runs / environment / precision / case_name
    if kind == "build":
        stable = promote_extended_build_failure(
            promotion,
            environment_id=environment,
            precision=precision_mode,
            case_id=f"{precision}-{case_name}",
            build_result_path=result_path,
            output_root=output_root,
            failure=failure,
        )
    else:
        stable = promote_extended_case(
            promotion,
            environment_id=environment,
            precision=precision_mode,
            case_id=f"{precision}-{case_name}",
            build_result_path=arguments.runs / environment / precision / "build.json",
            correctness_result_path=result_path,
            output_root=output_root,
            numerical=(),
            determinism_tolerance_stable=False,
            failure=failure,
        )
    case = {
        "precision": precision,
        "case": case_name,
        "status": status_for_failure(failure.code).value,
        "failure_code": failure.code.value,
        "workers": {environment: {"passed": False, "failure_code": failure.code.value}},
        "stable_artifacts": {environment: stable},
    }
    _write_result(
        arguments.output,
        specification_sha256=specification_sha256,
        repetitions=repetitions,
        cases=[case],
        failure=failure,
        promotion=promotion,
    )
    if failure.code is FailureCode.CORPUS_INVALID:
        raise SystemExit(2)
    raise RuntimeError(f"plugin worker {kind} failed: {failure.code.value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--matrix-lock", type=Path, required=True)
    parser.add_argument("--source-commit-file", type=Path, required=True)
    parser.add_argument("--plugin-build-root", type=Path, required=True)
    parser.add_argument("--plugin-build-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    specification = _specification(arguments.specification)
    specification_sha256 = sha256_file(arguments.specification)
    promotion = prepare_extended_promotion(
        suite="plugin",
        project_root=arguments.project_root,
        corpus_root=arguments.corpus,
        runs_root=arguments.runs,
        specification_path=arguments.specification,
        matrix_lock_path=arguments.matrix_lock,
        source_commit_file=arguments.source_commit_file,
        plugin_build_root=arguments.plugin_build_root,
        plugin_build_log=arguments.plugin_build_log,
    )
    repetitions = specification.determinism.repetitions
    _promote_retained_worker_failure(
        arguments=arguments,
        promotion=promotion,
        specification_sha256=specification_sha256,
        repetitions=repetitions,
    )
    evidence: list[dict[str, object]] = []
    for precision, precision_mode in (
        ("fp32", PrecisionMode.FP32),
        ("fp16", PrecisionMode.EXPLICIT_FP16),
    ):
        policy = specification.numerical_policy(precision_mode)
        environment_tolerances = {
            "baseline": policy.baseline_to_reference,
            "candidate": policy.candidate_to_reference,
        }
        for case_root in sorted((arguments.corpus / precision).iterdir()):
            if not case_root.is_dir():
                continue
            expected_path = case_root / "expected.npy"
            expected = np.load(expected_path, allow_pickle=False)
            expected_input_hashes = {
                name: sha256_file(case_root / f"{name}.npy") for name in ("x", "residual", "gamma")
            }
            case_evidence: dict[str, object] = {
                "precision": precision,
                "case": case_root.name,
                "reference_sha256": sha256_file(expected_path),
                "workers": {},
            }
            worker_outputs = {}
            workers = case_evidence["workers"]
            if not isinstance(workers, dict):
                raise AssertionError("plugin worker evidence changed type")
            for environment in ("baseline", "candidate"):
                result_root = arguments.runs / environment / precision / case_root.name
                tolerance = environment_tolerances[environment]
                result_path = result_root / "correctness.json"
                try:
                    validation = validate_repetitions(
                        result_path=result_path,
                        runs_root=arguments.runs,
                        expected_output_name="output",
                        expected=expected,
                        atol=tolerance.atol,
                        rtol=tolerance.rtol,
                        expected_engine_sha256=sha256_file(
                            arguments.runs / environment / precision / "engine.plan"
                        ),
                        expected_input_hashes=expected_input_hashes,
                        expected_count=repetitions,
                        require_tactic_diagnostic=True,
                        enforce_numerical_gates=False,
                        determinism_atol=specification.determinism.tolerance.atol,
                        determinism_rtol=specification.determinism.tolerance.rtol,
                    )
                except RuntimeError as error:
                    code = worker_evidence_failure_code(
                        result_path,
                        environment_id=environment,
                        error=error,
                    )
                    workers[environment] = {
                        "passed": False,
                        "failure_code": code.value,
                    }
                    case_evidence["status"] = status_for_failure(code).value
                    case_evidence["failure_code"] = code.value
                    evidence.append(case_evidence)
                    worker_failure = failure_record(
                        code=code,
                        phase=Phase.CORRECTNESS,
                        environment_id=environment,
                        model_id="residual-rmsnorm-plugin",
                        precision=precision_mode,
                        case_id=case_root.name,
                        output_name="output",
                        gate="worker_evidence_validation",
                        observed="worker correctness evidence failed host validation",
                        threshold="worker evidence must satisfy its retained artifact contract",
                        runs_root=arguments.runs,
                        evidence_paths=(result_path,),
                    )
                    if is_output_schema_failure(error):
                        try:
                            tolerance_stable = worker_output_tolerance_stable(
                                result_path,
                                runs_root=arguments.runs,
                                policy=specification.determinism.tolerance,
                            )
                        except RuntimeError:
                            pass
                        else:
                            case_evidence["stable_artifacts"] = {
                                environment: promote_extended_case(
                                    promotion,
                                    environment_id=environment,
                                    precision=precision_mode,
                                    case_id=f"{precision}-{case_root.name}",
                                    build_result_path=(
                                        arguments.runs / environment / precision / "build.json"
                                    ),
                                    correctness_result_path=result_path,
                                    output_root=result_root,
                                    numerical=(),
                                    determinism_tolerance_stable=tolerance_stable,
                                    failure=worker_failure,
                                )
                            }
                    _write_result(
                        arguments.output,
                        specification_sha256=specification_sha256,
                        repetitions=repetitions,
                        cases=evidence,
                        failure=worker_failure,
                        promotion=promotion,
                    )
                    raise RuntimeError(
                        f"plugin worker evidence failed for {precision}/{case_root.name}: "
                        f"{code.value}"
                    ) from error
                workers[environment] = {"passed": True, **validation}
                worker_outputs[environment] = _outputs(result_root, "output", repetitions)
            numerical_repetitions = []
            decisions: list[ThreeWayDecision] = []
            failed_decision: ThreeWayDecision | None = None
            precedence_error: ThreeWayPrecedenceError | None = None
            for index, (baseline, candidate) in enumerate(
                zip(worker_outputs["baseline"], worker_outputs["candidate"], strict=True)
            ):
                try:
                    decision = decide_three_way(
                        "output",
                        expected,
                        baseline,
                        candidate,
                        policy=policy,
                    )
                except ThreeWayPrecedenceError as error:
                    precedence_error = error
                    break
                decisions.append(decision)
                numerical_repetitions.append(_decision_evidence(index, decision))
                if decision.failure_code is FailureCode.CORPUS_INVALID:
                    if (
                        failed_decision is None
                        or failed_decision.failure_code is not FailureCode.CORPUS_INVALID
                    ):
                        failed_decision = decision
                elif decision.failure_code is FailureCode.NONFINITE_OUTPUT:
                    if failed_decision is None or failed_decision.failure_code not in {
                        FailureCode.CORPUS_INVALID,
                        FailureCode.NONFINITE_OUTPUT,
                    }:
                        failed_decision = decision
                elif not decision.passed and failed_decision is None:
                    failed_decision = decision
            case_evidence["numerical"] = {
                "effective_policy": policy.model_dump(mode="json"),
                "semantic_kind": None,
                "repetitions": numerical_repetitions,
            }
            failure_code = (
                precedence_error.failure_code
                if precedence_error is not None
                else failed_decision.failure_code
                if failed_decision
                else None
            )
            failure_phase = Phase.CORRECTNESS
            failed_gates = (
                precedence_error.failed_gates
                if precedence_error is not None
                else failed_decision.failed_gates
                if failed_decision
                else ()
            )
            if failure_code is None and (
                workers["baseline"]["tolerance_stable"] is not True
                or (
                    specification.determinism.require_bitwise
                    and workers["baseline"]["bitwise_stable"] is not True
                )
            ):
                failure_code = FailureCode.CORPUS_INVALID
                failure_phase = Phase.DETERMINISM
                failed_gates = ("baseline_determinism",)
            if failure_code is None and (
                workers["candidate"]["tolerance_stable"] is not True
                or (
                    specification.determinism.require_bitwise
                    and workers["candidate"]["bitwise_stable"] is not True
                )
            ):
                failure_code = FailureCode.NONDETERMINISM_REGRESSION
                failure_phase = Phase.DETERMINISM
                failed_gates = ("candidate_determinism",)
            case_evidence["status"] = status_for_failure(failure_code).value
            case_evidence["failure_code"] = failure_code.value if failure_code else None
            failure: FailureRecord | None = None
            if failure_code is not None:
                environment = (
                    "baseline" if failure_code is FailureCode.CORPUS_INVALID else "candidate"
                )
                failure = failure_record(
                    code=failure_code,
                    phase=failure_phase,
                    environment_id=environment,
                    model_id="residual-rmsnorm-plugin",
                    precision=precision_mode,
                    case_id=case_root.name,
                    output_name="output",
                    gate=(
                        "determinism"
                        if failure_phase is Phase.DETERMINISM
                        else "output_schema"
                        if failure_code is FailureCode.OUTPUT_SCHEMA_CHANGED
                        else "finite_outputs"
                        if failure_code is FailureCode.NONFINITE_OUTPUT
                        else "three_way_numerical"
                    ),
                    observed=",".join(failed_gates),
                    threshold="all authored numerical, semantic, and determinism gates must pass",
                    runs_root=arguments.runs,
                    evidence_paths=(
                        arguments.runs
                        / "baseline"
                        / precision
                        / case_root.name
                        / "correctness.json",
                        arguments.runs
                        / "candidate"
                        / precision
                        / case_root.name
                        / "correctness.json",
                    ),
                )
            stable_artifacts: dict[str, object] = {}
            for environment in ("baseline", "candidate"):
                environment_failure = (
                    failure
                    if failure is not None and failure.environment_id == environment
                    else None
                )
                numerical = tuple(decision.baseline_to_reference for decision in decisions)
                if environment == "candidate":
                    numerical = tuple(
                        summary
                        for decision in decisions
                        for summary in (
                            decision.candidate_to_reference,
                            decision.candidate_to_baseline,
                        )
                    )
                stable_artifacts[environment] = promote_extended_case(
                    promotion,
                    environment_id=environment,
                    precision=precision_mode,
                    case_id=f"{precision}-{case_root.name}",
                    build_result_path=(arguments.runs / environment / precision / "build.json"),
                    correctness_result_path=(
                        arguments.runs
                        / environment
                        / precision
                        / case_root.name
                        / "correctness.json"
                    ),
                    output_root=(arguments.runs / environment / precision / case_root.name),
                    numerical=numerical,
                    determinism_tolerance_stable=(workers[environment]["tolerance_stable"] is True),
                    failure=environment_failure,
                )
            case_evidence["stable_artifacts"] = stable_artifacts
            evidence.append(case_evidence)
            if failure is not None:
                _write_result(
                    arguments.output,
                    specification_sha256=specification_sha256,
                    repetitions=repetitions,
                    cases=evidence,
                    failure=failure,
                    promotion=promotion,
                )
                if failure.code is FailureCode.CORPUS_INVALID:
                    raise SystemExit(2)
                raise RuntimeError(
                    f"plugin three-way gate failed for {precision}/{case_root.name}: "
                    f"{failure.code.value}"
                )
    _write_result(
        arguments.output,
        specification_sha256=specification_sha256,
        repetitions=repetitions,
        cases=evidence,
        failure=None,
        promotion=promotion,
    )


if __name__ == "__main__":
    main()
