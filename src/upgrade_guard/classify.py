"""Failure-code precedence and public exit mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal

from upgrade_guard.contracts.common import ResultStatus
from upgrade_guard.errors import ExitCode, FailureCode


@dataclass(frozen=True, slots=True)
class CaseSignals:
    """Ordered observations used to classify one case."""

    preflight_supported: bool = True
    corpus_valid: bool = True
    plugin_compiled: bool = True
    parsed: bool = True
    built: bool = True
    deserialized: bool = True
    profile_accepted: bool = True
    executed: bool = True
    output_schema_valid: bool = True
    finite: bool = True
    numerical_valid: bool = True
    deterministic: bool = True
    performance_valid: bool = True
    memory_valid: bool = True
    sanitizer_valid: bool = True
    infrastructure_valid: bool = True
    conclusive: bool = True


WorkerPhase = Literal["build", "correctness"]


def signals_from_observed_facts(facts: Mapping[str, object]) -> CaseSignals:
    """Validate a sparse set of observed boolean facts before classification.

    Seed fixtures and worker adapters describe what was observed. They do not
    author their own failure code. Unknown facts and non-boolean values fail
    closed so a misspelled observation cannot silently become a passing case.
    """

    known = {field.name for field in fields(CaseSignals)}
    unknown = sorted(set(facts) - known)
    if unknown:
        raise ValueError(f"unknown case signal fields: {', '.join(unknown)}")
    invalid = sorted(name for name, value in facts.items() if not isinstance(value, bool))
    if invalid:
        raise ValueError(f"case signal values must be boolean: {', '.join(invalid)}")
    return CaseSignals(**facts)  # type: ignore[arg-type]


def classify_observed_facts(facts: Mapping[str, object]) -> FailureCode | None:
    """Classify validated observed facts using the production precedence."""

    return classify_signals(signals_from_observed_facts(facts))


def classify_worker_error(phase: WorkerPhase, message: str) -> FailureCode:
    """Map bounded worker error evidence to the stable failure taxonomy."""

    normalized = message.casefold()
    if "plugin" in normalized and any(
        token in normalized for token in ("load", "compile", "undefined symbol")
    ):
        return FailureCode.PLUGIN_COMPILE_FAILED
    if phase == "build":
        if "parser" in normalized or ("onnx" in normalized and "parse" in normalized):
            return FailureCode.ONNX_PARSE_FAILED
        if "profile" in normalized and any(
            token in normalized for token in ("reject", "invalid", "failed")
        ):
            return FailureCode.PROFILE_REJECTED
        if "deserialize" in normalized or "reload" in normalized:
            return FailureCode.ENGINE_DESERIALIZE_FAILED
        return FailureCode.ENGINE_BUILD_FAILED
    if "deserialize" in normalized:
        return FailureCode.ENGINE_DESERIALIZE_FAILED
    if any(
        token in normalized
        for token in (
            "input shape was rejected",
            "dynamic tensor was specified",
            "unresolved output shape",
            "profile",
        )
    ):
        return FailureCode.PROFILE_REJECTED
    if "output" in normalized and any(
        token in normalized for token in ("name", "dtype", "shape", "schema")
    ):
        return FailureCode.OUTPUT_SCHEMA_CHANGED
    if "nonfinite" in normalized:
        return FailureCode.NONFINITE_OUTPUT
    return FailureCode.EXECUTION_FAILED


def classify_signals(signals: CaseSignals) -> FailureCode | None:
    """Return the first stable failure by required precedence."""

    precedence = (
        (not signals.preflight_supported, FailureCode.PREFLIGHT_UNSUPPORTED),
        (not signals.corpus_valid, FailureCode.CORPUS_INVALID),
        (not signals.plugin_compiled, FailureCode.PLUGIN_COMPILE_FAILED),
        (not signals.parsed, FailureCode.ONNX_PARSE_FAILED),
        (not signals.built, FailureCode.ENGINE_BUILD_FAILED),
        (not signals.deserialized, FailureCode.ENGINE_DESERIALIZE_FAILED),
        (not signals.profile_accepted, FailureCode.PROFILE_REJECTED),
        (not signals.executed, FailureCode.EXECUTION_FAILED),
        (not signals.output_schema_valid, FailureCode.OUTPUT_SCHEMA_CHANGED),
        (not signals.finite, FailureCode.NONFINITE_OUTPUT),
        (not signals.numerical_valid, FailureCode.NUMERICAL_REGRESSION),
        (not signals.deterministic, FailureCode.NONDETERMINISM_REGRESSION),
        (not signals.performance_valid, FailureCode.PERFORMANCE_REGRESSION),
        (not signals.memory_valid, FailureCode.MEMORY_REGRESSION),
        (not signals.sanitizer_valid, FailureCode.SANITIZER_FAILURE),
        (not signals.infrastructure_valid, FailureCode.INFRASTRUCTURE_INVALID),
        (not signals.conclusive, FailureCode.INCONCLUSIVE),
    )
    return next((code for failed, code in precedence if failed), None)


def status_for_failure(code: FailureCode | None) -> ResultStatus:
    if code is None:
        return ResultStatus.PASSED
    if code is FailureCode.PREFLIGHT_UNSUPPORTED:
        return ResultStatus.UNSUPPORTED
    if code is FailureCode.INFRASTRUCTURE_INVALID:
        return ResultStatus.INFRASTRUCTURE_INVALID
    if code is FailureCode.INCONCLUSIVE:
        return ResultStatus.INCONCLUSIVE
    return ResultStatus.FAILED


def exit_code_for_failure(code: FailureCode | None) -> ExitCode:
    if code is None:
        return ExitCode.SUCCESS
    if code is FailureCode.PREFLIGHT_UNSUPPORTED:
        return ExitCode.UNSUPPORTED
    if code in {FailureCode.INFRASTRUCTURE_INVALID, FailureCode.INCONCLUSIVE}:
        return ExitCode.INFRASTRUCTURE_INVALID
    if code is FailureCode.CORPUS_INVALID:
        return ExitCode.INVALID_INPUT
    return ExitCode.QUALIFICATION_FAILED
