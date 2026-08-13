"""Failure-code precedence and public exit mapping."""

from __future__ import annotations

from dataclasses import dataclass

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
