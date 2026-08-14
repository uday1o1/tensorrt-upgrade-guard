"""Stable failure fixtures, precedence, status, and exit-code tests."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from upgrade_guard.classify import (
    CaseSignals,
    classify_observed_facts,
    classify_signals,
    classify_worker_error,
    exit_code_for_failure,
    status_for_failure,
)
from upgrade_guard.contracts.common import FailureRecord, ResultStatus
from upgrade_guard.errors import ExitCode, FailureCode

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "results" / "failure-codes.json"


def test_every_failure_code_has_a_strict_stored_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert set(payload) == {code.value for code in FailureCode}
    for code, fixture in payload.items():
        record = FailureRecord.model_validate(fixture)
        assert record.code.value == code


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("preflight_supported", FailureCode.PREFLIGHT_UNSUPPORTED),
        ("corpus_valid", FailureCode.CORPUS_INVALID),
        ("plugin_compiled", FailureCode.PLUGIN_COMPILE_FAILED),
        ("parsed", FailureCode.ONNX_PARSE_FAILED),
        ("built", FailureCode.ENGINE_BUILD_FAILED),
        ("deserialized", FailureCode.ENGINE_DESERIALIZE_FAILED),
        ("profile_accepted", FailureCode.PROFILE_REJECTED),
        ("executed", FailureCode.EXECUTION_FAILED),
        ("output_schema_valid", FailureCode.OUTPUT_SCHEMA_CHANGED),
        ("finite", FailureCode.NONFINITE_OUTPUT),
        ("numerical_valid", FailureCode.NUMERICAL_REGRESSION),
        ("deterministic", FailureCode.NONDETERMINISM_REGRESSION),
        ("performance_valid", FailureCode.PERFORMANCE_REGRESSION),
        ("memory_valid", FailureCode.MEMORY_REGRESSION),
        ("sanitizer_valid", FailureCode.SANITIZER_FAILURE),
        ("infrastructure_valid", FailureCode.INFRASTRUCTURE_INVALID),
        ("conclusive", FailureCode.INCONCLUSIVE),
    ],
)
def test_every_signal_maps_to_its_stable_failure(
    field_name: str,
    expected: FailureCode,
) -> None:
    defaults = {field.name: True for field in fields(CaseSignals)}
    defaults[field_name] = False
    assert classify_signals(CaseSignals(**defaults)) is expected


def test_precedence_uses_specific_execution_evidence_before_numerics() -> None:
    code = classify_signals(
        CaseSignals(
            output_schema_valid=False,
            finite=False,
            numerical_valid=False,
            infrastructure_valid=False,
        )
    )
    assert code is FailureCode.OUTPUT_SCHEMA_CHANGED


def test_observed_facts_are_strict_and_use_production_precedence() -> None:
    assert (
        classify_observed_facts({"finite": False, "numerical_valid": False})
        is FailureCode.NONFINITE_OUTPUT
    )
    with pytest.raises(ValueError, match="unknown"):
        classify_observed_facts({"numercial_valid": False})
    with pytest.raises(ValueError, match="boolean"):
        classify_observed_facts({"finite": "false"})


@pytest.mark.parametrize(
    ("phase", "message", "expected"),
    [
        ("build", "ONNX parser rejected model", FailureCode.ONNX_PARSE_FAILED),
        (
            "build",
            "freshly built engine failed same-environment reload",
            FailureCode.ENGINE_DESERIALIZE_FAILED,
        ),
        (
            "build",
            "TensorRT engine build returned no serialized engine",
            FailureCode.ENGINE_BUILD_FAILED,
        ),
        ("correctness", "input shape was rejected for x", FailureCode.PROFILE_REJECTED),
        (
            "correctness",
            "plugin load failed: undefined symbol",
            FailureCode.PLUGIN_COMPILE_FAILED,
        ),
        ("correctness", "output y contains nonfinite values", FailureCode.NONFINITE_OUTPUT),
        ("correctness", "TensorRT execute_async_v3 returned false", FailureCode.EXECUTION_FAILED),
    ],
)
def test_worker_errors_keep_specific_failure_codes(
    phase: str, message: str, expected: FailureCode
) -> None:
    assert classify_worker_error(phase, message) is expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code", "status", "exit_code"),
    [
        (None, ResultStatus.PASSED, ExitCode.SUCCESS),
        (
            FailureCode.PREFLIGHT_UNSUPPORTED,
            ResultStatus.UNSUPPORTED,
            ExitCode.UNSUPPORTED,
        ),
        (
            FailureCode.INFRASTRUCTURE_INVALID,
            ResultStatus.INFRASTRUCTURE_INVALID,
            ExitCode.INFRASTRUCTURE_INVALID,
        ),
        (
            FailureCode.INCONCLUSIVE,
            ResultStatus.INCONCLUSIVE,
            ExitCode.INFRASTRUCTURE_INVALID,
        ),
        (
            FailureCode.CORPUS_INVALID,
            ResultStatus.FAILED,
            ExitCode.INVALID_INPUT,
        ),
        (
            FailureCode.NUMERICAL_REGRESSION,
            ResultStatus.FAILED,
            ExitCode.QUALIFICATION_FAILED,
        ),
    ],
)
def test_failure_status_and_exit_mapping(
    code: FailureCode | None,
    status: ResultStatus,
    exit_code: ExitCode,
) -> None:
    assert status_for_failure(code) is status
    assert exit_code_for_failure(code) is exit_code
