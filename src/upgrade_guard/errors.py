"""Stable host-side errors and public exit codes."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any


class ExitCode(IntEnum):
    """Public CLI exit codes."""

    SUCCESS = 0
    QUALIFICATION_FAILED = 1
    INVALID_INPUT = 2
    UNSUPPORTED = 3
    INFRASTRUCTURE_INVALID = 4
    INTERNAL_ERROR = 5


class FailureCode(StrEnum):
    """Stable V1 qualification failure taxonomy."""

    PREFLIGHT_UNSUPPORTED = "PREFLIGHT_UNSUPPORTED"
    CORPUS_INVALID = "CORPUS_INVALID"
    PLUGIN_COMPILE_FAILED = "PLUGIN_COMPILE_FAILED"
    ONNX_PARSE_FAILED = "ONNX_PARSE_FAILED"
    ENGINE_BUILD_FAILED = "ENGINE_BUILD_FAILED"
    ENGINE_DESERIALIZE_FAILED = "ENGINE_DESERIALIZE_FAILED"
    PROFILE_REJECTED = "PROFILE_REJECTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    OUTPUT_SCHEMA_CHANGED = "OUTPUT_SCHEMA_CHANGED"
    NONFINITE_OUTPUT = "NONFINITE_OUTPUT"
    NUMERICAL_REGRESSION = "NUMERICAL_REGRESSION"
    NONDETERMINISM_REGRESSION = "NONDETERMINISM_REGRESSION"
    PERFORMANCE_REGRESSION = "PERFORMANCE_REGRESSION"
    MEMORY_REGRESSION = "MEMORY_REGRESSION"
    SANITIZER_FAILURE = "SANITIZER_FAILURE"
    INFRASTRUCTURE_INVALID = "INFRASTRUCTURE_INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class UpgradeGuardError(Exception):
    """Base class for expected failures with a stable public code."""

    exit_code = ExitCode.INTERNAL_ERROR
    error_code = "INTERNAL_TOOL_FAILURE"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidInputError(UpgradeGuardError):
    """An authored specification or artifact is invalid."""

    exit_code = ExitCode.INVALID_INPUT
    error_code = "INVALID_INPUT"


class UnsupportedEnvironmentError(UpgradeGuardError):
    """The requested host, image, or tool combination is unsupported."""

    exit_code = ExitCode.UNSUPPORTED
    error_code = "PREFLIGHT_UNSUPPORTED"


class InfrastructureError(UpgradeGuardError):
    """Infrastructure prevented a conclusive operation."""

    exit_code = ExitCode.INFRASTRUCTURE_INVALID
    error_code = "INFRASTRUCTURE_INVALID"
