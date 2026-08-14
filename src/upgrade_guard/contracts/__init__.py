"""Versioned data contracts used at the host and worker boundary."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from upgrade_guard.contracts.build import BuildManifest
    from upgrade_guard.contracts.bundle import BundleManifest
    from upgrade_guard.contracts.case import CaseManifest
    from upgrade_guard.contracts.environment import EnvironmentLock, MatrixLock
    from upgrade_guard.contracts.extended import (
        ExtendedCorpusManifest,
        ExtendedInvocationManifest,
    )
    from upgrade_guard.contracts.matrix import EnvironmentRequest, MatrixSpec
    from upgrade_guard.contracts.qualification import QualificationSpec
    from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock
    from upgrade_guard.contracts.results import RunResult

__all__ = [
    "BuildManifest",
    "BundleManifest",
    "CaseManifest",
    "EnvironmentLock",
    "EnvironmentRequest",
    "ExtendedCorpusManifest",
    "ExtendedInvocationManifest",
    "MatrixLock",
    "MatrixSpec",
    "QualificationSpec",
    "ReferenceEnvironmentLock",
    "RunResult",
]

_EXPORTS = {
    "BuildManifest": ("upgrade_guard.contracts.build", "BuildManifest"),
    "BundleManifest": ("upgrade_guard.contracts.bundle", "BundleManifest"),
    "CaseManifest": ("upgrade_guard.contracts.case", "CaseManifest"),
    "EnvironmentLock": ("upgrade_guard.contracts.environment", "EnvironmentLock"),
    "EnvironmentRequest": ("upgrade_guard.contracts.matrix", "EnvironmentRequest"),
    "ExtendedCorpusManifest": (
        "upgrade_guard.contracts.extended",
        "ExtendedCorpusManifest",
    ),
    "ExtendedInvocationManifest": (
        "upgrade_guard.contracts.extended",
        "ExtendedInvocationManifest",
    ),
    "MatrixLock": ("upgrade_guard.contracts.environment", "MatrixLock"),
    "MatrixSpec": ("upgrade_guard.contracts.matrix", "MatrixSpec"),
    "QualificationSpec": ("upgrade_guard.contracts.qualification", "QualificationSpec"),
    "ReferenceEnvironmentLock": (
        "upgrade_guard.contracts.reference_environment",
        "ReferenceEnvironmentLock",
    ),
    "RunResult": ("upgrade_guard.contracts.results", "RunResult"),
}


def __getattr__(name: str) -> Any:
    """Load public contracts without creating package import cycles."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
