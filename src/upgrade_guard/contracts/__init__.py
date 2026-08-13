"""Versioned data contracts used at the host and worker boundary."""

from upgrade_guard.contracts.build import BuildManifest
from upgrade_guard.contracts.bundle import BundleManifest
from upgrade_guard.contracts.case import CaseManifest
from upgrade_guard.contracts.environment import EnvironmentLock, MatrixLock
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
    "MatrixLock",
    "MatrixSpec",
    "QualificationSpec",
    "ReferenceEnvironmentLock",
    "RunResult",
]
