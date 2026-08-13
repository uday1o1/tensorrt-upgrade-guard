"""Machine-readable host preflight contract."""

from __future__ import annotations

from typing import Literal

from upgrade_guard.contracts.base import StrictModel


class DoctorIssue(StrictModel):
    """One stable preflight finding."""

    code: str
    category: Literal["unsupported", "infrastructure"]
    message: str
    evidence: str | None = None


class DoctorGpu(StrictModel):
    """Host-visible GPU summary used before a worker is selected."""

    name: str
    uuid: str
    compute_capability: str
    vram_mib: int
    driver_version: str


class DoctorDocker(StrictModel):
    """Docker client and server summary."""

    available: bool
    client_version: str | None = None
    server_version: str | None = None
    server_os: str | None = None
    server_architecture: str | None = None
    runtimes: tuple[str, ...] = ()
    context: str | None = None


class DoctorResult(StrictModel):
    """Complete host preflight result."""

    schema_version: Literal["upgradeguard.dev/doctor/v1"]
    outcome: Literal["supported", "unsupported", "infrastructure_invalid"]
    host_os: str
    host_release: str
    host_architecture: str
    python_version: str
    docker: DoctorDocker
    gpus: tuple[DoctorGpu, ...]
    issues: tuple[DoctorIssue, ...]
