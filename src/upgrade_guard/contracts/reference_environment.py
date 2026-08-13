"""Independent reference-runner environment lock."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.environment import ResolvedImage, Sha256Digest


class ReferenceEnvironmentLock(StrictModel):
    """Immutable CPU reference environment independent of both workers."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["ReferenceEnvironmentLock"]
    id: str
    image: ResolvedImage
    operating_system: str
    architecture: str
    python: str
    onnx: str
    onnxruntime: str
    execution_provider: Literal["CPUExecutionProvider"]
    provider_options: dict[str, str]
    numpy: str
    pytorch: str | None
    intra_op_threads: int
    inter_op_threads: int
    probe_command_sha256: Sha256Digest
    probe_output_sha256: Sha256Digest
    probed_at: AwareDatetime
    lock_sha256: Sha256Digest

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"lock_sha256"})
