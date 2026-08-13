"""Frozen model, input, profile, and reference case contract."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.common import (
    ArtifactReference,
    DeterminismPolicy,
    NumericalPolicy,
    PrecisionMode,
    TensorContract,
)
from upgrade_guard.contracts.environment import Sha256Digest


class SourceAttribution(StrictModel):
    """Model or input source and redistribution facts."""

    name: str
    source_url: str
    source_revision: str
    license_name: str
    license_url: str
    redistribution_allowed: bool


class ReferenceCapability(StrictModel):
    """Proof that the exact frozen artifact executes under its reference."""

    supported: bool
    execution_provider: str
    observed_input_dtypes: dict[str, str]
    observed_output_dtypes: dict[str, str]
    reason: str | None = None

    @model_validator(mode="after")
    def require_reason_for_exclusion(self) -> ReferenceCapability:
        if not self.supported and not self.reason:
            raise ValueError("unsupported reference capabilities require a reason")
        return self


class CaseManifest(StrictModel):
    """One executable frozen qualification case."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["CaseManifest"]
    id: str
    model_id: str
    source: SourceAttribution
    model: ArtifactReference
    external_data: tuple[ArtifactReference, ...] = ()
    opset: int = Field(gt=0)
    ir_version: int = Field(gt=0)
    exporter_environment_sha256: Sha256Digest
    precision: PrecisionMode
    profile_id: str
    shape_id: str
    inputs: tuple[TensorContract, ...]
    input_fixtures: tuple[ArtifactReference, ...]
    outputs: tuple[TensorContract, ...]
    reference_runner: Literal["onnxruntime_cpu", "project_formula"]
    reference_environment_sha256: Sha256Digest
    reference_capability: ReferenceCapability
    numerical: NumericalPolicy
    determinism: DeterminismPolicy
    workload_weight: float = Field(gt=0, le=1)
    semantic_policy: dict[str, str] = Field(default_factory=dict)
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_unique_tensor_names(self) -> CaseManifest:
        for label, tensors in (("inputs", self.inputs), ("outputs", self.outputs)):
            names = [tensor.name for tensor in tensors]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} tensor names must be unique")
        if not self.reference_capability.supported:
            raise ValueError("unsupported reference artifacts cannot enter qualification")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"manifest_sha256"})
