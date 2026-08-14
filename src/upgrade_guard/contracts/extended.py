"""Typed provenance inputs for extended plugin and MobileNet qualification."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from upgrade_guard.contracts.base import StrictModel, model_sha256
from upgrade_guard.contracts.case import SourceAttribution
from upgrade_guard.contracts.common import ArtifactReference, PrecisionMode, TensorContract
from upgrade_guard.contracts.environment import Sha256Digest

GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
ExtendedSuite = Literal["plugin", "mobilenet"]


class ExtendedCorpusModel(StrictModel):
    """One immutable model and its authored execution identity."""

    model_id: str
    precision: PrecisionMode
    artifact: ArtifactReference
    source: SourceAttribution
    opset: int = Field(gt=0)
    ir_version: int = Field(gt=0)
    profile_id: str
    reference_runner: Literal["onnxruntime_cpu", "project_formula"]
    semantic_policy: dict[str, str]


class ExtendedCorpusCase(StrictModel):
    """One exact case input, output, shape, and workload identity."""

    id: str
    model_id: str
    precision: PrecisionMode
    shape_id: str
    profile_id: str
    inputs: tuple[TensorContract, ...]
    input_fixtures: tuple[ArtifactReference, ...]
    outputs: tuple[TensorContract, ...]
    reference_output: ArtifactReference
    workload_weight: float = Field(gt=0, le=1)


class ExtendedCorpusManifest(StrictModel):
    """Self-hashed typed case metadata retained inside an extended corpus."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["ExtendedCorpusManifest"]
    suite: ExtendedSuite
    reference_environment_sha256: Sha256Digest
    models: tuple[ExtendedCorpusModel, ...]
    cases: tuple[ExtendedCorpusCase, ...]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_inventory(self) -> ExtendedCorpusManifest:
        model_keys = [(model.model_id, model.precision) for model in self.models]
        if not model_keys or len(model_keys) != len(set(model_keys)):
            raise ValueError("extended corpus models must be nonempty and unique")
        case_keys = [(case.precision, case.id) for case in self.cases]
        if not case_keys or len(case_keys) != len(set(case_keys)):
            raise ValueError("extended corpus cases must be nonempty and unique")
        model_map = {(model.model_id, model.precision): model for model in self.models}
        for case in self.cases:
            model = model_map.get((case.model_id, case.precision))
            if model is None or case.profile_id != model.profile_id:
                raise ValueError("extended case does not bind one declared model profile")
        for precision in {case.precision for case in self.cases}:
            weight = sum(case.workload_weight for case in self.cases if case.precision is precision)
            if abs(weight - 1.0) > 1e-9:
                raise ValueError("extended case weights must sum to one per precision")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"manifest_sha256"})


class PluginBuildProvenance(StrictModel):
    """Exact source, binary, commands, and build log for one plugin environment."""

    environment_id: str
    source_inventory: tuple[ArtifactReference, ...]
    source_inventory_sha256: Sha256Digest
    binary: ArtifactReference
    compile_commands: ArtifactReference
    build_log: ArtifactReference
    configure_command: tuple[str, ...]
    build_command: tuple[str, ...]
    test_command: tuple[str, ...]


class ExtendedInvocationManifest(StrictModel):
    """Self-hashed host invocation identity for stable extended artifacts."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["ExtendedInvocationManifest"]
    suite: ExtendedSuite
    source_git_commit: GitCommit
    matrix_lock_sha256: Sha256Digest
    specification_sha256: Sha256Digest
    corpus_lock_sha256: Sha256Digest
    corpus_manifest_sha256: Sha256Digest
    environment_ids: tuple[str, ...]
    plugin_builds: tuple[PluginBuildProvenance, ...] = ()
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_inventory(self) -> ExtendedInvocationManifest:
        if not self.environment_ids or len(self.environment_ids) != len(set(self.environment_ids)):
            raise ValueError("extended invocation environments must be nonempty and unique")
        plugin_ids = tuple(item.environment_id for item in self.plugin_builds)
        if self.suite == "plugin" and set(plugin_ids) != set(self.environment_ids):
            raise ValueError("plugin invocation requires provenance for every environment")
        if self.suite == "mobilenet" and self.plugin_builds:
            raise ValueError("MobileNet invocation cannot contain plugin provenance")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"manifest_sha256"})
