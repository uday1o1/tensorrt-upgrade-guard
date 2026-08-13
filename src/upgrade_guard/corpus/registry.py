"""Strict corpus recipe and generated lock contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.environment import Sha256Digest


class TransformerShape(StrictModel):
    """One concrete transformer shape."""

    id: str
    batch: int = Field(ge=1, le=8)
    sequence: int = Field(ge=8, le=512)
    weight: float = Field(gt=0, le=1)


class CorpusRecipe(StrictModel):
    """Authored, reviewable corpus recipe."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["CorpusRecipe"]
    id: str
    generator_version: Literal["tiny-transformer-v1"]
    precisions: tuple[Literal["fp32", "fp16"], ...]
    transformer_shapes: tuple[TransformerShape, ...]
    expected_model_sha256: dict[Literal["fp32", "fp16"], Sha256Digest]

    @model_validator(mode="after")
    def validate_recipe(self) -> CorpusRecipe:
        if set(self.precisions) != set(self.expected_model_sha256):
            raise ValueError("every precision requires exactly one expected model hash")
        identifiers = [shape.id for shape in self.transformer_shapes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("shape identifiers must be unique")
        if abs(sum(shape.weight for shape in self.transformer_shapes) - 1.0) > 1e-9:
            raise ValueError("transformer shape weights must sum to one")
        return self


class MaterializedArtifact(StrictModel):
    """One immutable generated file."""

    path: str
    sha256: Sha256Digest
    bytes: int = Field(ge=0)
    media_type: str


class CorpusLock(StrictModel):
    """Complete generated corpus identity."""

    api_version: Literal["upgradeguard.dev/v1alpha1"]
    kind: Literal["CorpusLock"]
    id: str
    recipe_sha256: Sha256Digest
    artifacts: tuple[MaterializedArtifact, ...]
