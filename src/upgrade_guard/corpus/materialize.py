"""Atomic, deterministic frozen corpus materialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

from upgrade_guard.contracts.base import canonical_json_bytes, sha256_bytes, sha256_file
from upgrade_guard.corpus.generators import generate_tiny_transformer
from upgrade_guard.corpus.reference import (
    deterministic_transformer_inputs,
    run_onnx_reference,
    save_inputs,
)
from upgrade_guard.corpus.registry import CorpusLock, CorpusRecipe, MaterializedArtifact
from upgrade_guard.errors import InvalidInputError


def load_recipe(path: Path) -> CorpusRecipe:
    """Load a strict YAML corpus recipe."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CorpusRecipe.model_validate(raw)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as error:
        raise InvalidInputError(
            "corpus recipe is invalid",
            details={"path": str(path), "reason": str(error)},
        ) from error


def materialize_corpus(recipe_path: Path, destination: Path) -> CorpusLock:
    """Generate, hash, reference-run, and atomically publish the corpus."""

    if destination.exists():
        raise InvalidInputError(
            "corpus destination already exists",
            details={"path": str(destination)},
        )
    recipe = load_recipe(recipe_path)
    recipe_hash = sha256_bytes(canonical_json_bytes(recipe.model_dump(mode="json")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "corpus"
        staging.mkdir()
        artifacts: list[MaterializedArtifact] = []
        models: dict[str, Path] = {}
        for precision in recipe.precisions:
            relative = Path("models") / f"tiny-transformer-{precision}.onnx"
            generated = generate_tiny_transformer(staging / relative, precision=precision)
            expected = recipe.expected_model_sha256[precision]
            if generated.sha256 != expected:
                raise InvalidInputError(
                    "generated model hash differs from corpus lock",
                    details={
                        "precision": precision,
                        "expected": expected,
                        "observed": generated.sha256,
                    },
                )
            models[precision] = generated.path
            artifacts.append(_artifact(staging, generated.path, "application/onnx"))
        for shape in recipe.transformer_shapes:
            for precision in recipe.precisions:
                relative = Path("inputs") / f"tiny-transformer-{precision}" / shape.id
                inputs = deterministic_transformer_inputs(
                    shape.batch,
                    shape.sequence,
                    precision=precision,
                )
                save_inputs(staging / relative, inputs)
                reference = run_onnx_reference(models[precision], inputs)
                for path in sorted((staging / relative).iterdir()):
                    artifacts.append(_artifact(staging, path, "application/x-npy"))
                reference_path = (
                    staging / "reference" / f"tiny-transformer-{precision}-{shape.id}.json"
                )
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                reference_path.write_text(
                    json.dumps(
                        [
                            {
                                "name": item.name,
                                "dtype": item.dtype,
                                "shape": item.shape,
                                "sha256": item.sha256,
                            }
                            for item in reference
                        ],
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                artifacts.append(_artifact(staging, reference_path, "application/json"))
        lock = CorpusLock(
            api_version="upgradeguard.dev/v1alpha1",
            kind="CorpusLock",
            id=recipe.id,
            recipe_sha256=recipe_hash,
            artifacts=tuple(sorted(artifacts, key=lambda item: item.path)),
        )
        (staging / "corpus.lock.json").write_text(
            lock.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
        return lock


def _artifact(root: Path, path: Path, media_type: str) -> MaterializedArtifact:
    return MaterializedArtifact(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        media_type=media_type,
    )
