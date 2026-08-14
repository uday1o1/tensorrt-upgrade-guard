"""ResidualRMSNorm micrograph cases and project-owned references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.case import SourceAttribution
from upgrade_guard.contracts.common import ArtifactReference, PrecisionMode, TensorContract
from upgrade_guard.contracts.extended import (
    ExtendedCorpusCase,
    ExtendedCorpusManifest,
    ExtendedCorpusModel,
)
from upgrade_guard.corpus.generators import generate_plugin_micrograph
from upgrade_guard.corpus.reference import residual_rmsnorm_reference
from upgrade_guard.errors import InvalidInputError


@dataclass(frozen=True)
class PluginCase:
    """One bounded dynamic plugin case."""

    id: str
    shape: tuple[int, int, int]
    values: str
    noncontiguous_generation: bool = False


CASES = (
    PluginCase("minimum-zero-h7", (1, 1, 7), "zero"),
    PluginCase("aligned-random-h256", (2, 17, 256), "random"),
    PluginCase("tail-random-h259", (2, 17, 259), "random"),
    PluginCase("large-finite-h32", (1, 8, 32), "large"),
    PluginCase("noncontiguous-h63", (2, 9, 63), "random", True),
    PluginCase("maximum-h256", (8, 512, 256), "random"),
)


def materialize_plugin_corpus(
    destination: Path,
    *,
    reference_environment_sha256: str,
) -> dict[str, object]:
    """Create FP32 and FP16 micrographs, inputs, references, and a complete lock."""

    if destination.exists():
        raise InvalidInputError("plugin corpus destination already exists")
    destination.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    models: list[ExtendedCorpusModel] = []
    cases: list[ExtendedCorpusCase] = []
    for precision in ("fp32", "fp16"):
        model = destination / f"residual-rmsnorm-{precision}.onnx"
        generated = generate_plugin_micrograph(model, precision=precision)
        artifacts.append(_artifact(destination, model))
        precision_mode = PrecisionMode.FP32 if precision == "fp32" else PrecisionMode.EXPLICIT_FP16
        tensor_dtype = "float32" if precision == "fp32" else "float16"
        model_id = f"residual-rmsnorm-plugin-{precision}"
        models.append(
            ExtendedCorpusModel(
                model_id=model_id,
                precision=precision_mode,
                artifact=_artifact_reference(destination, model),
                source=SourceAttribution(
                    name="Project-owned ResidualRMSNorm plugin micrograph",
                    source_url="https://github.com/uday1o1/tensorrt-upgrade-guard",
                    source_revision=generated.sha256,
                    license_name="Apache-2.0",
                    license_url="https://www.apache.org/licenses/LICENSE-2.0",
                    redistribution_allowed=True,
                ),
                opset=generated.opset,
                ir_version=generated.ir_version,
                profile_id="residual-rmsnorm-dynamic",
                reference_runner="project_formula",
                semantic_policy={"comparison": "elementwise"},
            )
        )
        for case in CASES:
            case_root = destination / precision / case.id
            case_root.mkdir(parents=True)
            x, residual, gamma = _inputs(case, precision)
            expected = residual_rmsnorm_reference(x, residual, gamma, epsilon=1e-5)
            repeated = residual_rmsnorm_reference(x, residual, gamma, epsilon=1e-5)
            if not np.array_equal(expected, repeated):
                raise InvalidInputError("plugin reference is not bitwise deterministic")
            for name, value in (
                ("x", x),
                ("residual", residual),
                ("gamma", gamma),
                ("expected", expected),
            ):
                path = case_root / f"{name}.npy"
                np.save(path, value, allow_pickle=False)
                artifacts.append(_artifact(destination, path))
            cases.append(
                ExtendedCorpusCase(
                    id=f"{precision}-{case.id}",
                    model_id=model_id,
                    precision=precision_mode,
                    shape_id=case.id,
                    profile_id="residual-rmsnorm-dynamic",
                    inputs=(
                        TensorContract(
                            name="x", dtype=cast(Any, tensor_dtype), shape=tuple(x.shape)
                        ),
                        TensorContract(
                            name="residual",
                            dtype=cast(Any, tensor_dtype),
                            shape=tuple(residual.shape),
                        ),
                        TensorContract(name="gamma", dtype="float32", shape=tuple(gamma.shape)),
                    ),
                    input_fixtures=tuple(
                        _artifact_reference(destination, case_root / f"{name}.npy")
                        for name in ("x", "residual", "gamma")
                    ),
                    outputs=(
                        TensorContract(
                            name="output",
                            dtype=cast(Any, tensor_dtype),
                            shape=tuple(expected.shape),
                        ),
                    ),
                    reference_output=_artifact_reference(destination, case_root / "expected.npy"),
                    workload_weight=1.0 / len(CASES),
                )
            )
            metadata = case_root / "case.json"
            metadata.write_text(
                json.dumps(
                    {
                        "id": case.id,
                        "shape": case.shape,
                        "precision": precision,
                        "values": case.values,
                        "noncontiguous_generation": case.noncontiguous_generation,
                        "epsilon": 1e-5,
                        "reference_repetitions": 2,
                        "reference_bitwise_deterministic": True,
                    },
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts.append(_artifact(destination, metadata))
    zero = "sha256:" + ("0" * 64)
    manifest = ExtendedCorpusManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ExtendedCorpusManifest",
        suite="plugin",
        reference_environment_sha256=reference_environment_sha256,
        models=tuple(models),
        cases=tuple(cases),
        manifest_sha256=zero,
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
    manifest_path = destination / "extended-corpus-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(_artifact(destination, manifest_path))
    lock: dict[str, object] = {
        "schema_version": "upgradeguard.dev/plugin-corpus/v1",
        "cases": [case.id for case in CASES],
        "precisions": ["fp32", "fp16"],
        "reference_environment_sha256": reference_environment_sha256,
        "extended_manifest": {
            "path": manifest_path.relative_to(destination).as_posix(),
            "sha256": sha256_file(manifest_path),
            "manifest_sha256": manifest.manifest_sha256,
        },
        "artifacts": sorted(artifacts, key=lambda item: str(item["path"])),
    }
    (destination / "plugin-corpus.lock.json").write_text(
        json.dumps(lock, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock


def _inputs(case: PluginCase, precision: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.float32 if precision == "fp32" else np.float16
    if case.values == "zero":
        x = np.zeros(case.shape, dtype=dtype)
        residual = np.zeros(case.shape, dtype=dtype)
    else:
        generator = np.random.Generator(
            np.random.PCG64(20260813 + sum(case.shape) + (0 if precision == "fp32" else 1))
        )
        scale = 1000.0 if case.values == "large" else 0.5
        if case.noncontiguous_generation:
            backing_shape = (case.shape[0], case.shape[2], case.shape[1] * 2)
            backing_x = generator.uniform(-scale, scale, size=backing_shape).astype(dtype)
            backing_residual = generator.uniform(-scale, scale, size=backing_shape).astype(dtype)
            x = np.transpose(backing_x[:, :, ::2], (0, 2, 1))
            residual = np.transpose(backing_residual[:, :, ::2], (0, 2, 1))
            if x.flags.c_contiguous or residual.flags.c_contiguous:
                raise AssertionError("noncontiguous fixture generation lost its intended layout")
        else:
            x = generator.uniform(-scale, scale, size=case.shape).astype(dtype)
            residual = generator.uniform(-scale, scale, size=case.shape).astype(dtype)
    gamma = np.linspace(0.5, 1.5, case.shape[-1], dtype=np.float32)
    return x, residual, gamma


def _artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": _media_type(path),
    }


def _artifact_reference(root: Path, path: Path) -> ArtifactReference:
    return ArtifactReference.model_validate(_artifact(root, path))


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".npy": "application/x-npy",
        ".onnx": "application/onnx",
    }.get(path.suffix, "application/octet-stream")
