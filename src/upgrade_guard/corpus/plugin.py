"""ResidualRMSNorm micrograph cases and project-owned references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from upgrade_guard.contracts.base import sha256_file
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


def materialize_plugin_corpus(destination: Path) -> dict[str, object]:
    """Create FP32 and FP16 micrographs, inputs, references, and a complete lock."""

    if destination.exists():
        raise InvalidInputError("plugin corpus destination already exists")
    destination.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    for precision in ("fp32", "fp16"):
        model = destination / f"residual-rmsnorm-{precision}.onnx"
        generate_plugin_micrograph(model, precision=precision)
        artifacts.append(_artifact(destination, model))
        for case in CASES:
            case_root = destination / precision / case.id
            case_root.mkdir(parents=True)
            x, residual, gamma = _inputs(case, precision)
            expected = residual_rmsnorm_reference(x, residual, gamma, epsilon=1e-5)
            for name, value in (
                ("x", x),
                ("residual", residual),
                ("gamma", gamma),
                ("expected", expected),
            ):
                path = case_root / f"{name}.npy"
                np.save(path, value, allow_pickle=False)
                artifacts.append(_artifact(destination, path))
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
                    },
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts.append(_artifact(destination, metadata))
    lock: dict[str, object] = {
        "schema_version": "upgradeguard.dev/plugin-corpus/v1",
        "cases": [case.id for case in CASES],
        "precisions": ["fp32", "fp16"],
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
    }
