"""Pinned CPU reference execution and project-owned plugin formula."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

from upgrade_guard.contracts.base import sha256_bytes, sha256_file
from upgrade_guard.errors import InvalidInputError

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class ReferenceOutput:
    """A named, finite, hash-addressed CPU reference output."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    values: Array


def deterministic_transformer_inputs(
    batch: int,
    sequence: int,
    *,
    precision: str = "fp32",
    seed: int = 20260813,
) -> dict[str, Array]:
    """Create reproducible real input and mask tensors for one concrete shape."""

    if batch < 1 or sequence < 1:
        raise InvalidInputError("batch and sequence must be positive")
    dtype = np.float32 if precision == "fp32" else np.float16
    rng = np.random.Generator(np.random.PCG64(seed + batch * 10_000 + sequence))
    tokens = rng.normal(0.0, 0.5, size=(batch, sequence, 256)).astype(dtype)
    mask = np.zeros((batch, 1, 1, sequence), dtype=dtype)
    return {"tokens": tokens, "mask": mask}


def run_onnx_reference(model: Path, inputs: dict[str, Array]) -> tuple[ReferenceOutput, ...]:
    """Run the exact model with ORT CPU and reject schema or finite-value drift."""

    if not model.is_file():
        raise InvalidInputError("reference model does not exist", details={"path": str(model)})
    session = ort.InferenceSession(
        str(model),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )
    expected = {item.name: item.type for item in session.get_inputs()}
    observed = {name: _ort_type(value.dtype) for name, value in inputs.items()}
    if observed != expected:
        raise InvalidInputError(
            "reference input schema differs from frozen model",
            details={"expected": expected, "observed": observed},
        )
    values = session.run(None, inputs)
    outputs: list[ReferenceOutput] = []
    for contract, value in zip(session.get_outputs(), values, strict=True):
        array = np.asarray(value)
        nonfinite = int(array.size - np.count_nonzero(np.isfinite(array)))
        if nonfinite:
            raise InvalidInputError(
                "reference output contains nonfinite values",
                details={"output": contract.name, "count": nonfinite},
            )
        outputs.append(
            ReferenceOutput(
                name=contract.name,
                dtype=str(array.dtype),
                shape=tuple(int(item) for item in array.shape),
                sha256=sha256_bytes(array.tobytes(order="C")),
                values=array,
            )
        )
    return tuple(outputs)


def save_inputs(directory: Path, inputs: dict[str, Array]) -> dict[str, str]:
    """Save exact NPY tensors and return content hashes."""

    directory.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, value in sorted(inputs.items()):
        path = directory / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        hashes[path.name] = sha256_file(path)
    return hashes


def residual_rmsnorm_reference(
    x: Array,
    residual: Array,
    gamma: Array,
    *,
    epsilon: float,
) -> Array:
    """Compute the plugin contract with FP32 accumulation."""

    if x.shape != residual.shape or x.ndim not in (2, 3):
        raise InvalidInputError("x and residual must have equal rank-2 or rank-3 shapes")
    if gamma.dtype != np.float32 or gamma.shape != (x.shape[-1],):
        raise InvalidInputError("gamma must be FP32 and match the hidden dimension")
    if x.dtype not in (np.dtype(np.float16), np.dtype(np.float32)) or residual.dtype != x.dtype:
        raise InvalidInputError("x and residual must share FP16 or FP32 dtype")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise InvalidInputError("epsilon must be finite and positive")
    combined = np.asarray(x, dtype=np.float32) + np.asarray(residual, dtype=np.float32)
    rms = np.sqrt(np.mean(combined * combined, axis=-1, keepdims=True) + epsilon)
    result = combined * np.asarray(gamma, dtype=np.float32) / rms
    return result.astype(x.dtype)


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def _ort_type(dtype: np.dtype[np.generic]) -> str:
    mapping = {
        np.dtype(np.float16): "tensor(float16)",
        np.dtype(np.float32): "tensor(float)",
    }
    result = mapping.get(dtype)
    if result is None:
        raise InvalidInputError("unsupported reference input dtype", details={"dtype": str(dtype)})
    return result
