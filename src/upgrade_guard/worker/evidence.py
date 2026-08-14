"""Validation of retained worker correctness evidence."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from upgrade_guard.contracts.base import sha256_file


def validate_repetitions(
    *,
    result_path: Path,
    runs_root: Path,
    expected_output_name: str,
    expected: np.ndarray[Any, Any],
    atol: float,
    rtol: float,
    expected_engine_sha256: str,
    expected_input_hashes: dict[str, str],
    expected_count: int = 20,
) -> dict[str, object]:
    """Validate result schema, artifact hashes, and numerical stability."""

    result = _json_object(result_path)
    if result.get("schema_version") != "upgradeguard.dev/worker-correctness/v1":
        raise RuntimeError(f"worker result schema changed: {result_path}")
    if result.get("status") != "passed":
        raise RuntimeError(f"worker result did not pass: {result_path}")
    if result.get("engine_sha256") != expected_engine_sha256:
        raise RuntimeError(f"worker engine hash changed: {result_path}")
    if result.get("input_sha256") != expected_input_hashes:
        raise RuntimeError(f"worker input hashes changed: {result_path}")
    repetitions = result.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != expected_count:
        raise RuntimeError(f"worker retained an unexpected repetition count: {result_path}")

    reference: np.ndarray[Any, Any] | None = None
    output_hashes: list[str] = []
    maximum_absolute_error = 0.0
    maximum_determinism_error = 0.0
    for expected_index, repetition in enumerate(repetitions):
        if not isinstance(repetition, dict) or repetition.get("index") != expected_index:
            raise RuntimeError(f"worker repetition indexes changed: {result_path}")
        outputs = repetition.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError(f"worker output inventory changed: {result_path}")
        output = outputs[0]
        if not isinstance(output, dict) or output.get("name") != expected_output_name:
            raise RuntimeError(f"worker output name changed: {result_path}")
        observed_path = _output_path(output.get("path"), runs_root)
        recorded_hash = output.get("sha256")
        if not isinstance(recorded_hash, str) or sha256_file(observed_path) != recorded_hash:
            raise RuntimeError(f"worker output hash changed: {observed_path}")
        observed = np.load(observed_path, allow_pickle=False)
        if output.get("dtype") != str(observed.dtype) or output.get("shape") != list(
            observed.shape
        ):
            raise RuntimeError(f"worker output metadata changed: {observed_path}")
        if observed.dtype != expected.dtype or observed.shape != expected.shape:
            raise RuntimeError(f"worker output schema differs from the reference: {observed_path}")
        if not np.all(np.isfinite(observed)):
            raise RuntimeError(f"worker output contains nonfinite values: {observed_path}")
        absolute = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
        if not bool(np.all(absolute <= atol + rtol * np.abs(expected.astype(np.float64)))):
            raise RuntimeError(f"worker numerical gate failed: {observed_path}")
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(absolute, initial=0.0)))
        if reference is None:
            reference = observed
        else:
            determinism_error = np.abs(observed.astype(np.float64) - reference.astype(np.float64))
            maximum_determinism_error = max(
                maximum_determinism_error,
                float(np.max(determinism_error, initial=0.0)),
            )
            if not bool(
                np.all(determinism_error <= atol + rtol * np.abs(reference.astype(np.float64)))
            ):
                raise RuntimeError(f"worker output is not tolerance-stable: {observed_path}")
        output_hashes.append(recorded_hash)
    return {
        "repetitions": len(repetitions),
        "output_sha256": output_hashes,
        "bitwise_stable": len(set(output_hashes)) == 1,
        "tolerance_stable": True,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_determinism_error": maximum_determinism_error,
    }


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"worker result is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"worker result is not a JSON object: {path}")
    return value


def _output_path(value: object, runs_root: Path) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("worker output path is not a string")
    container_path = PurePosixPath(value)
    output_root = PurePosixPath("/output")
    if not container_path.is_relative_to(output_root) or ".." in container_path.parts:
        raise RuntimeError(f"worker output path escaped /output: {value}")
    relative = container_path.relative_to(output_root)
    candidate = (runs_root / Path(*relative.parts)).resolve(strict=True)
    resolved_root = runs_root.resolve(strict=True)
    if not candidate.is_relative_to(resolved_root) or candidate.suffix != ".npy":
        raise RuntimeError(f"worker output path escaped the run directory: {value}")
    return candidate
