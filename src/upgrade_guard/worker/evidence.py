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
    require_tactic_diagnostic: bool = False,
    enforce_numerical_gates: bool = True,
    determinism_atol: float | None = None,
    determinism_rtol: float | None = None,
) -> dict[str, object]:
    """Validate retained artifacts and optionally enforce numerical gates immediately.

    Three-way callers disable immediate enforcement so baseline/reference failures can
    be classified as corpus failures and candidate failures as regressions. Artifact,
    schema, hash, input-integrity, and tactic checks always remain fail-closed.
    """

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
    retained_input_integrity = result.get("input_integrity_stable")
    if retained_input_integrity not in (None, True):
        raise RuntimeError(f"worker reported unstable input integrity: {result_path}")
    input_integrity_verified = retained_input_integrity is True
    maximum_absolute_error = 0.0
    maximum_determinism_error = 0.0
    tolerance_stable = True
    determinism_atol = atol if determinism_atol is None else determinism_atol
    determinism_rtol = rtol if determinism_rtol is None else determinism_rtol
    for expected_index, repetition in enumerate(repetitions):
        if not isinstance(repetition, dict) or repetition.get("index") != expected_index:
            raise RuntimeError(f"worker repetition indexes changed: {result_path}")
        if input_integrity_verified:
            _validate_input_integrity(
                repetition.get("inputs"), expected_input_hashes, result_path=result_path
            )
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
        observed_finite = bool(np.all(np.isfinite(observed)))
        if enforce_numerical_gates and not observed_finite:
            raise RuntimeError(f"worker output contains nonfinite values: {observed_path}")
        safe_observed = np.nan_to_num(
            observed.astype(np.float64), copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )
        safe_expected = np.nan_to_num(
            expected.astype(np.float64), copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )
        absolute = np.abs(safe_observed - safe_expected)
        reference_passed = observed_finite and bool(
            np.all(absolute <= atol + rtol * np.abs(safe_expected))
        )
        if enforce_numerical_gates and not reference_passed:
            raise RuntimeError(f"worker numerical gate failed: {observed_path}")
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(absolute, initial=0.0)))
        if reference is None:
            reference = observed
        else:
            reference_finite = bool(np.all(np.isfinite(reference)))
            safe_reference = np.nan_to_num(
                reference.astype(np.float64), copy=False, nan=0.0, posinf=0.0, neginf=0.0
            )
            determinism_error = np.abs(safe_observed - safe_reference)
            maximum_determinism_error = max(
                maximum_determinism_error,
                float(np.max(determinism_error, initial=0.0)),
            )
            repetition_stable = (
                observed_finite
                and reference_finite
                and bool(
                    np.all(
                        determinism_error
                        <= determinism_atol + determinism_rtol * np.abs(safe_reference)
                    )
                )
            )
            tolerance_stable = tolerance_stable and repetition_stable
            if enforce_numerical_gates and not repetition_stable:
                raise RuntimeError(f"worker output is not tolerance-stable: {observed_path}")
        output_hashes.append(recorded_hash)
    tactic = _validate_tactic_diagnostic(
        result.get("tactic_diagnostic"),
        result_path=result_path,
        runs_root=runs_root,
        expected_engine_sha256=expected_engine_sha256,
        expected_count=expected_count,
        required=require_tactic_diagnostic,
    )
    return {
        "repetitions": len(repetitions),
        "output_sha256": output_hashes,
        "bitwise_stable": len(set(output_hashes)) == 1,
        "tolerance_stable": tolerance_stable,
        "input_integrity_stable": True if input_integrity_verified else None,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_determinism_error": maximum_determinism_error,
        "tactic_diagnostic": tactic,
    }


def _validate_tactic_diagnostic(
    value: object,
    *,
    result_path: Path,
    runs_root: Path,
    expected_engine_sha256: str,
    expected_count: int,
    required: bool,
) -> dict[str, object] | None:
    if value is None:
        if required:
            raise RuntimeError(f"worker omitted selected-tactic evidence: {result_path}")
        return None
    if not isinstance(value, dict):
        raise RuntimeError(f"worker selected-tactic evidence is invalid: {result_path}")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise RuntimeError(f"worker selected-tactic path is invalid: {result_path}")
    container_path = PurePosixPath(path_value)
    if not container_path.is_relative_to(PurePosixPath("/output")):
        raise RuntimeError(f"worker selected-tactic path escaped /output: {result_path}")
    relative = container_path.relative_to(PurePosixPath("/output"))
    diagnostic_path = (runs_root / Path(*relative.parts)).resolve(strict=True)
    if not diagnostic_path.is_relative_to(runs_root.resolve(strict=True)):
        raise RuntimeError(f"worker selected-tactic path escaped the run root: {result_path}")
    selected = value.get("selected_tactic")
    if (
        value.get("sha256") != sha256_file(diagnostic_path)
        or value.get("bytes") != diagnostic_path.stat().st_size
        or value.get("engine_sha256") != expected_engine_sha256
        or value.get("enqueue_count") != expected_count
        or selected not in {"kSCALAR_REFERENCE", "kVECTORIZED_WARP"}
        or not isinstance(value.get("rows"), int)
        or not isinstance(value.get("hidden"), int)
    ):
        raise RuntimeError(f"worker selected-tactic evidence differs: {result_path}")
    return {
        "path": path_value,
        "sha256": value["sha256"],
        "bytes": value["bytes"],
        "engine_sha256": value["engine_sha256"],
        "selected_tactic": selected,
        "rows": value["rows"],
        "hidden": value["hidden"],
        "enqueue_count": value["enqueue_count"],
    }


def _validate_input_integrity(
    value: object,
    expected_input_hashes: dict[str, str],
    *,
    result_path: Path,
) -> None:
    if not isinstance(value, list) or len(value) != len(expected_input_hashes):
        raise RuntimeError(f"worker input integrity inventory changed: {result_path}")
    observed: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError(f"worker input integrity record is invalid: {result_path}")
        name = item["name"]
        if name in observed or item.get("stable") is not True:
            raise RuntimeError(f"worker input integrity failed: {result_path}")
        source_sha256 = item.get("source_sha256")
        host_sha256 = item.get("host_value_sha256")
        device_sha256 = item.get("device_value_sha256")
        if (
            not isinstance(source_sha256, str)
            or not isinstance(host_sha256, str)
            or device_sha256 != host_sha256
        ):
            raise RuntimeError(f"worker input integrity hashes are invalid: {result_path}")
        observed[name] = source_sha256
    if observed != expected_input_hashes:
        raise RuntimeError(f"worker repetition input hashes changed: {result_path}")


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
