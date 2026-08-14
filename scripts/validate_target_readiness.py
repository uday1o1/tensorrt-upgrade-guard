"""Validate both-worker target readiness against frozen CPU references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.corpus.reference import run_onnx_reference
from upgrade_guard.worker.common import write_json_atomic
from upgrade_guard.worker.evidence import validate_repetitions

ENVIRONMENTS = ("baseline", "candidate")
STANDARD_CASES = ("b1_s8", "b1_s128")
PLUGIN_CASE = "tail-random-h259"
MOBILENET_CASE = "minimum"


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"artifact is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact is not a JSON object: {path}")
    return value


def _direct_directories(path: Path) -> set[str]:
    try:
        return {child.name for child in path.iterdir() if child.is_dir()}
    except OSError as error:
        raise RuntimeError(f"readiness directory is unavailable: {path}") from error


def _require_inventory(path: Path, expected: set[str]) -> None:
    observed = _direct_directories(path)
    if observed != expected:
        raise RuntimeError(
            f"readiness inventory changed at {path}: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )


def _artifact_hash(value: object, expected_path: Path, label: str) -> str:
    if not isinstance(value, dict):
        raise RuntimeError(f"build manifest {label} evidence is not an object")
    recorded_path = value.get("path")
    recorded_hash = value.get("sha256")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise RuntimeError(f"build manifest {label} path is not populated")
    authored = PurePosixPath(recorded_path)
    trusted_root = PurePosixPath("/corpus" if label == "model" else "/output")
    if (
        not authored.is_absolute()
        or not authored.is_relative_to(trusted_root)
        or ".." in authored.parts
        or authored.name != expected_path.name
    ):
        raise RuntimeError(f"build manifest {label} path is unsafe or mismatched")
    observed_hash = sha256_file(expected_path)
    if recorded_hash != observed_hash:
        raise RuntimeError(f"build manifest {label} hash changed: {expected_path}")
    return observed_hash


def _validate_build(root: Path, model: Path) -> dict[str, object]:
    manifest_path = root / "build.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != "upgradeguard.dev/worker-build/v1":
        raise RuntimeError(f"worker build schema changed: {manifest_path}")
    if manifest.get("status") != "passed" or manifest.get("strongly_typed") is not True:
        raise RuntimeError(f"worker build did not pass strong typing: {manifest_path}")
    if manifest.get("parser_errors") != []:
        raise RuntimeError(f"worker build retained parser errors: {manifest_path}")
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) and argument for argument in command)
        or manifest.get("command_sha256") != command_sha256(command)
    ):
        raise RuntimeError(f"worker build command evidence is invalid: {manifest_path}")
    version = manifest.get("tensorrt_version")
    duration = manifest.get("duration_seconds")
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"worker build TensorRT version is not populated: {manifest_path}")
    if (
        not isinstance(duration, int | float)
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise RuntimeError(f"worker build duration is invalid: {manifest_path}")
    model_hash = _artifact_hash(manifest.get("model"), model, "model")
    engine_path = root / "engine.plan"
    engine_hash = _artifact_hash(manifest.get("engine"), engine_path, "engine")
    engine = manifest["engine"]
    if not isinstance(engine, dict):
        raise AssertionError("validated engine object changed type")
    engine_bytes = engine.get("bytes")
    device_memory = engine.get("device_memory_bytes")
    if (
        not isinstance(engine_bytes, int)
        or isinstance(engine_bytes, bool)
        or engine_bytes <= 0
        or engine_bytes != engine_path.stat().st_size
        or not isinstance(device_memory, int)
        or isinstance(device_memory, bool)
        or device_memory < 0
    ):
        raise RuntimeError(f"worker build engine evidence is incomplete: {manifest_path}")
    inspector_hash = _artifact_hash(manifest.get("inspector"), root / "inspector.json", "inspector")
    timing_cache = manifest.get("timing_cache")
    if not isinstance(timing_cache, dict):
        raise RuntimeError(f"worker build timing-cache evidence is invalid: {manifest_path}")
    timing_cache_hash = timing_cache.get("output_sha256")
    timing_cache_path = root / "timing.cache"
    if timing_cache_hash != sha256_file(timing_cache_path):
        raise RuntimeError(f"worker build timing-cache hash changed: {manifest_path}")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "model_sha256": model_hash,
        "engine_sha256": engine_hash,
        "engine_bytes": engine_bytes,
        "engine_device_memory_bytes": device_memory,
        "inspector_sha256": inspector_hash,
        "timing_cache_sha256": timing_cache_hash,
        "tensorrt_version": version,
        "strongly_typed": True,
    }


def _load_finite(path: Path) -> np.ndarray[Any, Any]:
    value = np.load(path, allow_pickle=False)
    if value.dtype.kind != "f" or not bool(np.all(np.isfinite(value))):
        raise RuntimeError(f"reference tensor is not finite floating-point data: {path}")
    return cast(np.ndarray[Any, Any], value)


def _validate_case(
    *,
    result_path: Path,
    runs_root: Path,
    output_name: str,
    expected: np.ndarray[Any, Any],
    engine_path: Path,
    inputs: dict[str, Path],
    repetitions: int,
) -> dict[str, object]:
    input_hashes = {name: sha256_file(path) for name, path in sorted(inputs.items())}
    engine_hash = sha256_file(engine_path)
    validation = validate_repetitions(
        result_path=result_path,
        runs_root=runs_root,
        expected_output_name=output_name,
        expected=expected,
        atol=1e-5,
        rtol=1e-4,
        expected_engine_sha256=engine_hash,
        expected_input_hashes=input_hashes,
        expected_count=repetitions,
    )
    if validation.get("tolerance_stable") is not True:
        raise RuntimeError(f"readiness determinism gate failed: {result_path}")
    return {
        "engine_sha256": engine_hash,
        "input_sha256": input_hashes,
        **validation,
    }


def _mobilenet_output_name(corpus: Path) -> str:
    lock_path = corpus / "mobilenet-corpus.lock.json"
    lock = _json_object(lock_path)
    if lock.get("schema_version") != "upgradeguard.dev/mobilenet-corpus/v1":
        raise RuntimeError(f"MobileNet corpus schema changed: {lock_path}")
    cases = lock.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError(f"MobileNet corpus case inventory is invalid: {lock_path}")
    minimum = [
        case for case in cases if isinstance(case, dict) and case.get("id") == MOBILENET_CASE
    ]
    if len(minimum) != 1 or not isinstance(minimum[0].get("output_name"), str):
        raise RuntimeError(f"MobileNet minimum case is not uniquely populated: {lock_path}")
    return str(minimum[0]["output_name"])


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("repetitions must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-corpus", type=Path, required=True)
    parser.add_argument("--plugin-corpus", type=Path, required=True)
    parser.add_argument("--mobilenet-corpus", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=_positive, default=2)
    arguments = parser.parse_args()

    _require_inventory(arguments.runs, set(ENVIRONMENTS))
    standard_model = arguments.core_corpus / "models" / "tiny-transformer-fp32.onnx"
    plugin_model = arguments.plugin_corpus / "residual-rmsnorm-fp32.onnx"
    mobilenet_model = arguments.mobilenet_corpus / "mobilenetv3-small-075-dynamic.onnx"
    plugin_case = arguments.plugin_corpus / "fp32" / PLUGIN_CASE
    mobilenet_case = arguments.mobilenet_corpus / "inputs" / MOBILENET_CASE
    plugin_expected = _load_finite(plugin_case / "expected.npy")
    mobilenet_expected = _load_finite(mobilenet_case / "expected.npy")
    mobilenet_output = _mobilenet_output_name(arguments.mobilenet_corpus)

    environments = []
    for environment in ENVIRONMENTS:
        environment_root = arguments.runs / environment
        _require_inventory(environment_root, {"standard", "plugin", "mobilenet"})
        _require_inventory(environment_root / "standard", set(STANDARD_CASES))
        _require_inventory(environment_root / "plugin", {"fp32"})
        _require_inventory(environment_root / "plugin" / "fp32", {PLUGIN_CASE})
        _require_inventory(environment_root / "mobilenet", {MOBILENET_CASE})

        standard_root = environment_root / "standard"
        plugin_root = environment_root / "plugin" / "fp32"
        mobilenet_root = environment_root / "mobilenet"
        builds = {
            "standard": _validate_build(standard_root, standard_model),
            "plugin_fp32": _validate_build(plugin_root, plugin_model),
            "mobilenet": _validate_build(mobilenet_root, mobilenet_model),
        }
        workloads = []
        for case_name in STANDARD_CASES:
            case_root = arguments.core_corpus / "inputs" / "tiny-transformer-fp32" / case_name
            inputs = {name: case_root / f"{name}.npy" for name in ("tokens", "mask")}
            loaded_inputs = {name: _load_finite(path) for name, path in inputs.items()}
            reference = run_onnx_reference(standard_model, loaded_inputs)
            if len(reference) != 1:
                raise RuntimeError("standard readiness model output inventory changed")
            workloads.append(
                {
                    "id": f"standard/{case_name}",
                    "reference_sha256": reference[0].sha256,
                    **_validate_case(
                        result_path=standard_root / case_name / "correctness.json",
                        runs_root=arguments.runs,
                        output_name=reference[0].name,
                        expected=reference[0].values,
                        engine_path=standard_root / "engine.plan",
                        inputs=inputs,
                        repetitions=arguments.repetitions,
                    ),
                }
            )
        workloads.extend(
            [
                {
                    "id": f"plugin/fp32/{PLUGIN_CASE}",
                    "reference_sha256": sha256_file(plugin_case / "expected.npy"),
                    **_validate_case(
                        result_path=plugin_root / PLUGIN_CASE / "correctness.json",
                        runs_root=arguments.runs,
                        output_name="output",
                        expected=plugin_expected,
                        engine_path=plugin_root / "engine.plan",
                        inputs={
                            name: plugin_case / f"{name}.npy" for name in ("x", "residual", "gamma")
                        },
                        repetitions=arguments.repetitions,
                    ),
                },
                {
                    "id": f"mobilenet/{MOBILENET_CASE}",
                    "reference_sha256": sha256_file(mobilenet_case / "expected.npy"),
                    **_validate_case(
                        result_path=mobilenet_root / MOBILENET_CASE / "correctness.json",
                        runs_root=arguments.runs,
                        output_name=mobilenet_output,
                        expected=mobilenet_expected,
                        engine_path=mobilenet_root / "engine.plan",
                        inputs={"x": mobilenet_case / "x.npy"},
                        repetitions=arguments.repetitions,
                    ),
                },
            ]
        )
        environments.append(
            {
                "environment": environment,
                "builds": builds,
                "workloads": workloads,
            }
        )

    write_json_atomic(
        arguments.output,
        {
            "schema_version": "upgradeguard.dev/target-readiness/v1",
            "status": "passed",
            "repetitions": arguments.repetitions,
            "environment_inventory": list(ENVIRONMENTS),
            "workload_inventory": [
                "standard/b1_s8",
                "standard/b1_s128",
                f"plugin/fp32/{PLUGIN_CASE}",
                f"mobilenet/{MOBILENET_CASE}",
            ],
            "environments": environments,
        },
    )


if __name__ == "__main__":
    main()
