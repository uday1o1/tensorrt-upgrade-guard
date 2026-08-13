"""Versioned fail-closed compatibility policy for exact worker probes."""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib.resources import files
from typing import Any

from upgrade_guard.contracts.environment import CompatibilityEvidence, WorkerProbe
from upgrade_guard.contracts.matrix import CapabilityPolicy
from upgrade_guard.errors import InvalidInputError


def evaluate_compatibility(
    probe: WorkerProbe,
    capabilities: CapabilityPolicy,
    *,
    expected_gpu_uuid: str,
) -> CompatibilityEvidence:
    """Evaluate current documented prerequisites and full-V1 tool availability."""

    rules = _load_rules()
    reasons: list[str] = []
    if probe.gpu.uuid != expected_gpu_uuid:
        reasons.append(
            f"worker observed GPU {probe.gpu.uuid}, expected selected GPU {expected_gpu_uuid}"
        )

    cuda_major = _leading_integer(probe.cuda_runtime)
    minimum_drivers = rules["minimum_driver_by_cuda_major"]
    minimum_driver = (
        minimum_drivers.get(str(cuda_major)) if isinstance(minimum_drivers, dict) else None
    )
    if not isinstance(minimum_driver, str):
        runtime = probe.cuda_runtime or "<missing>"
        reasons.append(f"CUDA runtime {runtime} has no locked driver policy")
        minimum_driver = "unsupported"
    elif _version_tuple(probe.observed_driver) < _version_tuple(minimum_driver):
        reasons.append(
            f"driver {probe.observed_driver} is older than required {minimum_driver} "
            f"for CUDA {probe.cuda_runtime}"
        )

    minimum_compute = _required_string(rules, "minimum_compute_capability")
    if _version_tuple(probe.gpu.compute_capability) < _version_tuple(minimum_compute):
        reasons.append(
            f"GPU compute capability {probe.gpu.compute_capability} is below {minimum_compute}"
        )
    minimum_vram = rules.get("minimum_vram_mib")
    if not isinstance(minimum_vram, int):
        raise InvalidInputError("compatibility policy minimum_vram_mib must be an integer")
    if probe.gpu.vram_mib < minimum_vram:
        reasons.append(f"GPU VRAM {probe.gpu.vram_mib} MiB is below required {minimum_vram} MiB")

    for field, label in (
        ("tensorrt", "TensorRT"),
        ("cuda_runtime", "CUDA runtime"),
        ("cuda_toolkit", "CUDA toolkit"),
        ("python", "Python"),
        ("polygraphy", "Polygraphy"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
    ):
        if not getattr(probe, field):
            reasons.append(f"{label} version was not observed")

    required_tools = {
        "trtexec": (capabilities.trtexec, probe.trtexec.available),
        "C++ compiler": (capabilities.cxx_compiler, probe.cxx_compiler.available),
        "CMake": (capabilities.cmake, probe.cmake.available),
        "Ninja": (capabilities.ninja, probe.ninja.available),
        "CUDA compiler": (capabilities.cuda_compiler, probe.cuda_compiler.available),
        "Compute Sanitizer": (
            capabilities.compute_sanitizer,
            probe.compute_sanitizer.available,
        ),
        "Nsight Systems": (capabilities.nsight_systems, probe.nsight_systems.available),
        "Nsight Compute": (capabilities.nsight_compute, probe.nsight_compute.available),
    }
    for label, (required, available) in required_tools.items():
        if required and not available:
            reasons.append(f"required tool is unavailable: {label}")

    if capabilities.cuda_headers and not probe.cuda_headers:
        reasons.append("required CUDA headers are unavailable")
    if capabilities.tensorrt_headers and not probe.tensorrt_headers:
        reasons.append("required TensorRT headers are unavailable")
    if capabilities.cmake and probe.cmake.available:
        minimum_cmake = _required_string(rules, "minimum_cmake")
        if _version_tuple(probe.cmake.version or "") < _version_tuple(minimum_cmake):
            reasons.append(f"CMake {probe.cmake.version} is older than required {minimum_cmake}")

    sources = rules.get("source_urls")
    if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
        raise InvalidInputError("compatibility policy source_urls must be a string list")
    checked_at = rules.get("checked_at")
    if not isinstance(checked_at, str):
        raise InvalidInputError("compatibility policy checked_at must be a timestamp")
    return CompatibilityEvidence(
        policy_version=_required_string(rules, "policy_version"),
        source_urls=tuple(sources),
        checked_at=datetime.fromisoformat(checked_at.replace("Z", "+00:00")),
        minimum_driver=minimum_driver,
        minimum_compute_capability=minimum_compute,
        compatible=not reasons,
        reasons=tuple(reasons),
    )


def _load_rules() -> dict[str, Any]:
    resource = files("upgrade_guard.matrix").joinpath("compatibility-rules.json")
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidInputError("compatibility policy could not be loaded") from error
    if not isinstance(value, dict):
        raise InvalidInputError("compatibility policy must be a JSON object")
    return value


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise InvalidInputError(f"compatibility policy {key} must be a nonempty string")
    return item


def _leading_integer(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = tuple(int(number) for number in re.findall(r"\d+", value))
    return numbers or (0,)
