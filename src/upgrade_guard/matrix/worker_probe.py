"""Self-contained worker probe executed only inside a trusted GPU container."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "upgradeguard.dev/worker-probe/v1"
MAX_TOOL_OUTPUT = 16_384


def _run(args: Sequence[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(  # noqa: S603
            tuple(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""
    output = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    return result.returncode, output[:MAX_TOOL_OUTPUT]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tool(command: str, version_args: Sequence[str] = ("--version",)) -> dict[str, Any]:
    path = shutil.which(command)
    if path is None:
        return {"available": False}
    returncode, version = _run((path, *version_args))
    if returncode != 0 or not version:
        return {"available": False}
    return {
        "available": True,
        "path": str(Path(path).resolve()),
        "version": version.splitlines()[0],
        "sha256": _sha256_file(str(Path(path).resolve())),
    }


def _trtexec() -> dict[str, Any]:
    observation = _tool("trtexec")
    if not observation["available"]:
        return observation
    path = str(observation["path"])
    returncode, help_text = _run((path, "--help"))
    if returncode != 0 or not help_text:
        return {"available": False}
    options = sorted(set(re.findall(r"--([A-Za-z0-9][A-Za-z0-9-]*)", help_text)))
    if not options:
        return {"available": False}
    observation["help_sha256"] = f"sha256:{hashlib.sha256(help_text.encode()).hexdigest()}"
    observation["options"] = options
    return observation


def _package_version(distribution: str, module: str | None = None) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if module is None:
            return ""
        try:
            imported = importlib.import_module(module)
        except (ImportError, OSError):
            return ""
        value = getattr(imported, "__version__", "")
        return str(value) if value else ""


def _cuda_runtime_version() -> str:
    candidates = (
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so",
    )
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        version = ctypes.c_int()
        if library.cudaRuntimeGetVersion(ctypes.byref(version)) == 0:
            return f"{version.value // 1000}.{(version.value % 1000) // 10}"
    return ""


def _cuda_toolkit_version(cuda_compiler: dict[str, Any]) -> str:
    if not cuda_compiler["available"]:
        return ""
    version = str(cuda_compiler["version"])
    match = re.search(r"(?:release|V)\s*([0-9]+\.[0-9]+)", version)
    return match.group(1) if match else version


def _gpu() -> tuple[dict[str, Any], str]:
    fields = "name,uuid,compute_cap,memory.total,vbios_version,driver_version,power.limit"
    returncode, output = _run(
        ("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits")
    )
    if returncode != 0:
        raise RuntimeError("nvidia-smi could not inspect the selected worker GPU")
    rows = [row for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"worker must see exactly one GPU, observed {len(rows)}")
    values = [value.strip() for value in rows[0].split(",", maxsplit=6)]
    if len(values) != 7:
        raise RuntimeError("nvidia-smi returned an invalid GPU row")
    name, uuid, compute_capability, vram, vbios, driver, power = values
    power_value = None if power in {"N/A", "[N/A]"} else float(power)
    return (
        {
            "name": name,
            "uuid": uuid,
            "compute_capability": compute_capability,
            "vram_mib": int(vram),
            "vbios_version": None if vbios in {"N/A", "[N/A]"} else vbios,
            "power_limit_watts": power_value,
        },
        driver,
    )


def _headers(patterns: Sequence[str]) -> list[str]:
    paths: set[str] = set()
    for pattern in patterns:
        for path in Path("/").glob(pattern.lstrip("/")):
            if path.is_file():
                paths.add(str(path))
    return sorted(paths)


def _operating_system() -> str:
    os_release = Path("/etc/os-release")
    if not os_release.is_file():
        return platform.system()
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"')
    return platform.system()


def collect_probe() -> dict[str, Any]:
    """Collect observed worker versions without trusting its authored tag."""

    manifest_digest = os.environ.get("UG_WORKER_MANIFEST_DIGEST", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_digest):
        raise RuntimeError("UG_WORKER_MANIFEST_DIGEST must identify the selected manifest")
    gpu, driver = _gpu()
    cuda_compiler = _tool("nvcc", ("--version",))
    return {
        "schema_version": SCHEMA_VERSION,
        "image_manifest_digest": manifest_digest,
        "gpu": gpu,
        "observed_driver": driver,
        "cuda_runtime": _cuda_runtime_version(),
        "cuda_toolkit": _cuda_toolkit_version(cuda_compiler),
        "tensorrt": _package_version("tensorrt", "tensorrt"),
        "python": platform.python_version(),
        "polygraphy": _package_version("polygraphy"),
        "onnx": _package_version("onnx", "onnx"),
        "onnxruntime": _package_version("onnxruntime", "onnxruntime"),
        "operating_system": _operating_system(),
        "kernel": platform.release(),
        "trtexec": _trtexec(),
        "compute_sanitizer": _tool("compute-sanitizer", ("--version",)),
        "nsight_systems": _tool("nsys", ("--version",)),
        "nsight_compute": _tool("ncu", ("--version",)),
        "c_compiler": _tool("gcc", ("--version",)),
        "cxx_compiler": _tool("g++", ("--version",)),
        "cuda_compiler": cuda_compiler,
        "cmake": _tool("cmake", ("--version",)),
        "ninja": _tool("ninja", ("--version",)),
        "cuda_headers": _headers(
            (
                "/usr/local/cuda/include/cuda_runtime_api.h",
                "/usr/include/cuda_runtime_api.h",
            )
        ),
        "tensorrt_headers": _headers(
            (
                "/usr/include/**/NvInfer.h",
                "/usr/local/include/**/NvInfer.h",
                "/opt/tensorrt/include/NvInfer.h",
            )
        ),
    }


def main() -> int:
    try:
        result = collect_probe()
    except Exception as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
