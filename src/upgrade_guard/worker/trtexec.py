"""Version-inventoried trtexec command and raw-timing adapter."""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError


@dataclass(frozen=True)
class TrtexecTiming:
    """Raw primary compute timings and summary."""

    milliseconds: tuple[float, ...]
    median_milliseconds: float
    mean_milliseconds: float
    coefficient_of_variation: float


@dataclass(frozen=True)
class FrozenTrtexecInput:
    """One named contiguous raw tensor and its source identity."""

    name: str
    source_sha256: str
    raw_sha256: str
    raw_bytes: int
    dtype: str
    shape: tuple[int, ...]
    container_path: str


@dataclass(frozen=True)
class TrtexecBenchmarkInvocation:
    """Exact benchmark command bound to the raw tensors it consumes."""

    command: tuple[str, ...]
    inputs: tuple[FrozenTrtexecInput, ...]

    def evidence(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "command_sha256": command_sha256(self.command),
            "input_sha256": {item.name: item.raw_sha256 for item in self.inputs},
            "inputs": [
                {
                    "name": item.name,
                    "source_sha256": item.source_sha256,
                    "raw_sha256": item.raw_sha256,
                    "raw_bytes": item.raw_bytes,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "path": item.container_path,
                }
                for item in self.inputs
            ],
        }


def freeze_raw_inputs(
    input_paths: dict[str, Path],
    output_directory: Path,
    container_directory: str,
) -> tuple[FrozenTrtexecInput, ...]:
    """Materialize exact C-order raw tensors for trtexec `--loadInputs`."""

    container_root = PurePosixPath(container_directory)
    if (
        not container_root.is_absolute()
        or not container_root.is_relative_to(PurePosixPath("/output"))
        or ".." in container_root.parts
        or any(character in container_directory for character in (",", "\x00"))
    ):
        raise InvalidInputError("trtexec raw-input container directory is unsafe")
    if not input_paths:
        raise InvalidInputError("trtexec requires at least one frozen input")
    output_directory.mkdir(parents=True, exist_ok=True)
    frozen = []
    for name, source in sorted(input_paths.items()):
        _validate_input_name(name)
        try:
            value = np.load(source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise InvalidInputError(f"trtexec source input is invalid: {name}") from error
        if value.dtype.hasobject or any(dimension <= 0 for dimension in value.shape):
            raise InvalidInputError(f"trtexec source input has an unsupported schema: {name}")
        if value.dtype.kind in "fc" and not bool(np.all(np.isfinite(value))):
            raise InvalidInputError(f"trtexec source input contains nonfinite values: {name}")
        contiguous = np.ascontiguousarray(value)
        destination = output_directory / f"{name}.raw"
        _write_atomic(destination, contiguous.tobytes(order="C"))
        frozen.append(
            FrozenTrtexecInput(
                name=name,
                source_sha256=sha256_file(source),
                raw_sha256=sha256_file(destination),
                raw_bytes=destination.stat().st_size,
                dtype=str(contiguous.dtype),
                shape=tuple(int(item) for item in contiguous.shape),
                container_path=str(container_root / destination.name),
            )
        )
    return tuple(frozen)


def benchmark_command(
    *,
    trtexec_path: str,
    supported_options: tuple[str, ...],
    engine: str,
    shapes: dict[str, tuple[int, ...]],
    export_times: str,
    warmup_milliseconds: int,
    measurement_milliseconds: int,
    inputs: tuple[FrozenTrtexecInput, ...],
) -> TrtexecBenchmarkInvocation:
    """Build an explicit one-stream, no-transfer, unprofiled benchmark command."""

    required = {
        "--loadEngine",
        "--shapes",
        "--exportTimes",
        "--warmUp",
        "--duration",
        "--noDataTransfers",
        "--loadInputs",
    }
    missing = sorted(required - set(supported_options))
    stream_option = (
        "--infStreams=1"
        if "--infStreams" in supported_options
        else "--streams=1"
        if "--streams" in supported_options
        else None
    )
    if missing:
        raise UnsupportedEnvironmentError(
            "locked trtexec lacks required primary benchmark options",
            details={"missing": missing},
        )
    if stream_option is None:
        raise UnsupportedEnvironmentError(
            "locked trtexec lacks a one-stream option",
            details={"missing": ["--infStreams or --streams"]},
        )
    input_map = {item.name: item for item in inputs}
    if not inputs or len(input_map) != len(inputs) or set(input_map) != set(shapes):
        raise InvalidInputError("trtexec inputs must exactly match the named shape inventory")
    for name, shape in shapes.items():
        _validate_input_name(name)
        if input_map[name].shape != shape:
            raise InvalidInputError(f"trtexec raw-input shape differs for {name}")
    shape_value = ",".join(
        f"{name}:{'x'.join(str(item) for item in shape)}" for name, shape in sorted(shapes.items())
    )
    duration_seconds = max(1, (measurement_milliseconds + 999) // 1000)
    input_value = ",".join(f"{name}:{input_map[name].container_path}" for name in sorted(input_map))
    command = (
        trtexec_path,
        f"--loadEngine={engine}",
        f"--shapes={shape_value}",
        f"--loadInputs={input_value}",
        f"--exportTimes={export_times}",
        f"--warmUp={warmup_milliseconds}",
        f"--duration={duration_seconds}",
        stream_option,
        "--noDataTransfers",
    )
    return TrtexecBenchmarkInvocation(
        command=command,
        inputs=tuple(sorted(inputs, key=lambda item: item.name)),
    )


def _validate_input_name(name: str) -> None:
    if not re.fullmatch(r"[^,:=\s\x00]+", name):
        raise InvalidInputError("trtexec input name is unsafe")


def _write_atomic(path: Path, value: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_exported_times(path: Path) -> TrtexecTiming:
    """Parse version-tolerant trtexec JSON without accepting missing raw samples."""

    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidInputError("trtexec raw timing export is invalid") from error
    records = value.get("times") if isinstance(value, dict) else value
    if not isinstance(records, list) or not records:
        raise InvalidInputError("trtexec raw timing export contains no samples")
    samples: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            raise InvalidInputError("trtexec timing record must be an object")
        observed = next(
            (
                record[key]
                for key in ("computeMs", "latencyMs", "gpuComputeTimeMs")
                if key in record
            ),
            None,
        )
        if not isinstance(observed, int | float) or observed <= 0:
            raise InvalidInputError("trtexec timing record lacks a positive compute duration")
        samples.append(float(observed))
    mean = statistics.fmean(samples)
    variation = statistics.pstdev(samples) / mean if len(samples) > 1 else 0.0
    return TrtexecTiming(
        milliseconds=tuple(samples),
        median_milliseconds=statistics.median(samples),
        mean_milliseconds=mean,
        coefficient_of_variation=variation,
    )
