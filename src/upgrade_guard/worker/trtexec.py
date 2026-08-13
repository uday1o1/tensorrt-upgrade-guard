"""Version-inventoried trtexec command and raw-timing adapter."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError


@dataclass(frozen=True)
class TrtexecTiming:
    """Raw primary compute timings and summary."""

    milliseconds: tuple[float, ...]
    median_milliseconds: float
    mean_milliseconds: float
    coefficient_of_variation: float


def benchmark_command(
    *,
    trtexec_path: str,
    supported_options: tuple[str, ...],
    engine: str,
    shapes: dict[str, tuple[int, ...]],
    export_times: str,
    warmup_milliseconds: int,
    measurement_milliseconds: int,
) -> tuple[str, ...]:
    """Build an explicit one-stream, no-transfer, unprofiled benchmark command."""

    required = {
        "--loadEngine",
        "--shapes",
        "--exportTimes",
        "--warmUp",
        "--duration",
        "--noDataTransfers",
    }
    missing = sorted(required - set(supported_options))
    if missing:
        raise UnsupportedEnvironmentError(
            "locked trtexec lacks required primary benchmark options",
            details={"missing": missing},
        )
    shape_value = ",".join(
        f"{name}:{'x'.join(str(item) for item in shape)}" for name, shape in sorted(shapes.items())
    )
    duration_seconds = max(1, (measurement_milliseconds + 999) // 1000)
    return (
        trtexec_path,
        f"--loadEngine={engine}",
        f"--shapes={shape_value}",
        f"--exportTimes={export_times}",
        f"--warmUp={warmup_milliseconds}",
        f"--duration={duration_seconds}",
        "--streams=1",
        "--noDataTransfers",
    )


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
