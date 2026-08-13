"""trtexec inventory and raw timing adapter tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.worker.trtexec import benchmark_command, load_exported_times

OPTIONS = (
    "--loadEngine",
    "--shapes",
    "--exportTimes",
    "--warmUp",
    "--duration",
    "--noDataTransfers",
    "--infStreams",
)


def test_benchmark_command_declares_every_primary_policy() -> None:
    command = benchmark_command(
        trtexec_path="/opt/tensorrt/bin/trtexec",
        supported_options=OPTIONS,
        engine="/output/engine.plan",
        shapes={"tokens": (1, 8, 256), "mask": (1, 1, 1, 8)},
        export_times="/output/times.json",
        warmup_milliseconds=200,
        measurement_milliseconds=1000,
    )
    assert "--infStreams=1" in command
    assert "--noDataTransfers" in command
    assert not any("useCudaGraph" in item for item in command)
    with pytest.raises(UnsupportedEnvironmentError, match="lacks required"):
        benchmark_command(
            trtexec_path="trtexec",
            supported_options=(),
            engine="engine.plan",
            shapes={"x": (1,)},
            export_times="times.json",
            warmup_milliseconds=1,
            measurement_milliseconds=1,
        )
    legacy = benchmark_command(
        trtexec_path="trtexec",
        supported_options=tuple(
            "--streams" if option == "--infStreams" else option for option in OPTIONS
        ),
        engine="engine.plan",
        shapes={"x": (1,)},
        export_times="times.json",
        warmup_milliseconds=1,
        measurement_milliseconds=1,
    )
    assert "--streams=1" in legacy
    with pytest.raises(UnsupportedEnvironmentError, match="one-stream"):
        benchmark_command(
            trtexec_path="trtexec",
            supported_options=tuple(option for option in OPTIONS if option != "--infStreams"),
            engine="engine.plan",
            shapes={"x": (1,)},
            export_times="times.json",
            warmup_milliseconds=1,
            measurement_milliseconds=1,
        )


def test_raw_timing_parser_requires_positive_iteration_records(tmp_path: Path) -> None:
    path = tmp_path / "times.json"
    path.write_text(json.dumps([{"computeMs": 1.0}, {"computeMs": 3.0}]))
    timing = load_exported_times(path)
    assert timing.milliseconds == (1.0, 3.0)
    assert timing.median_milliseconds == 2.0
    path.write_text("[]")
    with pytest.raises(InvalidInputError, match="no samples"):
        load_exported_times(path)
