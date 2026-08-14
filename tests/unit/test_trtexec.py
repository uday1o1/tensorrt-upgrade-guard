"""trtexec inventory and raw timing adapter tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.worker.trtexec import (
    benchmark_command,
    freeze_raw_inputs,
    load_exported_times,
)

OPTIONS = (
    "--loadEngine",
    "--shapes",
    "--exportTimes",
    "--warmUp",
    "--duration",
    "--noDataTransfers",
    "--loadInputs",
    "--infStreams",
)


def _inputs(tmp_path: Path) -> tuple:
    source = tmp_path / "source"
    source.mkdir()
    tokens = np.arange(8, dtype=np.float32).reshape(1, 8)
    mask = np.zeros((1, 1, 1, 8), dtype=np.float32)
    np.save(source / "tokens.npy", tokens, allow_pickle=False)
    np.save(source / "mask.npy", mask, allow_pickle=False)
    return freeze_raw_inputs(
        {"tokens": source / "tokens.npy", "mask": source / "mask.npy"},
        tmp_path / "raw",
        "/output/raw",
    )


def test_benchmark_command_declares_every_primary_policy(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    invocation = benchmark_command(
        trtexec_path="/opt/tensorrt/bin/trtexec",
        supported_options=OPTIONS,
        engine="/output/engine.plan",
        shapes={"tokens": (1, 8), "mask": (1, 1, 1, 8)},
        export_times="/output/times.json",
        warmup_milliseconds=200,
        measurement_milliseconds=1000,
        inputs=inputs,
    )
    command = invocation.command
    assert "--infStreams=1" in command
    assert "--noDataTransfers" in command
    assert "--loadInputs=mask:/output/raw/mask.raw,tokens:/output/raw/tokens.raw" in command
    assert invocation.evidence()["input_sha256"] == {item.name: item.raw_sha256 for item in inputs}
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
            inputs=inputs,
        )
    single = inputs[0]
    legacy = benchmark_command(
        trtexec_path="trtexec",
        supported_options=tuple(
            "--streams" if option == "--infStreams" else option for option in OPTIONS
        ),
        engine="engine.plan",
        shapes={single.name: single.shape},
        export_times="times.json",
        warmup_milliseconds=1,
        measurement_milliseconds=1,
        inputs=(inputs[0],),
    )
    assert "--streams=1" in legacy.command
    with pytest.raises(UnsupportedEnvironmentError, match="one-stream"):
        benchmark_command(
            trtexec_path="trtexec",
            supported_options=tuple(option for option in OPTIONS if option != "--infStreams"),
            engine="engine.plan",
            shapes={single.name: single.shape},
            export_times="times.json",
            warmup_milliseconds=1,
            measurement_milliseconds=1,
            inputs=(inputs[0],),
        )


def test_freeze_raw_inputs_is_exact_and_fail_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    tokens = next(item for item in inputs if item.name == "tokens")
    assert tokens.raw_bytes == 8 * np.dtype(np.float32).itemsize
    assert (tmp_path / "raw" / "tokens.raw").read_bytes() == np.arange(
        8, dtype=np.float32
    ).tobytes()
    assert tokens.source_sha256 != tokens.raw_sha256
    source = tmp_path / "nonfinite.npy"
    np.save(source, np.asarray([np.nan], dtype=np.float32), allow_pickle=False)
    with pytest.raises(InvalidInputError, match="nonfinite"):
        freeze_raw_inputs({"x": source}, tmp_path / "other", "/output/raw")
    with pytest.raises(InvalidInputError, match="unsafe"):
        freeze_raw_inputs({"x": source}, tmp_path / "other", "/corpus/raw")


def test_raw_timing_parser_requires_positive_iteration_records(tmp_path: Path) -> None:
    path = tmp_path / "times.json"
    path.write_text(json.dumps([{"computeMs": 1.0}, {"computeMs": 3.0}]))
    timing = load_exported_times(path)
    assert timing.milliseconds == (1.0, 3.0)
    assert timing.median_milliseconds == 2.0
    path.write_text("[]")
    with pytest.raises(InvalidInputError, match="no samples"):
        load_exported_times(path)
