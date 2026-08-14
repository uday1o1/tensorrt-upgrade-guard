"""Dependency-light GPU-worker filesystem helper tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from upgrade_guard.worker.common import (
    load_json,
    process_memory_evidence,
    sha256_file,
    write_json_atomic,
)


def test_worker_common_hashes_and_atomically_publishes_json(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"upgrade-guard")
    assert sha256_file(payload) == (
        "sha256:6b1b1935f48cadbe6734807d88b7f433f05d638ce5302e36ddc48e35957ac6bd"
    )
    output = tmp_path / "nested" / "result.json"
    write_json_atomic(output, {"status": "passed", "values": [1, 2]})
    assert load_json(output) == {"status": "passed", "values": [1, 2]}
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert not list(output.parent.glob(".result.json.*"))


def test_worker_atomic_writer_cleans_up_on_serialization_failure(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    with pytest.raises(TypeError):
        write_json_atomic(output, {"not-json": object()})
    assert not output.exists()
    assert not list(tmp_path.glob(".result.json.*"))


def test_process_memory_evidence_keeps_host_and_gpu_sources_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = type("Result", (), {"returncode": 0, "stdout": "GPU-1, 7, python3, 12\n"})()
    monkeypatch.setattr(
        "upgrade_guard.worker.common.subprocess.run", lambda *args, **kwargs: result
    )
    evidence = process_memory_evidence()
    assert evidence["host_peak_rss_bytes"] > 0
    assert evidence["gpu_process_rows"] == ["GPU-1, 7, python3, 12"]
    assert "coarse" in evidence["gpu_process_observation"]
