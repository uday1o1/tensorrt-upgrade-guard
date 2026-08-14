"""Tests for hash-addressed remote qualification resume state."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "qualification_state.py"
SPEC = importlib.util.spec_from_file_location("qualification_state", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification_state)


def test_step_payload_binds_source_gpu_mode_and_artifact_hash(tmp_path: Path) -> None:
    state = tmp_path / "state"
    project = tmp_path / "project"
    state.mkdir()
    project.mkdir()
    (state / "source.commit").write_text("abc\n", encoding="utf-8")
    (state / "gpu.uuid").write_text("GPU-1\n", encoding="utf-8")
    (state / "gpu-preflight.csv").write_text("GPU, GPU-1\n", encoding="utf-8")
    payload = qualification_state._payload(
        state,
        project,
        "preflight",
        "a" * 40,
        "GPU-1",
        "full",
    )
    assert payload["source_git_commit"] == "a" * 40
    assert payload["gpu_uuid"] == "GPU-1"
    assert payload["mode"] == "full"
    original = payload["artifacts"]["source.commit"]["sha256"]
    (state / "source.commit").write_text("changed\n", encoding="utf-8")
    changed = qualification_state._payload(
        state,
        project,
        "preflight",
        "a" * 40,
        "GPU-1",
        "full",
    )
    assert changed["artifacts"]["source.commit"]["sha256"] != original


def test_step_payload_requires_passing_semantic_result(tmp_path: Path) -> None:
    state = tmp_path / "state"
    project = tmp_path / "project"
    state.mkdir()
    project.mkdir()
    result = state / "core-run" / "qualification-summary.json"
    result.parent.mkdir()
    result.write_text('{"status":"failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        qualification_state._payload(
            state,
            project,
            "core-qualification",
            "a" * 40,
            "GPU-1",
            "full",
        )


def test_corpus_verifier_detects_tampering_and_inventory_drift(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    artifact = root / "model.onnx"
    artifact.write_bytes(b"model")
    digest = qualification_state._sha256(artifact)
    lock = {
        "artifacts": [
            {
                "path": "model.onnx",
                "bytes": artifact.stat().st_size,
                "sha256": digest,
            }
        ]
    }
    (root / "corpus.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    qualification_state._verify_corpus(root.resolve())
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its lock"):
        qualification_state._verify_corpus(root.resolve())
    artifact.write_bytes(b"model")
    (root / "undeclared.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        qualification_state._verify_corpus(root.resolve())


def test_worker_lock_consistency_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digests = ["sha256:" + character * 64 for character in ("1", "2")]
    workers = tmp_path / "workers.json"
    matrix = tmp_path / "matrix.json"
    workers.write_text(
        json.dumps({"images": [{"manifest_digest": digest} for digest in digests]}),
        encoding="utf-8",
    )
    matrix.write_text(
        json.dumps(
            {
                "environments": [
                    {"worker_image": {"manifest_digest": digest}} for digest in reversed(digests)
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "verify-worker-lock",
            "--workers",
            str(workers),
            "--matrix",
            str(matrix),
        ],
    )
    assert qualification_state.main() == 0
