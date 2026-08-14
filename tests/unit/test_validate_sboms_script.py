"""Tests for fail-closed SPDX validation before measured GPU gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_sboms import validate_documents, validate_spdx
from upgrade_guard.contracts.base import sha256_file


def _spdx(path: Path, identity: str) -> None:
    path.write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "documentComment": f"Observed package inventory for {identity}",
                "packages": [{"name": "package"}],
            }
        ),
        encoding="utf-8",
    )


def test_documents_bind_host_lock_and_both_exact_workers(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    baseline_image = "registry/worker@sha256:" + "1" * 64
    candidate_image = "registry/worker@sha256:" + "2" * 64
    host = tmp_path / "host.spdx.json"
    baseline = tmp_path / "baseline.spdx.json"
    candidate = tmp_path / "candidate.spdx.json"
    _spdx(host, sha256_file(lock))
    _spdx(baseline, baseline_image)
    _spdx(candidate, candidate_image)

    value = validate_documents(
        host=host,
        baseline=baseline,
        candidate=candidate,
        baseline_image=baseline_image,
        candidate_image=candidate_image,
        lock=lock,
    )

    assert value["status"] == "passed"
    assert set(value["documents"]) == {"host", "baseline_worker", "candidate_worker"}


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"spdxVersion": "SPDX-2.2", "documentComment": "image", "packages": [{}]},
        {"spdxVersion": "SPDX-2.3", "documentComment": "image", "packages": []},
    ],
)
def test_malformed_or_empty_spdx_fails_closed(tmp_path: Path, value: dict[str, object]) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="populated SPDX"):
        validate_spdx(path, expected_image="image")


def test_wrong_image_binding_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    _spdx(path, "other-image")
    with pytest.raises(RuntimeError, match="locked image"):
        validate_spdx(path, expected_image="expected-image")
