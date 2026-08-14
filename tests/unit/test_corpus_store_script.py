"""Append-only content-addressed corpus store tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.corpus_store import (
    MATERIALIZER_NAME,
    corpus_index,
    materializer_document,
    publish,
    tree_inventory,
    verify_sidecar,
    write_sidecar,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _staging(tmp_path: Path, document: dict[str, object], name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "corpus.lock.json").write_text("{}\n", encoding="utf-8")
    (root / "model.bin").write_bytes(b"model")
    write_sidecar(root, document)
    return root


def test_materializer_document_binds_exact_sources() -> None:
    first = materializer_document(PROJECT_ROOT, "core")
    second = materializer_document(PROJECT_ROOT, "core")
    assert first == second
    assert first["materializer_sha256"].startswith("sha256:")
    assert "corpus/registry.yaml" in first["sources"]
    assert "uv.lock" in first["sources"]


def test_publish_is_append_only_and_reuses_only_identical_content(tmp_path: Path) -> None:
    document = materializer_document(PROJECT_ROOT, "plugin")
    identity = str(document["materializer_sha256"]).removeprefix("sha256:")
    destination = tmp_path / "by-id" / "plugin" / identity
    first = _staging(tmp_path, document, "first")
    assert publish(first, destination, document) == "published"
    assert destination.is_dir()
    assert not first.exists()
    second = _staging(tmp_path, document, "second")
    assert publish(second, destination, document) == "reused"
    assert not second.exists()
    assert list((tmp_path / "by-id" / "stale").iterdir())
    verify_sidecar(destination, document)


def test_same_identity_different_content_fails_closed(tmp_path: Path) -> None:
    document = materializer_document(PROJECT_ROOT, "mobilenet")
    destination = tmp_path / "by-id" / "mobilenet" / "identity"
    first = _staging(tmp_path, document, "first")
    publish(first, destination, document)
    destination.chmod(0o755)
    (destination / "model.bin").chmod(0o644)
    (destination / "model.bin").write_bytes(b"tampered")
    second = _staging(tmp_path, document, "second")
    with pytest.raises(ValueError, match="different content"):
        publish(second, destination, document)
    assert second.exists()


def test_sidecar_and_inventory_reject_tamper_and_symlink(tmp_path: Path) -> None:
    document = materializer_document(PROJECT_ROOT, "core")
    root = _staging(tmp_path, document, "root")
    sidecar = root / MATERIALIZER_NAME
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    value["materializer_sha256"] = "sha256:" + ("0" * 64)
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        verify_sidecar(root, document)
    sidecar.unlink()
    write_sidecar(root, document)
    (root / "link").symlink_to(root / "model.bin")
    with pytest.raises(ValueError, match="symlink"):
        tree_inventory(root)


def test_corpus_index_binds_lock_materializer_and_inventory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for relative in (
        "pyproject.toml",
        "uv.lock",
        "src/upgrade_guard/contracts/base.py",
        "src/upgrade_guard/corpus/generators.py",
        "src/upgrade_guard/corpus/reference.py",
        "corpus/registry.yaml",
        "models/generators/tiny_transformer.py",
        "models/locks/tiny_transformer.lock.json",
        "src/upgrade_guard/corpus/materialize.py",
        "src/upgrade_guard/corpus/registry.py",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    reference_environment_sha256 = "sha256:" + ("e" * 64)
    document = materializer_document(
        project,
        "core",
        reference_environment_sha256,
    )
    root = project / ".upgrade-guard/corpora/by-id/core/identity"
    root.mkdir(parents=True)
    (root / "corpus.lock.json").write_text(
        json.dumps(
            {"reference_environment_sha256": reference_environment_sha256},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sidecar(root, document)
    value = corpus_index(project, {"core": root})
    entry = value["corpora"]["core"]
    assert entry["root"].endswith("/core/identity")
    assert entry["materializer_sha256"] == document["materializer_sha256"]
    assert entry["lock_sha256"].startswith("sha256:")
    assert entry["inventory_sha256"].startswith("sha256:")
    assert entry["reference_environment_sha256"] == reference_environment_sha256
    assert value["reference_environment_sha256"] == reference_environment_sha256
