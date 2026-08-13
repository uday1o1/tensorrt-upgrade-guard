"""Hash, traversal, symlink, duplicate, expansion, and trust-gate tests."""

from __future__ import annotations

import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.factories import digest, failure_record
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.bundle import BundleManifest, SourceBuildRequest
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.reproduce.run import prepare_replay, require_gpu_for_replay
from upgrade_guard.reproduce.verify import verify_bundle


def artifact(path: str, content: bytes, media_type: str = "application/json") -> ArtifactReference:
    return ArtifactReference(
        path=path,
        sha256=sha256_bytes(content),
        bytes=len(content),
        media_type=media_type,
    )


def create_bundle(
    root: Path,
    *,
    source_code: bool = False,
    included_engine: bool = False,
) -> BundleManifest:
    payloads = {
        "model.onnx": b"frozen-model",
        "inputs/input.npy": b"frozen-input",
        "baseline.environment.json": b'{"id":"baseline"}',
        "candidate.environment.json": b'{"id":"candidate"}',
        "qualification.yaml": b"kind: Qualification\n",
        "expected.json": b'{"failure":"NUMERICAL_REGRESSION"}',
        "README.md": b"# Reproduction\n",
        "reproduce.sh": b"#!/bin/sh\nexit 99\n",
    }
    if source_code:
        payloads["plugin-source/plugin.cu"] = b"__global__ void plugin() {}\n"
    if included_engine:
        payloads["trusted.plan"] = b"trusted executable bytes"
    references = {
        path: artifact(path, content, _media_type(path)) for path, content in payloads.items()
    }
    source_build = (
        SourceBuildRequest(
            sources=(references["plugin-source/plugin.cu"],),
            worker_image_manifest_digest=digest("8"),
            selected_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            command=("cmake", "--build", "build"),
        )
        if source_code
        else None
    )
    zero = digest("0")
    manifest = BundleManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ReproductionBundle",
        id="bundle-001",
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        files=tuple(references.values()),
        baseline_environment=references["baseline.environment.json"],
        candidate_environment=references["candidate.environment.json"],
        qualification=references["qualification.yaml"],
        expected_failure=failure_record(),
        model=references["model.onnx"],
        inputs=(references["inputs/input.npy"],),
        source_build=source_build,
        included_engine=references.get("trusted.plan"),
        manifest_sha256=zero,
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
    root.mkdir()
    for path, content in payloads.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (root / "bundle.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _media_type(path: str) -> str:
    if path.endswith(".onnx"):
        return "application/onnx"
    if path.endswith(".npy"):
        return "application/x-npy"
    if path.endswith(".md"):
        return "text/markdown"
    if path.endswith(".sh"):
        return "text/x-shellscript"
    if path.endswith(".cu"):
        return "text/x-cuda"
    return "application/json"


def test_directory_bundle_verifies_every_hash_without_executing_script(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    manifest = create_bundle(root)
    verified = verify_bundle(root)
    assert verified.manifest == manifest
    assert "reproduce.sh" in verified.observed_files
    assert not verified.source_code_present
    assert not verified.engine_present


def test_hash_size_inventory_and_self_hash_mismatches_fail(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    create_bundle(root)
    (root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(InvalidInputError, match="size mismatch|hash mismatch"):
        verify_bundle(root)

    root = tmp_path / "bundle-two"
    create_bundle(root)
    (root / "undeclared.txt").write_text("undeclared", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="inventory differs"):
        verify_bundle(root)

    root = tmp_path / "bundle-three"
    manifest = create_bundle(root)
    bad = manifest.model_copy(update={"manifest_sha256": digest("f")})
    (root / "bundle.json").write_text(bad.model_dump_json(), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="self-hash"):
        verify_bundle(root)


def test_directory_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    create_bundle(root)
    (root / "link.txt").symlink_to(root / "README.md")
    with pytest.raises(InvalidInputError, match="symlink"):
        verify_bundle(root)


def test_zip_and_tar_regular_bundles_verify(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    create_bundle(root)
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    assert verify_bundle(zip_path).manifest.id == "bundle-001"

    tar_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix())
    assert verify_bundle(tar_path).manifest.id == "bundle-001"


def test_archive_traversal_duplicate_and_symlink_are_rejected(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", "escape")
    with pytest.raises(InvalidInputError, match="unsafe bundle path"):
        verify_bundle(traversal)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("bundle.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("bundle.json", "{}")
    with pytest.raises(InvalidInputError, match="duplicate"):
        verify_bundle(duplicate)

    symlink = tmp_path / "symlink.tar"
    with tarfile.open(symlink, "w") as archive:
        member = tarfile.TarInfo("link.txt")
        member.type = tarfile.SYMTYPE
        member.linkname = "README.md"
        archive.addfile(member)
    with pytest.raises(InvalidInputError, match="non-regular"):
        verify_bundle(symlink)


def test_expansion_and_file_count_limits_fail_before_payload_reads(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    create_bundle(root)
    with pytest.raises(InvalidInputError, match="file-count"):
        verify_bundle(root, maximum_files=1)
    with pytest.raises(InvalidInputError, match="expanded-size"):
        verify_bundle(root, maximum_expanded_bytes=1)


def test_source_and_engine_require_independent_explicit_trust(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    create_bundle(root, source_code=True, included_engine=True)
    with pytest.raises(UnsupportedEnvironmentError, match="trust-source-code"):
        prepare_replay(
            root,
            trust_source_code=False,
            trust_included_engine=False,
        )
    with pytest.raises(UnsupportedEnvironmentError, match="trust-included-engine"):
        prepare_replay(
            root,
            trust_source_code=True,
            trust_included_engine=False,
        )
    plan = prepare_replay(
        root,
        trust_source_code=True,
        trust_included_engine=True,
    )
    assert plan.source_paths == ("plugin-source/plugin.cu",)
    assert plan.build_commands == (("cmake", "--build", "build"),)
    assert plan.included_engine_trusted
    with pytest.raises(UnsupportedEnvironmentError, match="GPU worker"):
        require_gpu_for_replay()


def test_unsupported_bundle_type_and_suffix_fail_closed(tmp_path: Path) -> None:
    plain = tmp_path / "bundle.bin"
    plain.write_bytes(b"not an archive")
    with pytest.raises(InvalidInputError, match="directory, ZIP, or tar"):
        verify_bundle(plain)

    root = tmp_path / "bundle"
    create_bundle(root)
    (root / "forbidden.exe").write_bytes(b"executable")
    with pytest.raises(InvalidInputError, match="unsupported bundle file type"):
        verify_bundle(root)
