"""Hash, traversal, symlink, duplicate, expansion, and trust-gate tests."""

from __future__ import annotations

import json
import tarfile
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.factories import digest, environment_lock, failure_record
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.runtime import WorkerMounts
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.bundle import BundleManifest, SourceBuildRequest
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.reproduce.bundle import BundleExport, export_bundle
from upgrade_guard.reproduce.run import (
    ReplayRecipe,
    ReplayStep,
    _validate_step_evidence,
    execute_replay,
    prepare_replay,
    require_gpu_for_replay,
)
from upgrade_guard.reproduce.verify import materialize_verified_bundle, verify_bundle


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


def test_exported_bundle_verifies_and_materializes_into_clean_directory(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    source_payloads = {
        "baseline.json": b'{"id":"baseline"}\n',
        "candidate.json": b'{"id":"candidate"}\n',
        "qualification.yaml": b"kind: Qualification\n",
        "model.onnx": b"frozen-model",
        "input.npy": b"frozen-input",
        "plugin.cu": b"__global__ void plugin() {}\n",
    }
    for name, content in source_payloads.items():
        (sources / name).write_bytes(content)
    request = BundleExport(
        id="exported-001",
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        baseline_environment=sources / "baseline.json",
        candidate_environment=sources / "candidate.json",
        qualification=sources / "qualification.yaml",
        model=sources / "model.onnx",
        inputs=(sources / "input.npy",),
        expected_failure=failure_record(),
        extra_files={},
        source_files={"plugin-source/plugin.cu": sources / "plugin.cu"},
        worker_image_manifest_digest=digest("8"),
        selected_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        source_build_command=("cmake", "--build", "build"),
    )
    bundle = tmp_path / "bundle"
    manifest = export_bundle(request, bundle)
    verified = verify_bundle(bundle)
    assert verified.manifest == manifest
    assert verified.source_code_present
    assert "SHA256SUMS" in verified.observed_files

    fresh = tmp_path / "empty" / "materialized"
    copied = materialize_verified_bundle(bundle, fresh)
    assert copied.manifest.manifest_sha256 == manifest.manifest_sha256
    assert not (fresh / "replay").exists()
    with pytest.raises(UnsupportedEnvironmentError, match="trust-source-code"):
        prepare_replay(
            fresh,
            trust_source_code=False,
            trust_included_engine=False,
        )


def test_typed_replay_rebuilds_and_confirms_expected_failure(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    environment = environment_lock()
    (sources / "baseline.json").write_text(
        environment.model_copy(update={"id": "baseline"}).model_dump_json(), encoding="utf-8"
    )
    (sources / "candidate.json").write_text(environment.model_dump_json(), encoding="utf-8")
    (sources / "qualification.yaml").write_text("kind: Qualification\n", encoding="utf-8")
    (sources / "model.onnx").write_bytes(b"model")
    (sources / "input.npy").write_bytes(b"input")
    (sources / "worker.py").write_text("# reviewed source\n", encoding="utf-8")
    build_command = ("python3", "-m", "worker", "build")
    recipe = {
        "schema_version": "upgradeguard.dev/replay-recipe/v1",
        "expected_failure_code": "NUMERICAL_REGRESSION",
        "steps": [
            {
                "id": "build-engine",
                "command": list(build_command),
                "result_file": "build.json",
                "expected_result_status": "passed",
            },
            {
                "id": "seeded-failure",
                "command": ["python3", "-m", "worker", "fail"],
                "accepted_returncodes": [1],
                "result_file": "failure.json",
                "expected_result_status": "failed",
                "result_message_contains": "intended numerical failure",
                "stdout_json_equals": {"detected": True, "control": "passed"},
            },
        ],
    }
    (sources / "replay.json").write_text(json.dumps(recipe), encoding="utf-8")
    bundle = tmp_path / "bundle"
    export_bundle(
        BundleExport(
            id="typed-replay",
            created_at=datetime(2026, 8, 13, tzinfo=UTC),
            baseline_environment=sources / "baseline.json",
            candidate_environment=sources / "candidate.json",
            qualification=sources / "qualification.yaml",
            model=sources / "model.onnx",
            inputs=(sources / "input.npy",),
            expected_failure=failure_record(),
            extra_files={"commands/replay.json": sources / "replay.json"},
            source_files={"worker.py": sources / "worker.py"},
            worker_image_manifest_digest=environment.worker_image.manifest_digest,
            selected_gpu_uuid=environment.probe.gpu.uuid,
            source_build_command=build_command,
        ),
        bundle,
    )

    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

        def run(
            self,
            *,
            image: str,
            gpu_uuid: str,
            mounts: WorkerMounts,
            command: tuple[str, ...],
            timeout_seconds: float,
            accepted_returncodes: tuple[int, ...] = (0,),
        ) -> CommandResult:
            del image, gpu_uuid, timeout_seconds
            self.calls.append((command, accepted_returncodes))
            if len(self.calls) == 1:
                (mounts.output / "build.json").write_text('{"status":"passed"}')
                return CommandResult(command, 0, "", "", 0.1)
            (mounts.output / "failure.json").write_text(
                '{"status":"failed","message":"intended numerical failure"}'
            )
            return CommandResult(command, 1, '{"detected":true,"control":"passed"}', "", 0.2)

    worker = FakeWorker()
    replay = execute_replay(
        bundle,
        tmp_path / "replay",
        trust_source_code=True,
        trust_included_engine=False,
        worker=worker,  # type: ignore[arg-type]
    )
    assert replay.status == "passed"
    assert replay.expected_failure_code == "NUMERICAL_REGRESSION"
    assert replay.step_results == ("build-engine", "seeded-failure")
    assert worker.calls[1][1] == (1,)
    assert (
        json.loads((tmp_path / "replay" / "replay-result.json").read_text())[
            "bundle_manifest_sha256"
        ]
        == replay.bundle_manifest_sha256
    )
    assert "--trust-source-code" in (bundle / "README.md").read_text()

    class FailingWorker:
        def run(self, **kwargs: object) -> CommandResult:
            del kwargs
            raise RuntimeError("interrupted worker")

    failed_output = tmp_path / "failed-replay"
    with pytest.raises(RuntimeError, match="interrupted"):
        execute_replay(
            bundle,
            failed_output,
            trust_source_code=True,
            trust_included_engine=False,
            worker=FailingWorker(),  # type: ignore[arg-type]
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed-replay.*"))

    with pytest.raises(InvalidInputError, match="overwrite"):
        execute_replay(
            bundle,
            tmp_path / "replay",
            trust_source_code=True,
            trust_included_engine=False,
            worker=worker,  # type: ignore[arg-type]
        )

    with pytest.raises(InvalidInputError, match="outside"):
        execute_replay(
            bundle,
            bundle / "replay",
            trust_source_code=True,
            trust_included_engine=False,
            worker=worker,  # type: ignore[arg-type]
        )


def test_replay_recipe_and_evidence_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="argument arrays"):
        ReplayStep(id="empty", command=())
    with pytest.raises(ValueError, match="return codes"):
        ReplayStep(id="codes", command=("true",), accepted_returncodes=())
    with pytest.raises(ValueError, match="authored together"):
        ReplayStep(id="result", command=("true",), result_file="result.json")
    with pytest.raises(ValueError, match="requires result_file"):
        ReplayStep(id="message", command=("true",), result_message_contains="reason")
    with pytest.raises(ValueError, match="safe relative"):
        ReplayStep(
            id="escape",
            command=("true",),
            result_file="../result.json",
            expected_result_status="passed",
        )
    duplicate = ReplayStep(id="same", command=("true",))
    with pytest.raises(ValueError, match="unique"):
        ReplayRecipe(
            schema_version="upgradeguard.dev/replay-recipe/v1",
            expected_failure_code="PROFILE_REJECTED",
            steps=(duplicate, duplicate),
        )

    work = tmp_path / "work"
    work.mkdir()
    result_step = ReplayStep(
        id="result-status",
        command=("true",),
        result_file="result.json",
        expected_result_status="failed",
        result_message_contains="intended reason",
    )
    with pytest.raises(InvalidInputError, match="did not produce"):
        _validate_step_evidence(result_step, "", work)
    (work / "result.json").write_text('{"status":"passed"}')
    with pytest.raises(InvalidInputError, match="status differed"):
        _validate_step_evidence(result_step, "", work)
    (work / "result.json").write_text('{"status":"failed","message":"other"}')
    with pytest.raises(InvalidInputError, match="different reason"):
        _validate_step_evidence(result_step, "", work)
    (work / "result.json").write_text(
        '{"status":"failed","message":"the intended reason was observed"}'
    )
    _validate_step_evidence(result_step, "", work)

    stdout_step = ReplayStep(
        id="stdout",
        command=("true",),
        stdout_json_equals={"nested.detected": True},
    )
    with pytest.raises(InvalidInputError, match="valid JSON"):
        _validate_step_evidence(stdout_step, "not-json", work)
    with pytest.raises(InvalidInputError, match="JSON object"):
        _validate_step_evidence(stdout_step, "[]", work)
    with pytest.raises(InvalidInputError, match="lacks"):
        _validate_step_evidence(stdout_step, '{"nested":{}}', work)
    with pytest.raises(InvalidInputError, match="predicate differed"):
        _validate_step_evidence(stdout_step, '{"nested":{"detected":false}}', work)
    _validate_step_evidence(stdout_step, '{"nested":{"detected":true}}', work)


def test_export_rejects_source_metadata_without_sources(tmp_path: Path) -> None:
    source = tmp_path / "file.json"
    source.write_text("{}", encoding="utf-8")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    input_path = tmp_path / "input.npy"
    input_path.write_bytes(b"input")
    qualification = tmp_path / "qualification.yaml"
    qualification.write_text("kind: Qualification\n", encoding="utf-8")
    request = BundleExport(
        id="invalid",
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        baseline_environment=source,
        candidate_environment=source,
        qualification=qualification,
        model=model,
        inputs=(input_path,),
        expected_failure=failure_record(),
        extra_files={},
        source_files={},
        worker_image_manifest_digest=digest("8"),
    )
    with pytest.raises(InvalidInputError, match="metadata requires source"):
        export_bundle(request, tmp_path / "invalid-bundle")

    valid = replace(request, worker_image_manifest_digest=None, included_engine=model)
    destination = tmp_path / "valid-bundle"
    manifest = export_bundle(valid, destination)
    assert manifest.source_build is None
    assert manifest.included_engine is not None
    assert "contains no source" in (destination / "README.md").read_text(encoding="utf-8")
    with pytest.raises(InvalidInputError, match="overwrite"):
        export_bundle(valid, destination)

    duplicate = replace(valid, extra_files={"README.md": source})
    with pytest.raises(InvalidInputError, match="duplicate"):
        export_bundle(duplicate, tmp_path / "duplicate-bundle")
