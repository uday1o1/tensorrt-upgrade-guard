"""Hash, traversal, symlink, duplicate, expansion, and trust-gate tests."""

from __future__ import annotations

import json
import tarfile
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.factories import digest, environment_lock, failure_record
from upgrade_guard.cli import app
from upgrade_guard.containers.commands import CommandResult, command_sha256
from upgrade_guard.containers.runtime import WorkerMounts
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.bundle import (
    BundleManifest,
    CudaArchitectureBuild,
    LocalWorkerBuild,
    ReplayRequirements,
    SourceBuildRequest,
    WorkerBuildArgument,
    canonical_cmake_cuda_architecture,
    validate_cmake_cuda_command,
)
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.errors import FailureCode, InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.reproduce.bundle import BundleExport, export_bundle
from upgrade_guard.reproduce.run import (
    RebuiltWorkerImage,
    ReplayRecipe,
    ReplayResult,
    ReplayStep,
    ReplayTarget,
    _validate_recipe_cuda_architecture,
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
    cuda_compute_capability: str | None = None,
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
        payloads["containers/Dockerfile.worker"] = b"ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n"
        payloads["containers/requirements-worker.txt"] = b"package==1 --hash=sha256:test\n"
    if included_engine:
        payloads["trusted.plan"] = b"trusted executable bytes"
    references = {
        path: artifact(path, content, _media_type(path)) for path, content in payloads.items()
    }
    source_build = (
        SourceBuildRequest(
            sources=(references["plugin-source/plugin.cu"],),
            original_worker_image_manifest_digest=digest("8"),
            original_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            replay_requirements=ReplayRequirements(
                minimum_compute_capability="8.0",
                minimum_driver="580.0",
                minimum_vram_mib=8192,
            ),
            cuda_architecture=(
                CudaArchitectureBuild(
                    original_compute_capability=cuda_compute_capability,
                    cmake_cuda_architecture=canonical_cmake_cuda_architecture(
                        cuda_compute_capability
                    ),
                )
                if cuda_compute_capability is not None
                else None
            ),
            local_worker_build=LocalWorkerBuild(
                base_image=f"registry.example/base@{digest('3')}",
                base_image_manifest_digest=digest("3"),
                dockerfile=references["containers/Dockerfile.worker"],
                worker_lock=references["containers/requirements-worker.txt"],
                build_arguments=(
                    WorkerBuildArgument(
                        name="BASE_IMAGE", value=f"registry.example/base@{digest('3')}"
                    ),
                    WorkerBuildArgument(name="BASE_MANIFEST_DIGEST", value=digest("3")),
                ),
                cuda_architecture=(
                    CudaArchitectureBuild(
                        original_compute_capability=cuda_compute_capability,
                        cmake_cuda_architecture=canonical_cmake_cuda_architecture(
                            cuda_compute_capability
                        ),
                    )
                    if cuda_compute_capability is not None
                    else None
                ),
            ),
            command=(
                (
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    "build",
                    "-DCMAKE_CUDA_ARCHITECTURES="
                    + canonical_cmake_cuda_architecture(cuda_compute_capability),
                )
                if cuda_compute_capability is not None
                else ("cmake", "--build", "build")
            ),
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


def test_bundle_rejects_dangling_expected_failure_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    manifest = create_bundle(root)
    dangling = ArtifactReference(
        path="logs/missing-failure.json",
        sha256=digest("a"),
        bytes=10,
        media_type="application/json",
    )
    failure = manifest.expected_failure.model_copy(update={"evidence": (dangling,)})
    changed = manifest.model_copy(update={"expected_failure": failure})
    changed = changed.model_copy(update={"manifest_sha256": changed.computed_sha256()})
    (root / "bundle.json").write_text(changed.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(InvalidInputError, match="expected-failure evidence"):
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
    manifest = create_bundle(root, source_code=True, included_engine=True)
    replay_target = ReplayTarget(
        gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
        compute_capability="8.9",
        driver_version="610.0",
        vram_mib=24576,
    )
    with pytest.raises(UnsupportedEnvironmentError, match="trust-source-code") as refused:
        prepare_replay(
            root,
            trust_source_code=False,
            trust_included_engine=False,
            replay_target=replay_target,
        )
    inventory = refused.value.details["review_inventory"]
    assert isinstance(inventory, dict)
    assert inventory["sources"] == [
        manifest.source_build.sources[0].model_dump(mode="json")  # type: ignore[union-attr]
    ]
    assert inventory["original_worker_image_manifest_digest"] == digest("8")
    assert inventory["worker_rebuild_recipe"]["base_image_manifest_digest"] == digest("3")  # type: ignore[index]
    assert inventory["worker_rebuild_recipe_sha256"] == (
        manifest.source_build.local_worker_build.computed_sha256()  # type: ignore[union-attr]
    )
    assert inventory["build_command"] == ["cmake", "--build", "build"]
    assert inventory["build_command_sha256"] == command_sha256(("cmake", "--build", "build"))
    compatibility = inventory["target_compatibility"]
    assert isinstance(compatibility, dict)
    assert compatibility["selected_gpu"] == replay_target.model_dump(mode="json")
    assert compatibility["status"] == "passed"
    assert compatibility["reasons"] == []
    with pytest.raises(UnsupportedEnvironmentError, match="trust-included-engine"):
        prepare_replay(
            root,
            trust_source_code=True,
            trust_included_engine=False,
            replay_target=replay_target,
        )
    plan = prepare_replay(
        root,
        trust_source_code=True,
        trust_included_engine=True,
        replay_target=replay_target,
    )
    assert plan.source_paths == ("plugin-source/plugin.cu",)
    assert plan.build_commands == (("cmake", "--build", "build"),)
    assert plan.original_gpu_uuid == "GPU-11111111-1111-1111-1111-111111111111"
    assert plan.selected_replay_gpu_uuid == replay_target.gpu_uuid
    assert plan.base_image == f"registry.example/base@{digest('3')}"
    assert plan.included_engine_trusted
    assert plan.review_inventory is not None
    assert plan.review_inventory.model_dump(mode="json") == inventory
    with pytest.raises(UnsupportedEnvironmentError, match="GPU worker"):
        require_gpu_for_replay()


@pytest.mark.parametrize(("capability", "architecture"), [("8.9", "89"), ("12.0", "120")])
def test_cuda_compute_capability_has_canonical_cmake_conversion(
    capability: str, architecture: str
) -> None:
    assert canonical_cmake_cuda_architecture(capability) == architecture
    value = CudaArchitectureBuild(
        original_compute_capability=capability,
        cmake_cuda_architecture=architecture,
    )
    command = ("cmake", "-S", ".", f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
    validate_cmake_cuda_command(command, value)
    with pytest.raises(ValueError, match="requires locked architecture"):
        validate_cmake_cuda_command(command, None)
    with pytest.raises(ValueError, match="wrong or ambiguous"):
        validate_cmake_cuda_command(("cmake", "-S", ".", "-DCMAKE_CUDA_ARCHITECTURES=75"), value)


def test_every_replay_cmake_configure_step_uses_locked_architecture() -> None:
    architecture = CudaArchitectureBuild(
        original_compute_capability="8.9",
        cmake_cuda_architecture="89",
    )
    recipe = ReplayRecipe(
        schema_version="upgradeguard.dev/replay-recipe/v1",
        expected_failure_code=FailureCode.PROFILE_REJECTED,
        steps=(
            ReplayStep(
                id="configure",
                command=("cmake", "-S", ".", "-DCMAKE_CUDA_ARCHITECTURES=89"),
            ),
            ReplayStep(
                id="nested-configure",
                command=("cmake", "-S", "nested", "-DCMAKE_CUDA_ARCHITECTURES=75"),
            ),
            ReplayStep(
                id="failure",
                command=("python3", "fail.py"),
                expected_failure_code=FailureCode.PROFILE_REJECTED,
                failure_code_source="stdout",
            ),
        ),
    )
    with pytest.raises(InvalidInputError, match="architecture is invalid"):
        _validate_recipe_cuda_architecture(recipe, architecture)


@pytest.mark.parametrize("capability", ["", "8", "8.90", "08.9", "sm_89", "8.x"])
def test_cuda_compute_capability_rejects_missing_or_malformed_values(
    capability: str,
) -> None:
    with pytest.raises(ValueError, match="canonical major.minor"):
        canonical_cmake_cuda_architecture(capability)
    with pytest.raises(ValueError):
        CudaArchitectureBuild(
            original_compute_capability=capability,
            cmake_cuda_architecture="89",
        )


def test_cuda_architecture_is_command_hashed_and_target_checked_before_build(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "cuda-source-bundle"
    manifest = create_bundle(
        bundle,
        source_code=True,
        cuda_compute_capability="8.9",
    )
    source_build = manifest.source_build
    assert source_build is not None
    assert source_build.cuda_architecture == CudaArchitectureBuild(
        original_compute_capability="8.9",
        cmake_cuda_architecture="89",
    )
    assert source_build.local_worker_build.cuda_architecture == source_build.cuda_architecture
    assert (
        source_build.local_worker_build.computed_sha256()
        != source_build.local_worker_build.model_copy(
            update={"cuda_architecture": None}
        ).computed_sha256()
    )
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in source_build.command
    matching = ReplayTarget(
        gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
        compute_capability="8.9",
        driver_version="610.0",
        vram_mib=24576,
    )
    plan = prepare_replay(
        bundle,
        trust_source_code=True,
        trust_included_engine=False,
        replay_target=matching,
    )
    assert plan.cuda_architecture == source_build.cuda_architecture
    assert plan.review_inventory is not None
    assert plan.review_inventory.build_command_sha256 == command_sha256(source_build.command)
    mismatched = matching.model_copy(update={"compute_capability": "12.0"})
    with pytest.raises(UnsupportedEnvironmentError, match="does not satisfy") as error:
        prepare_replay(
            bundle,
            trust_source_code=True,
            trust_included_engine=False,
            replay_target=mismatched,
        )
    assert "architecture 120 differs from locked 89" in str(error.value.details["reasons"])


def test_source_review_inventory_is_identical_for_refused_and_trusted_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    redaction_marker = "not-rendered-local-value"
    monkeypatch.setenv("UPGRADE_GUARD_TEST_TOKEN", redaction_marker)
    bundle = tmp_path / redaction_marker / "bundle"
    bundle.parent.mkdir()
    manifest = create_bundle(bundle, source_code=True)
    source_build = manifest.source_build
    assert source_build is not None
    replay_target = ReplayTarget(
        gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
        compute_capability="8.9",
        driver_version="610.0",
        vram_mib=24576,
    )
    monkeypatch.setattr("upgrade_guard.cli.observe_replay_target", lambda gpu: replay_target)
    runner = CliRunner()
    base_arguments = [
        "reproduce",
        "run",
        str(bundle),
        "--out",
        str(tmp_path / "replay"),
        "--gpu",
        replay_target.gpu_uuid,
    ]

    refused_json = runner.invoke(app, [*base_arguments, "--json"])
    assert refused_json.exit_code == 3
    refused_payload = json.loads(refused_json.stdout)
    inventory = refused_payload["details"]["review_inventory"]
    assert inventory["schema_version"] == "upgradeguard.dev/replay-source-review/v1"
    assert inventory["bundle_manifest_sha256"] == manifest.manifest_sha256
    assert inventory["sources"] == [
        {
            "path": source_build.sources[0].path,
            "sha256": source_build.sources[0].sha256,
            "bytes": source_build.sources[0].bytes,
            "media_type": source_build.sources[0].media_type,
        }
    ]
    assert inventory["original_worker_image_manifest_digest"] == digest("8")
    assert inventory["worker_rebuild_recipe"]["base_image"] == (
        f"registry.example/base@{digest('3')}"
    )
    assert inventory["worker_rebuild_recipe"]["base_image_manifest_digest"] == digest("3")
    assert inventory["worker_rebuild_recipe_sha256"] == (
        source_build.local_worker_build.computed_sha256()
    )
    assert inventory["build_command"] == ["cmake", "--build", "build"]
    assert inventory["build_command_sha256"] == command_sha256(source_build.command)
    assert inventory["target_compatibility"] == {
        "selected_gpu": replay_target.model_dump(mode="json"),
        "requirements": source_build.replay_requirements.model_dump(mode="json"),
        "status": "passed",
        "reasons": [],
    }
    assert redaction_marker not in refused_json.stdout
    assert str(tmp_path) not in refused_json.stdout
    assert all(not Path(source["path"]).is_absolute() for source in inventory["sources"])

    refused_human = runner.invoke(app, base_arguments)
    assert refused_human.exit_code == 3
    assert "Source review inventory:" in refused_human.stderr
    assert source_build.sources[0].path in refused_human.stderr
    assert source_build.sources[0].sha256 in refused_human.stderr
    assert inventory["build_command_sha256"] in refused_human.stderr
    assert replay_target.gpu_uuid in refused_human.stderr
    assert redaction_marker not in refused_human.stderr
    assert str(tmp_path) not in refused_human.stderr

    captured: dict[str, object] = {}

    def trusted_execute(
        source: Path,
        output: Path,
        *,
        trust_source_code: bool,
        trust_included_engine: bool,
        replay_target: ReplayTarget,
        image_builder: object,
    ) -> ReplayResult:
        del output, image_builder
        plan = prepare_replay(
            source,
            trust_source_code=trust_source_code,
            trust_included_engine=trust_included_engine,
            replay_target=replay_target,
        )
        assert plan.review_inventory is not None
        captured["inventory"] = plan.review_inventory.model_dump(mode="json")
        return ReplayResult(
            schema_version="upgradeguard.dev/replay-result/v1",
            status="passed",
            bundle_id=plan.bundle_id,
            bundle_manifest_sha256=manifest.manifest_sha256,
            worker_image=f"registry.example/worker@{digest('8')}",
            worker_rebuild_recipe_sha256=(source_build.local_worker_build.computed_sha256()),
            worker_build_log_sha256=digest("9"),
            worker_build_log=ArtifactReference(
                path="logs/worker-build.log",
                sha256=digest("9"),
                bytes=5,
                media_type="text/plain",
            ),
            original_gpu_uuid=source_build.original_gpu_uuid,
            selected_gpu_uuid=replay_target.gpu_uuid,
            expected_failure_code=FailureCode.NUMERICAL_REGRESSION,
            observed_failure_code=FailureCode.NUMERICAL_REGRESSION,
            step_results=("seeded-failure",),
        )

    monkeypatch.setattr("upgrade_guard.cli.execute_replay", trusted_execute)
    trusted = runner.invoke(app, [*base_arguments, "--trust-source-code", "--json"])
    assert trusted.exit_code == 0
    assert json.loads(trusted.stdout)["status"] == "passed"
    assert captured["inventory"] == inventory


def test_source_review_refusal_rejects_unsafe_manifest_path_without_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    create_bundle(bundle, source_code=True)
    value = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    value["source_build"]["sources"][0]["path"] = "../private-token.txt"
    (bundle / "bundle.json").write_text(json.dumps(value), encoding="utf-8")
    replay_target = ReplayTarget(
        gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
        compute_capability="8.9",
        driver_version="610.0",
        vram_mib=24576,
    )
    monkeypatch.setattr("upgrade_guard.cli.observe_replay_target", lambda gpu: replay_target)

    result = CliRunner().invoke(
        app,
        [
            "reproduce",
            "run",
            str(bundle),
            "--out",
            str(tmp_path / "replay"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error_code"] == "INVALID_INPUT"
    assert "review_inventory" not in payload["details"]


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
        "Dockerfile.worker": b"ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n",
        "requirements-worker.txt": b"package==1 --hash=sha256:test\n",
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
        original_worker_image_manifest_digest=digest("8"),
        original_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        base_image=f"registry.example/base@{digest('3')}",
        base_image_manifest_digest=digest("3"),
        dockerfile=sources / "Dockerfile.worker",
        worker_lock=sources / "requirements-worker.txt",
        worker_build_arguments=(
            ("BASE_IMAGE", f"registry.example/base@{digest('3')}"),
            ("BASE_MANIFEST_DIGEST", digest("3")),
        ),
        minimum_compute_capability="8.0",
        minimum_driver="580.0",
        minimum_vram_mib=8192,
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
    (sources / "Dockerfile.worker").write_text(
        "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n", encoding="utf-8"
    )
    (sources / "requirements-worker.txt").write_text(
        "package==1 --hash=sha256:test\n", encoding="utf-8"
    )
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
                "expected_failure_code": "NUMERICAL_REGRESSION",
                "failure_code_source": "result_file",
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
            original_worker_image_manifest_digest=environment.worker_image.manifest_digest,
            original_gpu_uuid=environment.probe.gpu.uuid,
            base_image=environment.base_image.canonical_reference,
            base_image_manifest_digest=environment.base_image.manifest_digest,
            dockerfile=sources / "Dockerfile.worker",
            worker_lock=sources / "requirements-worker.txt",
            worker_build_arguments=(
                ("BASE_IMAGE", environment.base_image.canonical_reference),
                ("BASE_MANIFEST_DIGEST", environment.base_image.manifest_digest),
            ),
            minimum_compute_capability="8.0",
            minimum_driver="580.0",
            minimum_vram_mib=8192,
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
                '{"status":"failed","failure_code":"NUMERICAL_REGRESSION",'
                '"message":"intended numerical failure"}'
            )
            return CommandResult(command, 1, '{"detected":true,"control":"passed"}', "", 0.2)

    class FakeImageBuilder:
        def build(
            self,
            *,
            bundle_root: Path,
            request: SourceBuildRequest,
            timeout_seconds: int,
        ) -> RebuiltWorkerImage:
            assert (bundle_root / request.local_worker_build.dockerfile.path).is_file()
            assert (bundle_root / request.local_worker_build.worker_lock.path).is_file()
            assert timeout_seconds == 1800
            return RebuiltWorkerImage(
                canonical_reference=environment.worker_image.canonical_reference,
                recipe_sha256=request.local_worker_build.computed_sha256(),
                build_log_sha256=sha256_bytes(b"built"),
                build_log="built",
            )

    worker = FakeWorker()
    image_builder = FakeImageBuilder()
    replay_target = ReplayTarget(
        gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
        compute_capability="8.9",
        driver_version="610.0",
        vram_mib=24576,
    )
    plan = prepare_replay(
        bundle,
        trust_source_code=True,
        trust_included_engine=False,
        replay_target=replay_target,
    )
    assert plan.original_gpu_uuid != plan.selected_replay_gpu_uuid
    with pytest.raises(UnsupportedEnvironmentError, match="does not satisfy"):
        prepare_replay(
            bundle,
            trust_source_code=True,
            trust_included_engine=False,
            replay_target=replay_target.model_copy(update={"compute_capability": "7.5"}),
        )
    replay = execute_replay(
        bundle,
        tmp_path / "replay",
        trust_source_code=True,
        trust_included_engine=False,
        worker=worker,  # type: ignore[arg-type]
        replay_target=replay_target,
        image_builder=image_builder,
    )
    assert replay.status == "passed"
    assert replay.expected_failure_code == "NUMERICAL_REGRESSION"
    assert replay.observed_failure_code is FailureCode.NUMERICAL_REGRESSION
    assert replay.original_gpu_uuid == environment.probe.gpu.uuid
    assert replay.selected_gpu_uuid == replay_target.gpu_uuid
    assert replay.worker_rebuild_recipe_sha256 == (
        verify_bundle(bundle).manifest.source_build.local_worker_build.computed_sha256()  # type: ignore[union-attr]
    )
    assert replay.worker_build_log_sha256 == sha256_bytes(b"built")
    assert replay.worker_build_log.sha256 == replay.worker_build_log_sha256
    retained_log = tmp_path / "replay" / replay.worker_build_log.path
    assert retained_log.read_text(encoding="utf-8") == "built"
    assert replay.step_results == ("build-engine", "seeded-failure")
    assert worker.calls[1][1] == (1,)
    assert (
        json.loads((tmp_path / "replay" / "replay-result.json").read_text())[
            "bundle_manifest_sha256"
        ]
        == replay.bundle_manifest_sha256
    )
    replay_document = json.loads(
        (tmp_path / "replay" / "replay-result.json").read_text(encoding="utf-8")
    )
    assert replay_document["observed_failure_code"] == "NUMERICAL_REGRESSION"
    seeded_step = json.loads(
        (tmp_path / "replay" / "steps" / "seeded-failure.json").read_text(encoding="utf-8")
    )
    assert seeded_step["expected_failure_code"] == "NUMERICAL_REGRESSION"
    assert seeded_step["observed_failure_code"] == "NUMERICAL_REGRESSION"
    assert "--trust-source-code" in (bundle / "README.md").read_text()

    class MismatchedFailureWorker(FakeWorker):
        def run(self, **kwargs: object) -> CommandResult:
            result = super().run(**kwargs)
            if len(self.calls) == 2:
                output = kwargs["mounts"]
                assert isinstance(output, WorkerMounts)
                (output.output / "failure.json").write_text(
                    '{"status":"failed","failure_code":"EXECUTION_FAILED",'
                    '"message":"intended numerical failure"}',
                    encoding="utf-8",
                )
            return result

    mismatch_output = tmp_path / "mismatched-replay"
    with pytest.raises(InvalidInputError, match="failure code differed"):
        execute_replay(
            bundle,
            mismatch_output,
            trust_source_code=True,
            trust_included_engine=False,
            worker=MismatchedFailureWorker(),  # type: ignore[arg-type]
            replay_target=replay_target,
            image_builder=image_builder,
        )
    assert not mismatch_output.exists()

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
            replay_target=replay_target,
            image_builder=image_builder,
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
            replay_target=replay_target,
            image_builder=image_builder,
        )

    with pytest.raises(InvalidInputError, match="outside"):
        execute_replay(
            bundle,
            bundle / "replay",
            trust_source_code=True,
            trust_included_engine=False,
            worker=worker,  # type: ignore[arg-type]
            replay_target=replay_target,
            image_builder=image_builder,
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
    duplicate = ReplayStep(
        id="same",
        command=("true",),
        expected_failure_code=FailureCode.PROFILE_REJECTED,
        failure_code_source="stdout",
    )
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
        expected_failure_code=FailureCode.PROFILE_REJECTED,
        failure_code_source="result_file",
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
        '{"status":"failed","failure_code":"PROFILE_REJECTED",'
        '"message":"the intended reason was observed"}'
    )
    assert _validate_step_evidence(result_step, "", work) is FailureCode.PROFILE_REJECTED
    (work / "result.json").write_text(
        '{"status":"failed","failure_code":"EXECUTION_FAILED",'
        '"message":"the intended reason was observed"}'
    )
    with pytest.raises(InvalidInputError, match="failure code differed"):
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
        original_worker_image_manifest_digest=digest("8"),
    )
    with pytest.raises(InvalidInputError, match="metadata requires source"):
        export_bundle(request, tmp_path / "invalid-bundle")

    valid = replace(request, original_worker_image_manifest_digest=None, included_engine=model)
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
