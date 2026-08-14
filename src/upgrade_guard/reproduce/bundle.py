"""Atomic export of hash-verified reproduction bundles."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.bundle import (
    BundleManifest,
    CudaArchitectureBuild,
    LocalWorkerBuild,
    ReplayRequirements,
    SourceBuildRequest,
    WorkerBuildArgument,
    canonical_cmake_cuda_architecture,
    is_cmake_configure_command,
)
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord
from upgrade_guard.errors import InvalidInputError


@dataclass(frozen=True, slots=True)
class BundleExport:
    """Authored bundle sources and the stable expected predicate."""

    id: str
    created_at: datetime
    baseline_environment: Path
    candidate_environment: Path
    qualification: Path
    model: Path
    inputs: tuple[Path, ...]
    expected_failure: FailureRecord
    extra_files: Mapping[str, Path]
    source_files: Mapping[str, Path]
    original_worker_image_manifest_digest: str | None = None
    original_gpu_uuid: str | None = None
    base_image: str | None = None
    base_image_manifest_digest: str | None = None
    dockerfile: Path | None = None
    worker_lock: Path | None = None
    worker_build_arguments: tuple[tuple[str, str], ...] = ()
    minimum_compute_capability: str | None = None
    minimum_driver: str | None = None
    minimum_vram_mib: int | None = None
    original_compute_capability: str | None = None
    source_build_command: tuple[str, ...] = ()
    included_engine: Path | None = None


def export_bundle(request: BundleExport, destination: Path) -> BundleManifest:
    """Copy reviewed inputs into a new immutable bundle and hash every payload."""

    if destination.exists() or destination.is_symlink():
        raise InvalidInputError("refusing to overwrite reproduction bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        authored: dict[str, Path | bytes] = {
            "baseline.environment.json": request.baseline_environment,
            "candidate.environment.json": request.candidate_environment,
            "qualification.yaml": request.qualification,
            "model.onnx": request.model,
            "expected.json": (
                request.expected_failure.model_dump_json(indent=2).encode("utf-8") + b"\n"
            ),
            "README.md": _readme(request).encode("utf-8"),
            "reproduce.sh": _reproduce_script(request).encode("utf-8"),
        }
        input_names: list[str] = []
        for index, input_source in enumerate(request.inputs):
            name = f"inputs/{index:03d}-{input_source.name}"
            input_names.append(name)
            authored[name] = input_source
        if request.source_files:
            if request.dockerfile is not None:
                authored["containers/Dockerfile.worker"] = request.dockerfile
            if request.worker_lock is not None:
                authored["containers/requirements-worker.txt"] = request.worker_lock
        _merge_unique(authored, request.extra_files)
        _merge_unique(authored, request.source_files)
        engine_name: str | None = None
        if request.included_engine is not None:
            engine_name = "trusted.plan"
            authored[engine_name] = request.included_engine
        references: dict[str, ArtifactReference] = {}
        for name, payload_source in sorted(authored.items()):
            target = _safe_destination(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload_source, bytes):
                target.write_bytes(payload_source)
            else:
                _copy_regular_file(payload_source, target)
            references[name] = _reference(name, target)
        checksum_lines = [
            f"{reference.sha256.removeprefix('sha256:')}  {name}"
            for name, reference in sorted(references.items())
        ]
        checksum_path = staging / "SHA256SUMS"
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        references["SHA256SUMS"] = _reference("SHA256SUMS", checksum_path)

        source_build = _source_build(request, references)
        manifest = BundleManifest(
            api_version="upgradeguard.dev/v1alpha1",
            kind="ReproductionBundle",
            id=request.id,
            created_at=request.created_at,
            files=tuple(references[name] for name in sorted(references)),
            baseline_environment=references["baseline.environment.json"],
            candidate_environment=references["candidate.environment.json"],
            qualification=references["qualification.yaml"],
            expected_failure=request.expected_failure,
            model=references["model.onnx"],
            inputs=tuple(references[name] for name in input_names),
            source_build=source_build,
            included_engine=references.get(engine_name) if engine_name else None,
            manifest_sha256="sha256:" + "0" * 64,
        )
        manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
        (staging / "bundle.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _merge_unique(destination: dict[str, Path | bytes], sources: Mapping[str, Path]) -> None:
    for name, source in sources.items():
        if name in destination or name in {"bundle.json", "SHA256SUMS"}:
            raise InvalidInputError(f"duplicate or reserved bundle path: {name}")
        destination[name] = source


def _source_build(
    request: BundleExport, references: Mapping[str, ArtifactReference]
) -> SourceBuildRequest | None:
    if not request.source_files:
        if any(
            (
                request.original_worker_image_manifest_digest,
                request.original_gpu_uuid,
                request.base_image,
                request.base_image_manifest_digest,
                request.dockerfile,
                request.worker_lock,
                request.worker_build_arguments,
                request.minimum_compute_capability,
                request.minimum_driver,
                request.minimum_vram_mib,
                request.original_compute_capability,
                request.source_build_command,
            )
        ):
            raise InvalidInputError("source build metadata requires source files")
        return None
    if (
        request.original_worker_image_manifest_digest is None
        or request.original_gpu_uuid is None
        or request.base_image is None
        or request.base_image_manifest_digest is None
        or request.dockerfile is None
        or request.worker_lock is None
        or not request.worker_build_arguments
        or request.minimum_compute_capability is None
        or request.minimum_driver is None
        or request.minimum_vram_mib is None
        or not request.source_build_command
    ):
        raise InvalidInputError(
            "source-bearing bundles require original provenance, portable worker rebuild, "
            "replay requirements, and source build command"
        )
    cuda_architecture: CudaArchitectureBuild | None = None
    if request.original_compute_capability is not None:
        try:
            cuda_architecture = CudaArchitectureBuild(
                original_compute_capability=request.original_compute_capability,
                cmake_cuda_architecture=canonical_cmake_cuda_architecture(
                    request.original_compute_capability
                ),
            )
        except ValueError as error:
            raise InvalidInputError("original compute capability is invalid") from error
    if is_cmake_configure_command(request.source_build_command) and cuda_architecture is None:
        raise InvalidInputError("CMake source builds require original compute capability evidence")
    return SourceBuildRequest(
        sources=tuple(references[name] for name in sorted(request.source_files)),
        original_worker_image_manifest_digest=request.original_worker_image_manifest_digest,
        original_gpu_uuid=request.original_gpu_uuid,
        replay_requirements=ReplayRequirements(
            minimum_compute_capability=request.minimum_compute_capability,
            minimum_driver=request.minimum_driver,
            minimum_vram_mib=request.minimum_vram_mib,
        ),
        cuda_architecture=cuda_architecture,
        local_worker_build=LocalWorkerBuild(
            base_image=request.base_image,
            base_image_manifest_digest=request.base_image_manifest_digest,
            dockerfile=references["containers/Dockerfile.worker"],
            worker_lock=references["containers/requirements-worker.txt"],
            build_arguments=tuple(
                WorkerBuildArgument(name=name, value=value)
                for name, value in request.worker_build_arguments
            ),
            cuda_architecture=cuda_architecture,
        ),
        command=request.source_build_command,
    )


def _copy_regular_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise InvalidInputError(f"bundle source must be a regular file: {source}")
    shutil.copyfile(source, destination)


def _safe_destination(root: Path, relative: str) -> Path:
    reference = ArtifactReference(
        path=relative,
        sha256="sha256:" + "0" * 64,
        bytes=0,
        media_type="application/octet-stream",
    )
    destination = (root / reference.path).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise InvalidInputError("bundle path escaped its staging directory")
    return destination


def _reference(relative: str, path: Path) -> ArtifactReference:
    return ArtifactReference(
        path=relative,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        media_type=_media_type(relative),
    )


def _media_type(path: str) -> str:
    if path == "containers/Dockerfile.worker":
        return "text/x-dockerfile"
    suffix = Path(path).suffix.lower()
    return {
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".onnx": "application/onnx",
        ".npy": "application/x-npy",
        ".md": "text/markdown",
        ".sh": "text/x-shellscript",
        ".cpp": "text/x-c++src",
        ".cu": "text/x-cuda",
        ".cuh": "text/x-cuda",
        ".hpp": "text/x-c++hdr",
        ".h": "text/x-chdr",
        ".cmake": "text/x-cmake",
        ".plan": "application/octet-stream",
        ".py": "text/x-python",
    }.get(suffix, "application/octet-stream")


def _readme(request: BundleExport) -> str:
    trust = (
        "This bundle contains source code and requires --trust-source-code after review.\n"
        "The original GPU UUID is provenance only; select a compatible replay GPU.\n"
        "The worker is rebuilt locally from the hash-verified Dockerfile and lock.\n"
        if request.source_files
        else "This bundle contains no source build request.\n"
    )
    return (
        f"# Reproduction {request.id}\n\n"
        f"Expected failure: `{request.expected_failure.code.value}`.\n"
        "Verify every artifact before replay.\n"
        f"{trust}"
        "Serialized engines and containers are executable trust boundaries.\n\n"
        "The replay requires a Docker Registry v2 endpoint at 127.0.0.1:5500 by default.\n"
        "Pass --gpu when more than one GPU is visible and --local-registry for another "
        "operator-owned localhost endpoint.\n\n"
        "```bash\n"
        "upgrade-guard reproduce verify .\n"
        f"upgrade-guard reproduce run . --out ../replay{_trust_flags(request)}\n"
        "```\n"
    )


def _trust_flags(request: BundleExport) -> str:
    flags = ""
    if request.source_files:
        flags += " --trust-source-code"
    if request.included_engine is not None:
        flags += " --trust-included-engine"
    return flags


def _reproduce_script(request: BundleExport) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        'bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec upgrade-guard reproduce run "${bundle_root}" '
        f'--out "${{bundle_root}}/../replay-{request.id}"{_trust_flags(request)}\n'
    )
