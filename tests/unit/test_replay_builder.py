"""Portable local replay-worker build boundary tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.factories import digest
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.bundle import (
    LocalWorkerBuild,
    ReplayRequirements,
    SourceBuildRequest,
    WorkerBuildArgument,
)
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.errors import InfrastructureError, InvalidInputError
from upgrade_guard.reproduce.builder import LocalDockerReplayImageBuilder


class QueueRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = results
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        self.commands.append(command)
        result = self.results.pop(0)
        return CommandResult(command, result.returncode, result.stdout, result.stderr, 0.01)


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult((), returncode, stdout, stderr, 0.01)


def _request(root: Path) -> SourceBuildRequest:
    dockerfile = root / "containers" / "Dockerfile.worker"
    lock = root / "containers" / "requirements-worker.txt"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n", encoding="utf-8")
    lock.write_text("package==1 --hash=sha256:test\n", encoding="utf-8")

    def reference(path: Path, media_type: str) -> ArtifactReference:
        from upgrade_guard.contracts.base import sha256_file

        return ArtifactReference(
            path=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            media_type=media_type,
        )

    return SourceBuildRequest(
        sources=(reference(dockerfile, "text/x-dockerfile"),),
        original_worker_image_manifest_digest=digest("8"),
        original_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        replay_requirements=ReplayRequirements(
            minimum_compute_capability="8.0",
            minimum_driver="580.0",
            minimum_vram_mib=8192,
        ),
        local_worker_build=LocalWorkerBuild(
            base_image=f"registry.example/base@{digest('3')}",
            base_image_manifest_digest=digest("3"),
            dockerfile=reference(dockerfile, "text/x-dockerfile"),
            worker_lock=reference(lock, "text/plain"),
            build_arguments=(
                WorkerBuildArgument(
                    name="BASE_IMAGE", value=f"registry.example/base@{digest('3')}"
                ),
                WorkerBuildArgument(name="BASE_MANIFEST_DIGEST", value=digest("3")),
            ),
        ),
        command=("cmake", "--build", "/output/build"),
    )


def test_builds_linux_amd64_and_returns_pushed_immutable_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)
    repository = "127.0.0.1:5500/upgrade-guard/replay-worker"
    canonical = f"{repository}@{digest('a')}"
    runner = QueueRunner(
        [
            _result(stdout="built\n"),
            _result(stdout="pushed\n"),
            _result(stdout=json.dumps([{"RepoDigests": [canonical]}])),
        ]
    )
    result = LocalDockerReplayImageBuilder("127.0.0.1:5500", runner).build(
        bundle_root=tmp_path,
        request=request,
        timeout_seconds=900,
    )
    build = runner.commands[0]
    assert build[:6] == (
        "docker",
        "build",
        "--pull=false",
        "--platform",
        "linux/amd64",
        "--build-arg",
    )
    assert result.canonical_reference == canonical
    assert result.recipe_sha256 == request.local_worker_build.computed_sha256()
    assert result.build_log == "built\npushed\n" + json.dumps([{"RepoDigests": [canonical]}])
    assert result.build_log_sha256 == sha256_bytes(result.build_log.encode("utf-8"))


def test_builder_rejects_nonlocal_registry_and_missing_repo_digest(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="localhost registry"):
        LocalDockerReplayImageBuilder("registry.example:5000")
    request = _request(tmp_path)
    runner = QueueRunner([_result(), _result(), _result(stdout=json.dumps([{"RepoDigests": []}]))])
    with pytest.raises(InfrastructureError, match="no immutable repository digest"):
        LocalDockerReplayImageBuilder("localhost:5500", runner).build(
            bundle_root=tmp_path,
            request=request,
            timeout_seconds=900,
        )
