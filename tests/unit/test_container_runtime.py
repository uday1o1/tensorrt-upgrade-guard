"""GPU worker isolation command tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.containers.security import validate_locked_image, validated_mount
from upgrade_guard.errors import InfrastructureError, InvalidInputError


class CapturingRunner:
    def __init__(self) -> None:
        self.arguments: tuple[str, ...] = ()

    def run(
        self,
        args: tuple[str, ...],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: object = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        self.arguments = tuple(args)
        return CommandResult(tuple(args), 0, "", "", 0.1)


def test_gpu_worker_command_has_locked_isolation_boundary(tmp_path: Path) -> None:
    source = tmp_path / "deep" / "source"
    corpus = tmp_path / "deep" / "corpus"
    output = tmp_path / "deep" / "output"
    source.mkdir(parents=True)
    corpus.mkdir(parents=True)
    runner = CapturingRunner()
    worker = DockerGpuWorker(runner)
    digest = "sha256:" + "1" * 64
    worker.run(
        image=f"registry.example/worker@{digest}",
        gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        mounts=WorkerMounts(source, corpus, output),
        command=("python3", "-m", "worker"),
        timeout_seconds=30,
    )
    arguments = runner.arguments
    assert "--network" in arguments and arguments[arguments.index("--network") + 1] == "none"
    assert "--read-only" in arguments
    assert "--cap-drop" in arguments and arguments[arguments.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in arguments
    assert not any("docker.sock" in item for item in arguments)
    assert "device=GPU-11111111-1111-1111-1111-111111111111" in arguments
    assert arguments[arguments.index("--name") + 1].startswith("upgrade-guard-worker-")
    assert arguments[arguments.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert ("--entrypoint", "") == arguments[
        arguments.index("--entrypoint") : arguments.index("--entrypoint") + 2
    ]
    assert "HOME=/home/upgrade-guard" in arguments
    assert "XDG_CACHE_HOME=/home/upgrade-guard/.cache" in arguments
    assert "XDG_CONFIG_HOME=/home/upgrade-guard/.config" in arguments
    assert "CUDA_CACHE_PATH=/home/upgrade-guard/.cache/cuda" in arguments
    assert any(item.startswith("/home/upgrade-guard:rw,") for item in arguments)


def test_gpu_worker_refuses_mutable_image_and_broad_mount(tmp_path: Path) -> None:
    directory = tmp_path / "deep" / "directory"
    directory.mkdir(parents=True)
    worker = DockerGpuWorker(CapturingRunner())
    with pytest.raises(InvalidInputError, match="immutable"):
        worker.run(
            image="registry.example/worker:latest",
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            mounts=WorkerMounts(directory, directory, directory / "output"),
            command=("true",),
            timeout_seconds=1,
        )


def test_worker_boundary_rejects_malformed_inputs_and_failed_process(tmp_path: Path) -> None:
    directory = tmp_path / "deep" / "directory"
    directory.mkdir(parents=True)
    with pytest.raises(InvalidInputError, match="does not exist"):
        validated_mount(directory / "missing", must_exist=True)
    with pytest.raises(InvalidInputError, match="broad"):
        validated_mount(Path("/"), must_exist=True)
    link = tmp_path / "deep" / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(InvalidInputError, match="symlink"):
        validated_mount(link, must_exist=True)
    with pytest.raises(InvalidInputError, match="malformed"):
        validate_locked_image("registry.example/image@sha256:not-a-digest")

    class FailingRunner(CapturingRunner):
        def run(self, args: tuple[str, ...], **kwargs: object) -> CommandResult:
            del kwargs
            return CommandResult(tuple(args), 17, "stdout", "stderr", 0.1)

    worker = DockerGpuWorker(FailingRunner())
    with pytest.raises(InfrastructureError, match="worker command failed"):
        worker.run(
            image="registry.example/image@sha256:" + "1" * 64,
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            mounts=WorkerMounts(directory, directory, directory / "output"),
            command=("false",),
            timeout_seconds=1,
        )
    with pytest.raises(InvalidInputError, match="UUID"):
        worker.run(
            image="registry.example/image@sha256:" + "1" * 64,
            gpu_uuid="0",
            mounts=WorkerMounts(directory, directory, directory / "output"),
            command=("true",),
            timeout_seconds=1,
        )
    with pytest.raises(InvalidInputError, match="command"):
        worker.run(
            image="registry.example/image@sha256:" + "1" * 64,
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            mounts=WorkerMounts(directory, directory, directory / "output"),
            command=(),
            timeout_seconds=1,
        )
    with pytest.raises(InvalidInputError, match="return codes"):
        worker.run(
            image="registry.example/image@sha256:" + "1" * 64,
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            mounts=WorkerMounts(directory, directory, directory / "output"),
            command=("true",),
            timeout_seconds=1,
            accepted_returncodes=(),
        )


def test_worker_timeout_cleans_only_its_unique_container(tmp_path: Path) -> None:
    directory = tmp_path / "deep" / "directory"
    directory.mkdir(parents=True)

    class TimeoutRunner:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        def run(self, args: tuple[str, ...], **kwargs: object) -> CommandResult:
            del kwargs
            command = tuple(args)
            self.commands.append(command)
            if command[:2] == ("docker", "run"):
                raise InfrastructureError("command timed out")
            return CommandResult(command, 0, "", "", 0.1)

    runner = TimeoutRunner()
    worker = DockerGpuWorker(runner)
    with pytest.raises(InfrastructureError, match="timed out"):
        worker.run(
            image="registry.example/image@sha256:" + "1" * 64,
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            mounts=WorkerMounts(directory, directory, directory / "output"),
            command=("python3", "-m", "worker"),
            timeout_seconds=1,
        )
    run_command, cleanup = runner.commands
    name = run_command[run_command.index("--name") + 1]
    assert cleanup == ("docker", "container", "rm", "--force", name)
    assert name.startswith("upgrade-guard-worker-")
