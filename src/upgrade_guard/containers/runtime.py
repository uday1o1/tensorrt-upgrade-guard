"""Isolated Docker GPU worker execution."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from upgrade_guard.containers.commands import CommandResult, CommandRunner, Runner
from upgrade_guard.containers.security import validate_locked_image, validated_mount
from upgrade_guard.errors import InfrastructureError, InvalidInputError


@dataclass(frozen=True)
class WorkerMounts:
    """Narrow source, corpus, and result mounts."""

    source: Path
    corpus: Path
    output: Path


class DockerGpuWorker:
    """Launch exactly one GPU with no network, privileges, or Docker socket."""

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or CommandRunner()

    def run(
        self,
        *,
        image: str,
        gpu_uuid: str,
        mounts: WorkerMounts,
        command: Sequence[str],
        timeout_seconds: float,
    ) -> CommandResult:
        """Execute one bounded worker command and require successful completion."""

        if not gpu_uuid.startswith("GPU-") or any(character.isspace() for character in gpu_uuid):
            raise InvalidInputError("selected GPU UUID is malformed")
        locked_image = validate_locked_image(image)
        source = validated_mount(mounts.source, must_exist=True)
        corpus = validated_mount(mounts.corpus, must_exist=True)
        mounts.output.mkdir(parents=True, exist_ok=True)
        output = validated_mount(mounts.output, must_exist=True)
        if not command or any("\x00" in argument for argument in command):
            raise InvalidInputError("worker command must be a NUL-free argument array")
        arguments = (
            "docker",
            "run",
            "--rm",
            "--init",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--gpus",
            f"device={gpu_uuid}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--ipc",
            "private",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=1073741824",  # noqa: S108
            "--mount",
            f"type=bind,src={source},dst=/opt/upgrade-guard,readonly",
            "--mount",
            f"type=bind,src={corpus},dst=/corpus,readonly",
            "--mount",
            f"type=bind,src={output},dst=/output",
            "--env",
            "PYTHONPATH=/opt/upgrade-guard/src",
            locked_image,
            *command,
        )
        result = self.runner.run(arguments, timeout_seconds=timeout_seconds)
        if result.returncode != 0:
            raise InfrastructureError(
                "isolated GPU worker command failed",
                details={
                    "returncode": result.returncode,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                },
            )
        return result
