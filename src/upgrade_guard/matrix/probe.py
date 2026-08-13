"""Exact-image worker probing through an isolated Docker invocation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from upgrade_guard.containers.commands import CommandRunner, Runner, command_sha256
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.environment import ResolvedImage, WorkerProbe
from upgrade_guard.errors import InfrastructureError, InvalidInputError


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    """Validated probe plus exact command and output identities."""

    probe: WorkerProbe
    command_sha256: str
    output_sha256: str


class DockerWorkerProbe:
    """Pull and inspect one exact manifest, then expose one selected GPU."""

    def __init__(self, runner: Runner | None = None, *, script_path: Path | None = None) -> None:
        self.runner = runner or CommandRunner()
        self.script_path = script_path or Path(__file__).with_name("worker_probe.py")

    def run(self, image: ResolvedImage, gpu_uuid: str) -> ProbeExecution:
        """Run a network-isolated read-only probe by immutable manifest digest."""

        canonical = image.canonical_reference
        pull = self.runner.run(
            ("docker", "pull", "--platform", "linux/amd64", canonical),
            timeout_seconds=1800,
        )
        if pull.returncode != 0:
            raise InfrastructureError(
                "Docker could not pull the selected worker manifest",
                details={"image": canonical, "error": _bounded(pull.stderr, pull.stdout)},
            )
        self._verify_local_identity(canonical, image)

        script = self.script_path.resolve()
        if not script.is_file():
            raise InvalidInputError(f"worker probe script does not exist: {script}")
        if "," in str(script):
            raise InvalidInputError("worker probe path cannot contain a comma")
        container_name = f"upgrade-guard-probe-{uuid.uuid4().hex[:12]}"
        command = (
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--platform",
            "linux/amd64",
            "--gpus",
            f"device={gpu_uuid}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",  # noqa: S108
            "--mount",
            f"type=bind,source={script},target=/opt/upgrade-guard/worker_probe.py,readonly",
            "--env",
            f"UG_WORKER_MANIFEST_DIGEST={image.manifest_digest}",
            canonical,
            "python3",
            "/opt/upgrade-guard/worker_probe.py",
        )
        result = self.runner.run(command, timeout_seconds=300)
        if result.returncode != 0:
            self._cleanup_exact(container_name)
            raise InfrastructureError(
                "worker probe container failed",
                details={
                    "image": canonical,
                    "command_sha256": command_sha256(command),
                    "error": _bounded(result.stderr, result.stdout),
                },
            )
        try:
            payload: Any = json.loads(result.stdout)
            probe = WorkerProbe.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as error:
            raise InfrastructureError(
                "worker probe returned an invalid document",
                details={"image": canonical, "reason": str(error)},
            ) from error
        return ProbeExecution(
            probe=probe,
            command_sha256=command_sha256(command),
            output_sha256=sha256_bytes(result.stdout.encode()),
        )

    def _verify_local_identity(self, canonical: str, expected: ResolvedImage) -> None:
        result = self.runner.run(
            ("docker", "image", "inspect", "--format", "{{json .}}", canonical),
            timeout_seconds=30,
        )
        if result.returncode != 0:
            raise InfrastructureError(
                "Docker could not inspect the pulled worker image",
                details={"image": canonical, "error": _bounded(result.stderr, result.stdout)},
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise InfrastructureError("Docker returned invalid image inspection JSON") from error
        if not isinstance(value, dict):
            raise InfrastructureError("Docker image inspection must be a JSON object")
        if value.get("Id") != expected.config_digest:
            raise InfrastructureError(
                "Docker image configuration does not match the verified OCI config",
                details={"expected": expected.config_digest, "observed": value.get("Id")},
            )
        repo_digests = value.get("RepoDigests")
        if not isinstance(repo_digests, list) or not any(
            isinstance(item, str) and item.endswith(f"@{expected.manifest_digest}")
            for item in repo_digests
        ):
            raise InfrastructureError(
                "Docker image inspection does not retain the selected manifest digest",
                details={"expected": expected.manifest_digest},
            )

    def _cleanup_exact(self, container_name: str) -> None:
        self.runner.run(
            ("docker", "container", "rm", "--force", container_name),
            timeout_seconds=30,
        )


def _bounded(*values: str) -> str:
    return "\n".join(value.strip() for value in values if value.strip())[:1000]
