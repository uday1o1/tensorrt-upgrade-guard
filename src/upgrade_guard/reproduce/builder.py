"""Local content-addressed worker rebuild for portable replay bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path

from upgrade_guard.containers.commands import CommandRunner, Runner
from upgrade_guard.containers.security import validate_locked_image
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.bundle import SourceBuildRequest
from upgrade_guard.errors import InfrastructureError, InvalidInputError
from upgrade_guard.reproduce.run import RebuiltWorkerImage

_REGISTRY = re.compile(r"^(?:localhost|127\.0\.0\.1)(?::[0-9]{1,5})$")
_MAXIMUM_RETAINED_BUILD_LOG_CHARACTERS = 1_000_000


class LocalDockerReplayImageBuilder:
    """Build and push a reviewed bundle worker to an operator-owned local registry."""

    def __init__(self, registry: str, runner: Runner | None = None) -> None:
        if not _REGISTRY.fullmatch(registry):
            raise InvalidInputError("V1 replay registry must be an explicit localhost registry")
        self.registry = registry
        self.runner = runner or CommandRunner()

    def build(
        self,
        *,
        bundle_root: Path,
        request: SourceBuildRequest,
        timeout_seconds: int,
    ) -> RebuiltWorkerImage:
        """Build exact reviewed inputs and return the pushed manifest identity."""

        bundle_root = bundle_root.resolve(strict=True)
        recipe_sha256 = request.local_worker_build.computed_sha256()
        repository = f"{self.registry}/upgrade-guard/replay-worker"
        tag = f"{repository}:{recipe_sha256.removeprefix('sha256:')[:20]}"
        dockerfile = bundle_root / request.local_worker_build.dockerfile.path
        worker_lock = bundle_root / request.local_worker_build.worker_lock.path
        for path in (dockerfile, worker_lock):
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(bundle_root) or resolved.is_symlink():
                raise InvalidInputError("replay worker build input is unsafe")
        command = ["docker", "build", "--pull=false", "--platform", "linux/amd64"]
        for argument in request.local_worker_build.build_arguments:
            command.extend(("--build-arg", f"{argument.name}={argument.value}"))
        command.extend(("--tag", tag, "--file", str(dockerfile), str(bundle_root)))
        built = self.runner.run(
            tuple(command),
            timeout_seconds=timeout_seconds,
            cwd=bundle_root,
        )
        log = built.stdout + built.stderr
        if built.returncode != 0:
            raise InfrastructureError(
                "portable replay worker build failed",
                details={"returncode": built.returncode},
            )
        pushed = self.runner.run(
            ("docker", "push", tag),
            timeout_seconds=timeout_seconds,
        )
        log += pushed.stdout + pushed.stderr
        if pushed.returncode != 0:
            raise InfrastructureError(
                "portable replay worker push failed",
                details={"returncode": pushed.returncode},
            )
        inspected = self.runner.run(
            ("docker", "image", "inspect", tag),
            timeout_seconds=60,
        )
        log += inspected.stdout + inspected.stderr
        if inspected.returncode != 0:
            raise InfrastructureError("portable replay worker inspection failed")
        try:
            values = json.loads(inspected.stdout)
            repo_digests = values[0]["RepoDigests"]
            canonical = next(value for value in repo_digests if value.startswith(f"{repository}@"))
        except (IndexError, KeyError, StopIteration, TypeError, json.JSONDecodeError) as error:
            raise InfrastructureError(
                "portable replay worker has no immutable repository digest"
            ) from error
        if len(log) > _MAXIMUM_RETAINED_BUILD_LOG_CHARACTERS:
            omitted = len(log) - _MAXIMUM_RETAINED_BUILD_LOG_CHARACTERS
            log = (
                f"[upgrade-guard omitted {omitted} leading build-log characters]\n"
                + log[-_MAXIMUM_RETAINED_BUILD_LOG_CHARACTERS:]
            )
        return RebuiltWorkerImage(
            canonical_reference=validate_locked_image(canonical),
            recipe_sha256=recipe_sha256,
            build_log_sha256=sha256_bytes(log.encode("utf-8")),
            build_log=log,
        )
