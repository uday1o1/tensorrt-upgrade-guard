"""Exact Docker worker probe command tests."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.factories import digest, resolved_image, worker_probe
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.gpu_runtime import observe_nvidia_container_toolkit_version
from upgrade_guard.errors import InfrastructureError, UnsupportedEnvironmentError
from upgrade_guard.matrix.probe import DockerWorkerProbe


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
        if not self.results:
            raise AssertionError(f"unexpected command: {command}")
        result = self.results.pop(0)
        return CommandResult(command, result.returncode, result.stdout, result.stderr, 0.01)


def result(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult((), returncode, stdout, stderr, 0.01)


def inspect_json(manifest_digest: str, config_digest: str) -> str:
    return json.dumps(
        {
            "Id": config_digest,
            "RepoDigests": [f"registry.example/upgrade/worker@{manifest_digest}"],
        }
    )


def test_probe_runs_exact_manifest_with_isolation(tmp_path: Path) -> None:
    script = tmp_path / "worker_probe.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    image = resolved_image()
    probe = worker_probe(manifest_digest=image.manifest_digest)
    runner = QueueRunner(
        [
            result(stdout="pulled"),
            result(stdout=inspect_json(image.manifest_digest, image.config_digest)),
            result(stdout=probe.model_dump_json()),
        ]
    )
    execution = DockerWorkerProbe(runner, script_path=script).run(
        image,
        "GPU-11111111-1111-1111-1111-111111111111",
    )
    assert execution.probe == probe
    assert execution.command_sha256.startswith("sha256:")
    run_command = runner.commands[2]
    assert image.canonical_reference in run_command
    assert ("--network", "none") == run_command[
        run_command.index("--network") : run_command.index("--network") + 2
    ]
    assert "--read-only" in run_command
    assert ("--cap-drop", "ALL") == run_command[
        run_command.index("--cap-drop") : run_command.index("--cap-drop") + 2
    ]
    assert "/var/run/docker.sock" not in " ".join(run_command)
    assert f"UG_WORKER_MANIFEST_DIGEST={image.manifest_digest}" in run_command
    assert run_command[run_command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert ("--entrypoint", "") == run_command[
        run_command.index("--entrypoint") : run_command.index("--entrypoint") + 2
    ]
    assert "HOME=/home/upgrade-guard" in run_command
    assert "XDG_CACHE_HOME=/home/upgrade-guard/.cache" in run_command
    assert "CUDA_CACHE_PATH=/home/upgrade-guard/.cache/cuda" in run_command
    assert any(item.startswith("/home/upgrade-guard:rw,") for item in run_command)


def test_probe_rejects_local_config_identity_mismatch(tmp_path: Path) -> None:
    script = tmp_path / "worker_probe.py"
    script.write_text("", encoding="utf-8")
    image = resolved_image()
    runner = QueueRunner(
        [
            result(stdout="pulled"),
            result(stdout=inspect_json(image.manifest_digest, digest("f"))),
        ]
    )
    with pytest.raises(InfrastructureError, match="configuration"):
        DockerWorkerProbe(runner, script_path=script).run(
            image,
            "GPU-11111111-1111-1111-1111-111111111111",
        )
    assert len(runner.commands) == 2


def test_failed_probe_cleans_only_its_exact_container(tmp_path: Path) -> None:
    script = tmp_path / "worker_probe.py"
    script.write_text("", encoding="utf-8")
    image = resolved_image()
    runner = QueueRunner(
        [
            result(stdout="pulled"),
            result(stdout=inspect_json(image.manifest_digest, image.config_digest)),
            result(returncode=1, stderr="probe failed"),
            result(returncode=0),
        ]
    )
    with pytest.raises(InfrastructureError, match="probe container failed"):
        DockerWorkerProbe(runner, script_path=script).run(
            image,
            "GPU-11111111-1111-1111-1111-111111111111",
        )
    run_command = runner.commands[2]
    cleanup = runner.commands[3]
    name = run_command[run_command.index("--name") + 1]
    assert cleanup == ("docker", "container", "rm", "--force", name)
    assert name.startswith("upgrade-guard-probe-")


def test_target_cdi_failure_uses_typed_toolkit_evidence(tmp_path: Path) -> None:
    script = tmp_path / "worker_probe.py"
    script.write_text("", encoding="utf-8")
    image = resolved_image()
    runner = QueueRunner(
        [
            result(stdout="pulled"),
            result(stdout=inspect_json(image.manifest_digest, image.config_digest)),
            result(
                returncode=1,
                stderr=(
                    "docker: Error response from daemon: failed to discover GPU vendor "
                    "from CDI: no known GPU vendor found"
                ),
            ),
            result(returncode=0),
        ]
    )

    class UnavailableToolkitRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: float = 30.0,
            cwd: Path | None = None,
            env: Mapping[str, str] | None = None,
        ) -> CommandResult:
            del timeout_seconds, cwd, env
            return CommandResult(tuple(args), 127, "", "command not found", 0.01)

    observation = observe_nvidia_container_toolkit_version(UnavailableToolkitRunner())
    with pytest.raises(UnsupportedEnvironmentError) as captured:
        DockerWorkerProbe(runner, script_path=script).run(
            image,
            "GPU-11111111-1111-1111-1111-111111111111",
            toolkit_observation=observation,
        )
    assert captured.value.details["diagnosis_code"] == "NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE"
    assert runner.commands[-1][:4] == ("docker", "container", "rm", "--force")


def test_timed_out_probe_cleans_only_its_exact_container(tmp_path: Path) -> None:
    script = tmp_path / "worker_probe.py"
    script.write_text("", encoding="utf-8")
    image = resolved_image()

    class TimeoutRunner(QueueRunner):
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: float = 30.0,
            cwd: Path | None = None,
            env: Mapping[str, str] | None = None,
        ) -> CommandResult:
            command = tuple(args)
            if len(self.commands) == 2:
                self.commands.append(command)
                raise InfrastructureError("command timed out")
            return super().run(
                command,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )

    runner = TimeoutRunner(
        [
            result(stdout="pulled"),
            result(stdout=inspect_json(image.manifest_digest, image.config_digest)),
            result(),
        ]
    )
    with pytest.raises(InfrastructureError, match="timed out"):
        DockerWorkerProbe(runner, script_path=script).run(
            image,
            "GPU-11111111-1111-1111-1111-111111111111",
        )
    run_command = runner.commands[2]
    cleanup = runner.commands[3]
    name = run_command[run_command.index("--name") + 1]
    assert cleanup == ("docker", "container", "rm", "--force", name)


def test_probe_rejects_invalid_json_without_retry(tmp_path: Path) -> None:
    script = tmp_path / "worker_probe.py"
    script.write_text("", encoding="utf-8")
    image = resolved_image()
    runner = QueueRunner(
        [
            result(stdout="pulled"),
            result(stdout=inspect_json(image.manifest_digest, image.config_digest)),
            result(stdout="not-json"),
        ]
    )
    with pytest.raises(InfrastructureError, match="invalid document"):
        DockerWorkerProbe(runner, script_path=script).run(
            image,
            "GPU-11111111-1111-1111-1111-111111111111",
        )
