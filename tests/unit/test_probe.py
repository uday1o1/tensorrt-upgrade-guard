"""Exact Docker worker probe command tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.factories import digest, resolved_image, worker_probe
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.errors import InfrastructureError
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
