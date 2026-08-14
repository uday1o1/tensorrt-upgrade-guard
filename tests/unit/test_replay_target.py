"""Direct replay-target observation tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.errors import InfrastructureError, InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.reproduce.run import observe_replay_target

GPU_ONE = "GPU-11111111-1111-1111-1111-111111111111"
GPU_TWO = "GPU-22222222-2222-2222-2222-222222222222"


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


def test_observes_exact_selected_gpu_without_operator_entered_hardware_facts() -> None:
    runner = QueueRunner(
        [
            _result(stdout="linux/x86_64\n"),
            _result(
                stdout=(f"{GPU_ONE}, 8.9, 46068, 610.57.04\n{GPU_TWO}, 12.0, 97887, 610.57.04\n")
            ),
        ]
    )
    target = observe_replay_target(GPU_TWO, runner=runner)
    assert target.gpu_uuid == GPU_TWO
    assert target.compute_capability == "12.0"
    assert target.vram_mib == 97887
    assert target.driver_version == "610.57.04"
    assert runner.commands[1] == (
        "nvidia-smi",
        "--query-gpu=uuid,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )


def test_replay_target_requires_explicit_uuid_when_multiple_gpus_are_visible() -> None:
    runner = QueueRunner(
        [
            _result(stdout="linux/amd64\n"),
            _result(stdout=f"{GPU_ONE}, 8.9, 46068, 610.0\n{GPU_TWO}, 12.0, 97887, 610.0\n"),
        ]
    )
    with pytest.raises(InvalidInputError, match="multiple replay GPUs"):
        observe_replay_target(None, runner=runner)


@pytest.mark.parametrize(
    ("platform", "gpu_output", "error", "message"),
    [
        ("darwin/arm64\n", "", UnsupportedEnvironmentError, "linux/amd64"),
        ("linux/amd64\n", "malformed\n", InfrastructureError, "malformed"),
        ("linux/amd64\n", f"{GPU_ONE}, 8.9, 1.5, 610.0\n", InfrastructureError, "VRAM"),
    ],
)
def test_replay_target_fails_closed_on_unsupported_or_malformed_observations(
    platform: str,
    gpu_output: str,
    error: type[Exception],
    message: str,
) -> None:
    results = [_result(stdout=platform)]
    if platform.startswith("linux/"):
        results.append(_result(stdout=gpu_output))
    with pytest.raises(error, match=message):
        observe_replay_target(GPU_ONE, runner=QueueRunner(results))
