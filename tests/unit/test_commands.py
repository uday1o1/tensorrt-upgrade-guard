"""Safe subprocess boundary tests."""

from __future__ import annotations

import sys

import pytest

from upgrade_guard.containers.commands import CommandRunner, command_sha256
from upgrade_guard.errors import InfrastructureError, InvalidInputError


def test_runner_preserves_metacharacters_as_one_argument() -> None:
    marker = "$(touch should-not-exist)"
    result = CommandRunner().run((sys.executable, "-c", "import sys; print(sys.argv[1])", marker))
    assert result.returncode == 0
    assert result.stdout.strip() == marker


def test_runner_reports_missing_executable_without_shell() -> None:
    result = CommandRunner().run(("upgrade-guard-definitely-missing",))
    assert result.returncode == 127
    assert "command not found" in result.stderr


def test_runner_rejects_string_or_nul_arguments() -> None:
    with pytest.raises(InvalidInputError):
        CommandRunner().run("echo hello")
    with pytest.raises(InvalidInputError):
        CommandRunner().run(("echo", "bad\x00argument"))


def test_runner_turns_timeout_into_infrastructure_error() -> None:
    with pytest.raises(InfrastructureError, match="timed out"):
        CommandRunner().run(
            (sys.executable, "-c", "import time; time.sleep(1)"),
            timeout_seconds=0.01,
        )


def test_command_hash_uses_argument_boundaries() -> None:
    assert command_sha256(("ab", "c")) != command_sha256(("a", "bc"))
    assert command_sha256(("ab", "c")) == command_sha256(["ab", "c"])
