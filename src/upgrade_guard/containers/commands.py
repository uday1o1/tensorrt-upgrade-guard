"""Narrow, injectable subprocess boundary."""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from upgrade_guard.contracts.base import canonical_json_bytes
from upgrade_guard.errors import InfrastructureError, InvalidInputError


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured process result without shell interpretation."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


class Runner(Protocol):
    """Injectable command interface for host and container operations."""

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult: ...


class CommandRunner:
    """Execute argument arrays with bounded capture and no shell."""

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        normalized = _validate_args(args)
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603
                normalized,
                cwd=cwd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
            )
        except FileNotFoundError:
            return CommandResult(normalized, 127, "", f"command not found: {normalized[0]}", 0.0)
        except subprocess.TimeoutExpired as error:
            raise InfrastructureError(
                f"command timed out after {timeout_seconds:g} seconds",
                details={"command_sha256": command_sha256(normalized)},
            ) from error
        duration = time.monotonic() - started
        return CommandResult(
            normalized,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            duration,
        )


def _validate_args(args: Sequence[str]) -> tuple[str, ...]:
    if isinstance(args, str | bytes) or not args:
        raise InvalidInputError("commands must be nonempty argument arrays")
    normalized = tuple(args)
    if any(not isinstance(argument, str) or "\x00" in argument for argument in normalized):
        raise InvalidInputError("command arguments must be NUL-free strings")
    return normalized


def command_sha256(args: Sequence[str]) -> str:
    """Hash an exact argument vector for manifests without shell rendering."""

    normalized = _validate_args(args)
    digest = hashlib.sha256(canonical_json_bytes(list(normalized))).hexdigest()
    return f"sha256:{digest}"
