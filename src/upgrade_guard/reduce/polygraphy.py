"""Exact Polygraphy graph-reduction command adapter."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from upgrade_guard.containers.commands import CommandRunner, Runner, command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.errors import InvalidInputError


@dataclass(frozen=True, slots=True)
class PolygraphyReductionResult:
    """Bisect and linear graph identities with exact command hashes."""

    bisect_model: Path
    final_model: Path
    bisect_model_sha256: str
    final_model_sha256: str
    command_sha256: tuple[str, str]


def reduction_commands(
    *,
    model: Path,
    output: Path,
    predicate_command: tuple[str, ...],
    failure_returncode: int = 86,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Use bisect first, followed by linear reduction on the smaller graph."""

    bisect_output = output.with_name(f"{output.stem}.bisect{output.suffix}")
    bisect = (
        "polygraphy",
        "debug",
        "reduce",
        str(model),
        "--output",
        str(bisect_output),
        "--mode=bisect",
        "--fail-code",
        str(failure_returncode),
        "--check",
        *predicate_command,
    )
    linear = (
        "polygraphy",
        "debug",
        "reduce",
        str(bisect_output),
        "--output",
        str(output),
        "--mode=linear",
        "--fail-code",
        str(failure_returncode),
        "--check",
        *predicate_command,
    )
    return bisect, linear


def run_polygraphy_reduction(
    *,
    model: Path,
    output: Path,
    predicate_command: tuple[str, ...],
    maximum_seconds: float,
    runner: Runner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> PolygraphyReductionResult:
    """Execute bounded bisect then linear reduction and retain both graph identities."""

    if maximum_seconds <= 0:
        raise InvalidInputError("Polygraphy reduction time budget must be positive")
    if not model.is_file() or model.is_symlink():
        raise InvalidInputError("Polygraphy reduction model must be a regular file")
    if output.exists() or output.is_symlink():
        raise InvalidInputError("refusing to overwrite Polygraphy reduction output")
    commands = reduction_commands(
        model=model,
        output=output,
        predicate_command=predicate_command,
    )
    bisect_output = output.with_name(f"{output.stem}.bisect{output.suffix}")
    command_runner = runner or CommandRunner()
    started = clock()
    for index, (command, expected) in enumerate(
        zip(commands, (bisect_output, output), strict=True)
    ):
        remaining = maximum_seconds - (clock() - started)
        if remaining <= 0:
            raise InvalidInputError("Polygraphy reduction exhausted its wall-clock budget")
        result = command_runner.run(command, timeout_seconds=remaining)
        if result.returncode != 0:
            raise InvalidInputError(
                "Polygraphy graph reduction failed",
                details={
                    "stage": "bisect" if index == 0 else "linear",
                    "returncode": result.returncode,
                    "command_sha256": command_sha256(command),
                },
            )
        if not expected.is_file() or expected.is_symlink() or expected.stat().st_size == 0:
            raise InvalidInputError("Polygraphy did not produce the expected reduced model")
    return PolygraphyReductionResult(
        bisect_model=bisect_output,
        final_model=output,
        bisect_model_sha256=sha256_file(bisect_output),
        final_model_sha256=sha256_file(output),
        command_sha256=(command_sha256(commands[0]), command_sha256(commands[1])),
    )
