"""Validate one isolated worker's terminal failure before host promotion."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.build import WorkerBuildResult
from upgrade_guard.contracts.results import WorkerCorrectnessResult


def validate(path: Path, *, kind: str) -> None:
    """Require a strict failed result with an exact retained command identity."""

    model = WorkerBuildResult if kind == "build" else WorkerCorrectnessResult
    try:
        value = model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise RuntimeError("worker failure result is missing or malformed") from error
    if (
        value.status != "failed"
        or value.failure_code is None
        or value.command_sha256 != command_sha256(value.command)
    ):
        raise RuntimeError("worker result is not a strict typed terminal failure")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("build", "correctness"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    try:
        validate(arguments.path, kind=arguments.kind)
    except RuntimeError as error:
        print(str(error))
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
