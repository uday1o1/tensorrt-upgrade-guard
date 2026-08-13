"""Exact Polygraphy graph-reduction command adapter."""

from __future__ import annotations

from pathlib import Path


def reduction_commands(
    *,
    model: Path,
    output: Path,
    predicate_command: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Use bisect first, followed by linear reduction on the smaller graph."""

    shared = (
        "polygraphy",
        "debug",
        "reduce",
        str(model),
        "--output",
        str(output),
        "--check",
        *predicate_command,
    )
    return ((*shared, "--mode=bisect"), (*shared, "--mode=linear"))
