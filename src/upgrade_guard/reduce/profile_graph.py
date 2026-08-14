"""Bounded in-worker Polygraphy graph transformation and reduction commands."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from upgrade_guard.reduce.profile_check import INFRASTRUCTURE_INVALID


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation", choices=("freeze", "fold", "bisect", "linear"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--control-tokens", type=Path, required=True)
    parser.add_argument("--control-mask", type=Path, required=True)
    parser.add_argument("--failure-tokens", type=Path, required=True)
    parser.add_argument("--failure-mask", type=Path, required=True)
    parser.add_argument("--workspace-bytes", type=int, required=True)
    parser.add_argument("--optimization-level", type=int, required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--maximum-seconds", type=float, required=True)
    arguments = parser.parse_args()
    if arguments.maximum_seconds <= 0:
        return INFRASTRUCTURE_INVALID
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    observations = output.parent / f"{output.stem}-checks.jsonl"
    if output.exists() or output.is_symlink() or observations.exists():
        return INFRASTRUCTURE_INVALID
    if arguments.operation in {"freeze", "fold"}:
        command = _surgeon_command(arguments)
    else:
        command = _reduction_command(arguments, observations)
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=arguments.maximum_seconds,
            shell=False,
            cwd=output.parent,
        )
    except (OSError, subprocess.TimeoutExpired):
        return INFRASTRUCTURE_INVALID
    (output.parent / f"{output.stem}-command.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/polygraphy-command/v1",
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[-16384:],
                "stderr": result.stderr[-16384:],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0 or not output.is_file() or output.is_symlink():
        return INFRASTRUCTURE_INVALID
    if arguments.operation in {"bisect", "linear"}:
        try:
            records = [
                json.loads(line)
                for line in observations.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError):
            return INFRASTRUCTURE_INVALID
        if not records or any(
            item.get("outcome") == "infrastructure_invalid"
            or item.get("predicate_signature_sha256") != arguments.signature
            or item.get("failure_code") != "PROFILE_REJECTED"
            for item in records
        ):
            return INFRASTRUCTURE_INVALID
        if not any(item.get("outcome") == "reproduced" for item in records):
            return INFRASTRUCTURE_INVALID
    return 0


def _surgeon_command(arguments: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "polygraphy",
        "surgeon",
        "sanitize",
        str(arguments.model),
        "--output",
        str(arguments.output),
        "--fold-constants",
    ]
    if arguments.operation == "freeze":
        import numpy as np

        tokens = np.load(arguments.failure_tokens, allow_pickle=False)
        mask = np.load(arguments.failure_mask, allow_pickle=False)
        tokens_shape = ",".join(str(dimension) for dimension in tokens.shape)
        mask_shape = ",".join(str(dimension) for dimension in mask.shape)
        command.extend(
            (
                "--override-input-shapes",
                f"tokens:[{tokens_shape}]",
                f"mask:[{mask_shape}]",
            )
        )
    return tuple(command)


def _reduction_command(
    arguments: argparse.Namespace,
    observations: Path,
) -> tuple[str, ...]:
    check = (
        "python3",
        "-m",
        "upgrade_guard.reduce.profile_check",
        "--profile",
        str(arguments.profile),
        "--control-tokens",
        str(arguments.control_tokens),
        "--control-mask",
        str(arguments.control_mask),
        "--failure-tokens",
        str(arguments.failure_tokens),
        "--failure-mask",
        str(arguments.failure_mask),
        "--workspace-bytes",
        str(arguments.workspace_bytes),
        "--optimization-level",
        str(arguments.optimization_level),
        "--signature",
        arguments.signature,
        "--observations",
        str(observations),
    )
    return (
        "polygraphy",
        "debug",
        "reduce",
        str(arguments.model),
        "--output",
        str(arguments.output),
        f"--mode={arguments.operation}",
        "--fail-code",
        "86",
        "--check",
        *check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
