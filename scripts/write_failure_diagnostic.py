"""Write a redacted, local-only qualification failure diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = "upgradeguard.dev/failure-diagnostic/v1"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GPU_PATTERN = re.compile(r"^GPU-[A-Fa-f0-9-]+$")
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


def classify_failure(
    exit_code: int,
    requested: Literal["auto", "enospc", "infrastructure_invalid", "failed"] = "auto",
) -> tuple[str, str]:
    """Return a stable diagnostic classification and result status."""

    if requested == "enospc":
        return "ENOSPC", "infrastructure_invalid"
    if requested == "infrastructure_invalid" or (requested == "auto" and exit_code == 4):
        return "INFRASTRUCTURE_INVALID", "infrastructure_invalid"
    return "QUALIFICATION_FAILED", "failed"


def _safe_state(path: Path) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError("state must be an existing absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("state path cannot traverse symlinks")
    return resolved


def _safe_regular(path: Path, state: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        return path.resolve(strict=True).is_relative_to(state)
    except OSError:
        return False


def marker_inventory(state: Path) -> tuple[list[str], list[str], int]:
    """Classify marker filenames without following or exposing unsafe entries."""

    root = state / "done"
    if not root.exists():
        return [], [], 0
    if root.is_symlink() or not root.is_dir():
        raise ValueError("marker root must be a non-symlink directory")
    valid: list[str] = []
    invalid: list[str] = []
    unsafe = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not NAME_PATTERN.fullmatch(path.stem) or path.suffix != ".json":
            unsafe += 1
            continue
        if not _safe_regular(path, state):
            invalid.append(path.name)
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid.append(path.name)
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("schema_version"), str)
            and value.get("step") == path.stem
        ):
            valid.append(path.name)
        else:
            invalid.append(path.name)
    return valid, invalid, unsafe


def log_inventory(state: Path) -> tuple[list[dict[str, object]], int]:
    """Return local log pointers and sizes without reading log contents."""

    root = state / "logs"
    if not root.exists():
        return [], 0
    if root.is_symlink() or not root.is_dir():
        raise ValueError("log root must be a non-symlink directory")
    logs: list[dict[str, object]] = []
    unsafe = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not NAME_PATTERN.fullmatch(path.stem) or path.suffix != ".log":
            unsafe += 1
            continue
        if not _safe_regular(path, state):
            unsafe += 1
            continue
        logs.append({"path": path.relative_to(state).as_posix(), "bytes": path.stat().st_size})
    return logs, unsafe


def _resume_command(mode: str) -> list[str]:
    command = ["bash", "scripts/run_cuda_pm_qualification.sh"]
    if mode == "smoke":
        return ["env", "UG_SMOKE_ONLY=1", *command]
    if mode == "sanitizer":
        return ["env", "UG_SANITIZER_ONLY=1", *command]
    return command


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_diagnostic(
    *,
    state: Path,
    step: str,
    exit_code: int,
    requested_classification: Literal[
        "auto", "enospc", "infrastructure_invalid", "failed"
    ] = "auto",
    source: str | None = None,
    mode: str = "full",
    gpu: str | None = None,
) -> Path:
    """Inventory safe local pointers and publish one atomic diagnostic."""

    state = _safe_state(state)
    if not NAME_PATTERN.fullmatch(step):
        raise ValueError("current step is invalid")
    if source is not None and not SOURCE_PATTERN.fullmatch(source):
        raise ValueError("source must be a full lowercase Git commit")
    if mode not in {"full", "smoke", "sanitizer"}:
        raise ValueError("mode is invalid")
    if gpu is not None and not GPU_PATTERN.fullmatch(gpu):
        raise ValueError("GPU UUID is invalid")
    if exit_code < 0 or exit_code > 255:
        raise ValueError("exit code must be between 0 and 255")
    valid, invalid, unsafe_markers = marker_inventory(state)
    logs, unsafe_logs = log_inventory(state)
    classification, status = classify_failure(exit_code, requested_classification)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "occurred_at": datetime.now(UTC).isoformat(),
        "status": status,
        "classification": classification,
        "current_step": step,
        "exit_code": exit_code,
        "mode": mode,
        "markers": {
            "valid": valid,
            "invalid": invalid,
            "unsafe_entry_count": unsafe_markers,
        },
        "logs": logs,
        "unsafe_log_entry_count": unsafe_logs,
        "resume_command": _resume_command(mode),
    }
    if source is not None:
        payload["source_git_commit"] = source
    if gpu is not None:
        payload["gpu_uuid"] = gpu
    root = state / "diagnostics"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("diagnostic root must be a directory")
    root.mkdir(mode=0o700, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / f"failure-{timestamp}-{os.getpid()}.json"
    _write_atomic(output, payload)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument(
        "--classification",
        choices=("auto", "enospc", "infrastructure_invalid", "failed"),
        default="auto",
    )
    parser.add_argument("--source")
    parser.add_argument("--mode", choices=("full", "smoke", "sanitizer"), default="full")
    parser.add_argument("--gpu")
    arguments = parser.parse_args()
    try:
        output = write_diagnostic(
            state=arguments.state,
            step=arguments.step,
            exit_code=arguments.exit_code,
            requested_classification=arguments.classification,
            source=arguments.source,
            mode=arguments.mode,
            gpu=arguments.gpu,
        )
    except (OSError, ValueError):
        return 4
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
