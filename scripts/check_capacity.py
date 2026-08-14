"""Fail-closed workspace and Docker-volume capacity preflight."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = "upgradeguard.dev/capacity-check/v1"
INFRASTRUCTURE_EXIT = 4
INVALID_INPUT_EXIT = 2
DEFAULT_WORKSPACE_BYTES = 20 * 1024**3
DEFAULT_DOCKER_BYTES = 20 * 1024**3
DEFAULT_FREE_INODES = 100_000


@dataclass(frozen=True)
class CapacityObservation:
    available_bytes: int
    available_inodes: int
    required_bytes: int
    required_inodes: int
    sufficient: bool


def classify_exception(error: BaseException) -> Literal["enospc", "infrastructure_invalid"]:
    """Classify space exhaustion without serializing exception text."""

    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "enospc"
    if "no space left on device" in str(error).lower():
        return "enospc"
    return "infrastructure_invalid"


def workspace_capacity(
    path: Path, required_bytes: int, required_inodes: int
) -> CapacityObservation:
    """Read filesystem capacity for an existing workspace path."""

    resolved = path.resolve(strict=True)
    values = os.statvfs(resolved)
    available_bytes = int(values.f_bavail) * int(values.f_frsize)
    available_inodes = int(values.f_favail)
    if available_bytes < 0 or available_inodes < 0:
        raise RuntimeError("workspace capacity observation is unavailable")
    return CapacityObservation(
        available_bytes=available_bytes,
        available_inodes=available_inodes,
        required_bytes=required_bytes,
        required_inodes=required_inodes,
        sufficient=available_bytes >= required_bytes and available_inodes >= required_inodes,
    )


def parse_posix_df(text: str, *, kind: Literal["bytes", "inodes"]) -> int:
    """Parse the available field from one POSIX `df -P` record."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("POSIX df output must contain one header and one data row")
    header = tuple(field.lower() for field in lines[0].split())
    fields = lines[1].split()
    reports_blocks = any("block" in field for field in header)
    reports_inodes = any("inode" in field for field in header)
    reports_available = any(field in {"available", "avail", "ifree", "free"} for field in header)
    expected_kind = reports_blocks if kind == "bytes" else reports_inodes
    if not expected_kind or not reports_available or len(fields) < 6:
        raise ValueError("POSIX df output has an unsupported schema")
    try:
        available = int(fields[-3])
    except ValueError as error:
        raise ValueError("POSIX df available capacity is not an integer") from error
    if available < 0 or not fields[-2].endswith("%"):
        raise ValueError("POSIX df capacity values are malformed")
    return available * 1024 if kind == "bytes" else available


def docker_capacity(
    blocks_text: str,
    inodes_text: str,
    required_bytes: int,
    required_inodes: int,
) -> CapacityObservation:
    available_bytes = parse_posix_df(blocks_text, kind="bytes")
    available_inodes = parse_posix_df(inodes_text, kind="inodes")
    return CapacityObservation(
        available_bytes=available_bytes,
        available_inodes=available_inodes,
        required_bytes=required_bytes,
        required_inodes=required_inodes,
        sufficient=available_bytes >= required_bytes and available_inodes >= required_inodes,
    )


def _read_evidence(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _run_df(executable: str, path: Path, option: str) -> str:
    if not executable or "\x00" in executable:
        raise ValueError("df executable must be nonempty and NUL-free")
    try:
        result = subprocess.run(  # noqa: S603
            (executable, option, str(path)),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Docker-volume backing storage observation timed out") from error
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Docker-volume backing storage could not be observed")
    return result.stdout


def _docker_evidence(arguments: argparse.Namespace) -> tuple[str, str]:
    files = (arguments.docker_blocks_df, arguments.docker_inodes_df)
    if any(files):
        if not all(files) or files == ("-", "-"):
            raise ValueError("both Docker df files are required and only one may use stdin")
        return _read_evidence(files[0]), _read_evidence(files[1])
    if arguments.docker_path is None:
        raise ValueError("Docker storage requires captured df files or --docker-path")
    docker_path = arguments.docker_path.resolve(strict=True)
    return (
        _run_df(arguments.df_executable, docker_path, "-Pk"),
        _run_df(arguments.df_executable, docker_path, "-Pi"),
    )


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _threshold(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("capacity thresholds must be nonnegative")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-min-bytes", type=_threshold, default=DEFAULT_WORKSPACE_BYTES)
    parser.add_argument("--workspace-min-inodes", type=_threshold, default=DEFAULT_FREE_INODES)
    parser.add_argument("--docker-min-bytes", type=_threshold, default=DEFAULT_DOCKER_BYTES)
    parser.add_argument("--docker-min-inodes", type=_threshold, default=DEFAULT_FREE_INODES)
    parser.add_argument("--docker-blocks-df")
    parser.add_argument("--docker-inodes-df")
    parser.add_argument("--docker-path", type=Path)
    parser.add_argument("--df-executable", default="df")
    arguments = parser.parse_args()
    try:
        workspace = workspace_capacity(
            arguments.workspace,
            arguments.workspace_min_bytes,
            arguments.workspace_min_inodes,
        )
        blocks, inodes = _docker_evidence(arguments)
        docker = docker_capacity(
            blocks,
            inodes,
            arguments.docker_min_bytes,
            arguments.docker_min_inodes,
        )
        passed = workspace.sufficient and docker.sufficient
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if passed else "infrastructure_invalid",
            "classification": "capacity_sufficient" if passed else "insufficient_capacity",
            "workspace": asdict(workspace),
            "docker_volume_storage": asdict(docker),
        }
    except (OSError, RuntimeError, ValueError) as error:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "infrastructure_invalid",
            "classification": classify_exception(error),
            "workspace": None,
            "docker_volume_storage": None,
        }
        passed = False
    try:
        _write_atomic(arguments.output, payload)
    except OSError:
        return INFRASTRUCTURE_EXIT
    json.dump(payload, sys.stdout, allow_nan=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if passed else INFRASTRUCTURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
