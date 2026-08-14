"""Run one qualification command while holding an inherited POSIX file lock."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import NoReturn

SCHEMA_VERSION = "upgradeguard.dev/qualification-lock/v1"
INFRASTRUCTURE_EXIT = 4
INVALID_INPUT_EXIT = 2
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _emit_error(error_code: str, message: str, *, holder: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "invalid_input"
        if error_code == "INVALID_LOCK_LAUNCH"
        else "infrastructure_invalid",
        "error_code": error_code,
        "message": message,
    }
    if holder:
        payload["holder"] = holder
    print(json.dumps(payload, allow_nan=False, sort_keys=True), file=sys.stderr)


def _safe_lock_path(value: str) -> Path:
    if not value or "\x00" in value:
        raise ValueError("lock path must be nonempty and NUL-free")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError("lock path must be an explicit absolute file path")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("lock path parent must be an existing non-symlink directory")
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent:
        raise ValueError("lock path parent cannot traverse symlinks")
    if path.exists() and path.is_symlink():
        raise ValueError("lock path cannot be a symlink")
    return path


def _safe_command(command: list[str]) -> list[str]:
    if not command or command[0] == "--":
        command = command[1:]
    if not command or any(not item or "\x00" in item for item in command):
        raise ValueError("command must be a nonempty NUL-free argument array")
    return command


def _holder_metadata(descriptor: int) -> dict[str, object] | None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = os.read(descriptor, 4096)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    pid = value.get("pid")
    source = value.get("source_git_commit")
    holder: dict[str, object] = {}
    if isinstance(pid, int) and pid > 0:
        holder["pid"] = pid
    if isinstance(source, str) and SOURCE_PATTERN.fullmatch(source):
        holder["source_git_commit"] = source
    return holder or None


def _write_metadata(descriptor: int, source: str) -> None:
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "source_git_commit": source,
        },
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    written = 0
    while written < len(payload):
        written += os.write(descriptor, payload[written:])
    os.fsync(descriptor)


def launch(lock_path: Path, source: str, command: list[str]) -> NoReturn:
    """Acquire the exact lock and replace this process with the requested command."""

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("lock path must identify a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _emit_error(
                "QUALIFICATION_LOCK_CONTENDED",
                "another qualification process holds the qualification lock",
                holder=_holder_metadata(descriptor),
            )
            raise SystemExit(INFRASTRUCTURE_EXIT) from None
        _write_metadata(descriptor, source)
        os.set_inheritable(descriptor, True)
        environment = os.environ.copy()
        environment["UG_QUALIFICATION_LOCK_HELD"] = "1"
        environment["UG_QUALIFICATION_LOCK_FD"] = str(descriptor)
        os.execvpe(command[0], command, environment)  # noqa: S606
    finally:
        os.close(descriptor)
    raise AssertionError("os.execvpe unexpectedly returned")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    try:
        lock_path = _safe_lock_path(arguments.lock)
        if not SOURCE_PATTERN.fullmatch(arguments.source):
            raise ValueError("source must be a full lowercase Git commit")
        command = _safe_command(arguments.command)
        launch(lock_path, arguments.source, command)
    except ValueError as error:
        _emit_error("INVALID_LOCK_LAUNCH", str(error))
        return INVALID_INPUT_EXIT
    except OSError as error:
        _emit_error("LOCK_LAUNCH_INFRASTRUCTURE_INVALID", type(error).__name__)
        return INFRASTRUCTURE_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
