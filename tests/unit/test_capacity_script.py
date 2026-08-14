"""Capacity preflight parser, CLI, and classification tests."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.check_capacity import (
    _run_df,
    classify_exception,
    docker_capacity,
    parse_posix_df,
    workspace_capacity,
)

PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT / "scripts" / "check_capacity.py"
BLOCKS = (
    "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/test 100 10 90 10% /docker\n"
)
INODES = "Filesystem Inodes IUsed IFree IUse% Mounted on\n/dev/test 1000 100 900 10% /docker\n"


def _run(tmp_path: Path, blocks: str, inodes: str, *, bytes_required: int = 1) -> tuple[int, dict]:
    blocks_path = tmp_path / "blocks.txt"
    inodes_path = tmp_path / "inodes.txt"
    output = tmp_path / "capacity.json"
    blocks_path.write_text(blocks, encoding="utf-8")
    inodes_path.write_text(inodes, encoding="utf-8")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(tmp_path),
            "--output",
            str(output),
            "--workspace-min-bytes",
            "0",
            "--workspace-min-inodes",
            "0",
            "--docker-min-bytes",
            str(bytes_required),
            "--docker-min-inodes",
            "1",
            "--docker-blocks-df",
            str(blocks_path),
            "--docker-inodes-df",
            str(inodes_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode, json.loads(output.read_text(encoding="utf-8"))


def test_capacity_cli_passes_and_atomically_publishes(tmp_path: Path) -> None:
    status, payload = _run(tmp_path, BLOCKS, INODES)
    assert status == 0
    assert payload["status"] == "passed"
    assert payload["docker_volume_storage"]["available_bytes"] == 90 * 1024
    assert not list(tmp_path.glob(".capacity.json.*"))


@pytest.mark.parametrize(
    ("required_bytes", "inodes", "field"),
    [
        (100 * 1024, INODES, "available_bytes"),
        (1, INODES.replace("900", "0"), "available_inodes"),
    ],
)
def test_capacity_cli_fails_for_bytes_or_inodes(
    tmp_path: Path, required_bytes: int, inodes: str, field: str
) -> None:
    status, payload = _run(tmp_path, BLOCKS, inodes, bytes_required=required_bytes)
    assert status == 4
    assert payload["status"] == "infrastructure_invalid"
    assert payload["classification"] == "insufficient_capacity"
    assert payload["docker_volume_storage"][field] >= 0


def test_workspace_statvfs_observation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "scripts.check_capacity.os.statvfs",
        lambda _: SimpleNamespace(f_bavail=20, f_frsize=4096, f_favail=12),
    )
    observed = workspace_capacity(tmp_path, required_bytes=80_000, required_inodes=10)
    assert observed.available_bytes == 81_920
    assert observed.sufficient


def test_malformed_and_inaccessible_capacity_fail_closed(tmp_path: Path) -> None:
    status, payload = _run(tmp_path, "not df\n", INODES)
    assert status == 4
    assert payload["classification"] == "infrastructure_invalid"
    output = tmp_path / "missing.json"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--docker-blocks-df",
            str(tmp_path / "absent-blocks"),
            "--docker-inodes-df",
            str(tmp_path / "absent-inodes"),
        ],
        check=False,
        timeout=10,
    )
    assert result.returncode == 4
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "infrastructure_invalid"


def test_df_parser_and_enospc_classification() -> None:
    assert parse_posix_df(BLOCKS, kind="bytes") == 90 * 1024
    assert parse_posix_df(INODES, kind="inodes") == 900
    with pytest.raises(ValueError, match="schema"):
        parse_posix_df("header\ndata\n", kind="bytes")
    assert classify_exception(OSError(errno.ENOSPC, "redacted")) == "enospc"
    assert classify_exception(RuntimeError("No space left on device")) == "enospc"
    assert classify_exception(RuntimeError("other")) == "infrastructure_invalid"
    assert docker_capacity(BLOCKS, INODES, 1, 1).sufficient


def test_df_runner_uses_an_argument_array_without_a_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(arguments: tuple[str, ...], **keywords: object) -> SimpleNamespace:
        observed["arguments"] = arguments
        observed.update(keywords)
        return SimpleNamespace(returncode=0, stdout=BLOCKS, stderr="")

    monkeypatch.setattr("scripts.check_capacity.subprocess.run", fake_run)
    assert _run_df("/usr/bin/df", tmp_path, "-Pk") == BLOCKS
    assert observed["arguments"] == ("/usr/bin/df", "-Pk", str(tmp_path))
    assert observed["shell"] is False


def test_df_runner_timeout_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def timed_out(*_: object, **__: object) -> None:
        raise subprocess.TimeoutExpired("df", 30)

    monkeypatch.setattr("scripts.check_capacity.subprocess.run", timed_out)
    with pytest.raises(RuntimeError, match="timed out"):
        _run_df("df", tmp_path, "-Pi")
