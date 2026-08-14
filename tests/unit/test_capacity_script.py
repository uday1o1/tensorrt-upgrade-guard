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
    FilesystemIdentity,
    _run_df,
    capacity_decision,
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
BUSYBOX_INODES = (
    "Filesystem Inodes Used Available Use% Mounted on\n/dev/test 1000 100 900 10% /docker\n"
)


def _run(
    tmp_path: Path,
    blocks: str,
    inodes: str,
    *,
    bytes_required: int = 1,
    inodes_required: int = 1,
    workspace_bytes_required: int = 0,
    workspace_inodes_required: int = 0,
    filesystem_id: str | None = None,
) -> tuple[int, dict, str]:
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
            str(workspace_bytes_required),
            "--workspace-min-inodes",
            str(workspace_inodes_required),
            "--docker-min-bytes",
            str(bytes_required),
            "--docker-min-inodes",
            str(inodes_required),
            "--docker-blocks-df",
            str(blocks_path),
            "--docker-inodes-df",
            str(inodes_path),
            "--docker-filesystem-id",
            str(tmp_path.stat().st_dev) if filesystem_id is None else filesystem_id,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode, json.loads(output.read_text(encoding="utf-8")), result.stdout


def test_capacity_cli_passes_and_atomically_publishes(tmp_path: Path) -> None:
    status, payload, rendered = _run(tmp_path, BLOCKS, BUSYBOX_INODES)
    assert status == 0
    assert payload["status"] == "passed"
    assert payload["docker_volume_storage"]["available_bytes"] == 90 * 1024
    assert payload["workspace"]["filesystem_identity"] == {
        "kind": "device_number",
        "value": str(tmp_path.stat().st_dev),
    }
    assert payload["filesystem_decision"]["mode"] == "shared"
    assert json.loads(rendered) == payload
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
    status, payload, rendered = _run(tmp_path, BLOCKS, inodes, bytes_required=required_bytes)
    assert status == 4
    assert payload["status"] == "infrastructure_invalid"
    assert payload["classification"] == "insufficient_capacity"
    assert payload["docker_volume_storage"][field] >= 0
    assert json.loads(rendered) == payload


def test_workspace_statvfs_observation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "scripts.check_capacity.os.statvfs",
        lambda _: SimpleNamespace(f_bavail=20, f_frsize=4096, f_favail=12),
    )
    observed = workspace_capacity(tmp_path, required_bytes=80_000, required_inodes=10)
    assert observed.available_bytes == 81_920
    assert observed.sufficient


@pytest.mark.parametrize(
    ("workspace_bytes", "workspace_inodes", "docker_bytes", "docker_inodes"),
    [
        (60 * 1024, 0, 40 * 1024, 1),
        (0, 500, 1, 500),
    ],
)
def test_shared_filesystem_requires_aggregate_budget(
    tmp_path: Path,
    workspace_bytes: int,
    workspace_inodes: int,
    docker_bytes: int,
    docker_inodes: int,
) -> None:
    status, payload, _ = _run(
        tmp_path,
        BLOCKS,
        INODES,
        bytes_required=docker_bytes,
        inodes_required=docker_inodes,
        workspace_bytes_required=workspace_bytes,
        workspace_inodes_required=workspace_inodes,
    )
    assert status == 4
    assert payload["classification"] == "insufficient_capacity"
    assert payload["filesystem_decision"]["mode"] == "shared"
    assert payload["filesystem_decision"]["sufficient"] is False


def test_different_filesystems_keep_independent_thresholds(tmp_path: Path) -> None:
    status, payload, _ = _run(
        tmp_path,
        BLOCKS,
        INODES,
        bytes_required=40 * 1024,
        workspace_bytes_required=60 * 1024,
        filesystem_id=str(tmp_path.stat().st_dev + 1),
    )
    assert status == 0
    assert payload["filesystem_decision"] == {
        "available_bytes": None,
        "available_inodes": None,
        "docker_volume_identity": {
            "kind": "device_number",
            "value": str(tmp_path.stat().st_dev + 1),
        },
        "mode": "independent",
        "required_bytes": None,
        "required_inodes": None,
        "sufficient": True,
        "workspace_identity": {
            "kind": "device_number",
            "value": str(tmp_path.stat().st_dev),
        },
    }


@pytest.mark.parametrize("filesystem_id", ["", "-1", "1.5", "device:1", "18446744073709551616"])
def test_malformed_docker_filesystem_identity_fails_closed(
    tmp_path: Path, filesystem_id: str
) -> None:
    status, payload, rendered = _run(
        tmp_path,
        BLOCKS,
        INODES,
        filesystem_id=filesystem_id,
    )
    assert status == 4
    assert payload["classification"] == "infrastructure_invalid"
    assert payload["filesystem_decision"] is None
    assert json.loads(rendered) == payload


def test_missing_docker_filesystem_identity_fails_closed(tmp_path: Path) -> None:
    blocks_path = tmp_path / "blocks.txt"
    inodes_path = tmp_path / "inodes.txt"
    output = tmp_path / "capacity.json"
    blocks_path.write_text(BLOCKS, encoding="utf-8")
    inodes_path.write_text(INODES, encoding="utf-8")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(tmp_path),
            "--output",
            str(output),
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
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 4
    assert payload["classification"] == "infrastructure_invalid"
    assert payload["filesystem_decision"] is None


def test_malformed_and_inaccessible_capacity_fail_closed(tmp_path: Path) -> None:
    status, payload, rendered = _run(tmp_path, "not df\n", INODES)
    assert status == 4
    assert payload["classification"] == "infrastructure_invalid"
    assert json.loads(rendered) == payload
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
    identity = FilesystemIdentity(kind="device_number", value="1")
    assert docker_capacity(BLOCKS, INODES, identity, 1, 1).sufficient


def test_capacity_decision_uses_conservative_shared_observation(tmp_path: Path) -> None:
    identity = FilesystemIdentity(kind="device_number", value="1")
    workspace = workspace_capacity(tmp_path, required_bytes=1, required_inodes=1)
    workspace = workspace.__class__(identity, 100 * 1024, 200, 40, 80, True)
    docker = docker_capacity(BLOCKS, INODES, identity, 40, 80)
    decision = capacity_decision(workspace, docker)
    assert decision.mode == "shared"
    assert decision.available_bytes == 90 * 1024
    assert decision.available_inodes == 200
    assert decision.required_bytes == 80
    assert decision.required_inodes == 160


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
