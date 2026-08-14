"""Process-level tests for the inherited qualification lock launcher."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.run_locked import _safe_lock_path

PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT / "scripts" / "run_locked.py"
SOURCE = "a" * 40


def _launcher(lock: Path, command: list[str]) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--lock",
        str(lock),
        "--source",
        SOURCE,
        "--",
        *command,
    ]


def _wait_for(path: Path) -> None:
    for _ in range(200):
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_lock_is_inherited_through_exec(tmp_path: Path) -> None:
    lock = tmp_path / "qualification.lock"
    observed = tmp_path / "observed.json"
    code = (
        "import json,os,sys; from pathlib import Path; "
        "fd=int(os.environ['UG_QUALIFICATION_LOCK_FD']); os.fstat(fd); "
        "Path(sys.argv[1]).write_text(json.dumps({"
        "'held':os.environ.get('UG_QUALIFICATION_LOCK_HELD'),'inheritable':os.get_inheritable(fd)}))"
    )
    result = subprocess.run(  # noqa: S603
        _launcher(lock, [sys.executable, "-c", code, str(observed)]),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert json.loads(observed.read_text(encoding="utf-8")) == {
        "held": "1",
        "inheritable": True,
    }
    metadata = json.loads(lock.read_text(encoding="utf-8"))
    assert metadata["source_git_commit"] == SOURCE
    assert isinstance(metadata["pid"], int)


def test_second_process_fails_closed_without_mutating_lock(tmp_path: Path) -> None:
    lock = tmp_path / "qualification.lock"
    ready = tmp_path / "ready"
    holder_code = (
        "import signal,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text('ready'); signal.pause()"
    )
    holder = subprocess.Popen(  # noqa: S603
        _launcher(lock, [sys.executable, "-c", holder_code, str(ready)]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(ready)
        before = lock.read_bytes()
        contender = subprocess.run(  # noqa: S603
            _launcher(lock, [sys.executable, "-c", "raise SystemExit(0)"]),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert contender.returncode == 4
        error = json.loads(contender.stderr)
        assert error["schema_version"] == "upgradeguard.dev/qualification-lock/v1"
        assert error["error_code"] == "QUALIFICATION_LOCK_CONTENDED"
        assert error["holder"]["source_git_commit"] == SOURCE
        assert lock.read_bytes() == before
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_kernel_releases_lock_after_child_crash(tmp_path: Path) -> None:
    lock = tmp_path / "qualification.lock"
    ready = tmp_path / "crash-ready"
    crash_code = (
        "import os,signal,sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text('ready'); os.kill(os.getpid(), signal.SIGKILL)"
    )
    crashed = subprocess.Popen(  # noqa: S603
        _launcher(lock, [sys.executable, "-c", crash_code, str(ready)]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for(ready)
    assert crashed.wait(timeout=10) == -signal.SIGKILL
    successor = subprocess.run(  # noqa: S603
        _launcher(lock, [sys.executable, "-c", "raise SystemExit(0)"]),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert successor.returncode == 0


def test_lock_path_rejects_relative_and_symlink_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        _safe_lock_path("relative.lock")
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _safe_lock_path(str(linked / "qualification.lock"))
