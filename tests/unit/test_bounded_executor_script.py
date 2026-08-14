"""Process-level tests for the Linux qualification timeout executor."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
EXECUTOR = PROJECT / "scripts" / "bounded_executor.sh"


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_timeout(directory: Path) -> None:
    _executable(
        directory / "timeout",
        """#!/usr/bin/env python3
import json
import os
import sys

arguments = sys.argv[1:]
with open(os.environ["FAKE_TIMEOUT_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
index = 0
while arguments[index].startswith("--"):
    index += 1
index += 1
os.execvp(arguments[index], arguments[index:])
""",
    )


def _environment(fake_bin: Path, timeout_log: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_TIMEOUT_LOG"] = str(timeout_log)
    return environment


def test_executor_uses_class_timeout_and_preserves_command_arguments(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout_log = tmp_path / "timeout.jsonl"
    result = tmp_path / "result.json"
    _fake_timeout(fake_bin)
    environment = _environment(fake_bin, timeout_log)
    environment.update(
        {
            "UG_TIMEOUT_GPU_SECONDS": "17",
            "UG_TIMEOUT_KILL_AFTER_SECONDS": "9",
            "RESULT_PATH": str(result),
        }
    )
    command = """
source "$1"
initialize_bounded_executor
literal='literal;$(not-executed)'
bounded_run gpu python3 -c \
  'import json,os,sys; p=os.environ["RESULT_PATH"]; open(p, "w").write(json.dumps(sys.argv[1:]))' \
  "argument with spaces" "$literal"
"""
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", command, "bash", str(EXECUTOR)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    invocation = json.loads(timeout_log.read_text(encoding="utf-8").splitlines()[0])
    assert invocation[:4] == [
        "--foreground",
        "--signal=TERM",
        "--kill-after=9s",
        "17s",
    ]
    assert json.loads(result.read_text(encoding="utf-8")) == [
        "argument with spaces",
        "literal;$(not-executed)",
    ]


def test_executor_rejects_nonpositive_class_timeout(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout_log = tmp_path / "timeout.jsonl"
    _fake_timeout(fake_bin)
    environment = _environment(fake_bin, timeout_log)
    environment["UG_TIMEOUT_GPU_SECONDS"] = "0"
    completed = subprocess.run(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            'source "$1"; initialize_bounded_executor',
            "/bin/bash",
            str(EXECUTOR),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert completed.returncode == 2
    assert "UG_TIMEOUT_GPU_SECONDS must be a positive integer" in completed.stderr


def test_executor_supports_preflight_and_evidence_timeout_classes(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout_log = tmp_path / "timeout.jsonl"
    _fake_timeout(fake_bin)
    environment = _environment(fake_bin, timeout_log)
    environment.update(
        {
            "UG_TIMEOUT_PREFLIGHT_SECONDS": "301",
            "UG_TIMEOUT_EVIDENCE_SECONDS": "901",
        }
    )
    command = (
        'source "$1"; initialize_bounded_executor; '
        "bounded_run preflight true; bounded_run evidence true"
    )
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", command, "bash", str(EXECUTOR)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    invocations = [
        json.loads(line) for line in timeout_log.read_text(encoding="utf-8").splitlines()
    ]
    assert invocations[0][3] == "301s"
    assert invocations[1][3] == "901s"


def test_cached_exact_image_does_not_use_network(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout_log = tmp_path / "timeout.jsonl"
    docker_log = tmp_path / "docker.jsonl"
    _fake_timeout(fake_bin)
    _executable(
        fake_bin / "docker",
        """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["FAKE_DOCKER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
raise SystemExit(93 if sys.argv[1:2] == ["pull"] else 0)
""",
    )
    environment = _environment(fake_bin, timeout_log)
    environment["FAKE_DOCKER_LOG"] = str(docker_log)
    completed = subprocess.run(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            'source "$1"; initialize_bounded_executor; ensure_exact_docker_image "$2"',
            "bash",
            str(EXECUTOR),
            "registry@sha256:" + "1" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    commands = [json.loads(line) for line in docker_log.read_text(encoding="utf-8").splitlines()]
    assert commands == [["image", "inspect", "registry@sha256:" + "1" * 64]]
