"""Dependency-light worker helpers for locked NVIDIA containers."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json_atomic(path: Path, value: Any) -> None:
    """Publish one worker result without exposing a partial JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def command_evidence(module: str, arguments: list[str]) -> dict[str, object]:
    """Return a stable worker CLI argument array and its canonical SHA-256."""

    command = ("python3", "-m", module, *arguments)
    canonical = json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "command": list(command),
        "command_sha256": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    }


def process_memory_evidence() -> dict[str, Any]:
    """Return separately labeled host peak RSS and coarse GPU process observations."""

    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_peak_rss_bytes = int(maximum_rss if sys.platform == "darwin" else maximum_rss * 1024)
    command = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    rows = []
    if result is not None and result.returncode == 0:
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return {
        "host_peak_rss_bytes": host_peak_rss_bytes,
        "gpu_process_rows": rows,
        "gpu_process_observation": "coarse post-operation nvidia-smi sample",
    }
