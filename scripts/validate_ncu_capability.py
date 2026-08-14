"""Validate an early Nsight Compute protected-counter capability probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts.validate_profiler_outputs import validate_summary
from upgrade_guard.contracts.base import sha256_file

SCHEMA_VERSION = "upgradeguard.dev/profiler-preflight/v3"


def validate_capability(*, summary: Path, report: Path) -> dict[str, Any]:
    """Require a retained report and a finite selected-kernel counter value."""

    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError(f"required counter capability report is absent or empty: {report}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "capability_only": True,
        "primary_performance_gate": False,
        "diagnostic_profile": False,
        "hardware_counter_collection": True,
        "counter_permission_verified": True,
        "required_ncu_sections": ["SpeedOfLight"],
        "summary": validate_summary(summary, summary_kind="ncu"),
        "report": {"sha256": sha256_file(report), "bytes": report.stat().st_size},
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _write_atomic(
        arguments.output,
        validate_capability(summary=arguments.summary, report=arguments.report),
    )


if __name__ == "__main__":
    main()
