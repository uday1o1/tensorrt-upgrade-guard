"""Validate bounded Nsight summaries without treating them as benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from upgrade_guard.contracts.base import sha256_file

SCHEMA_VERSION = "upgradeguard.dev/profiler-validation/v1"
KERNEL = "residualRmsNormFloat4"
SummaryKind = Literal["nsys", "ncu"]


def validate_summary(path: Path, *, summary_kind: SummaryKind) -> dict[str, Any]:
    """Require a selected-kernel row with a finite tool-specific measurement."""

    text = path.read_text(encoding="utf-8")
    if "TensorRT Release" in text or KERNEL not in text:
        raise RuntimeError(
            f"profile summary lacks the selected kernel or contains a banner: {path}"
        )
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    header_index, kernel_index, measurement_indexes = _summary_columns(rows, summary_kind)
    kernel_rows = [
        row
        for row in rows[header_index + 1 :]
        if len(row) > kernel_index and KERNEL in row[kernel_index]
    ]
    if not kernel_rows:
        raise RuntimeError(f"profile summary has no selected-kernel CSV row: {path}")
    if not any(_has_finite_measurement(row, measurement_indexes) for row in kernel_rows):
        raise RuntimeError(f"profile summary has no finite selected-kernel measurement: {path}")
    header = rows[header_index]
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "measurement_columns": [header[index] for index in measurement_indexes],
    }


def validate_profiles(
    *,
    nsys_summary: Path,
    ncu_summary: Path,
    nsys_report: Path,
    nsys_sqlite: Path,
    ncu_report: Path,
) -> dict[str, Any]:
    """Validate both profile pipelines and their retained binary reports."""

    retained = {}
    for name, path in {
        "nsys_report": nsys_report,
        "nsys_sqlite": nsys_sqlite,
        "ncu_report": ncu_report,
    }.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required profiler artifact is absent or empty: {path}")
        retained[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "kernel": KERNEL,
        "profiled": True,
        "primary_performance_gate": False,
        "summaries": {
            "nsys": validate_summary(nsys_summary, summary_kind="nsys"),
            "ncu": validate_summary(ncu_summary, summary_kind="ncu"),
        },
        "retained_reports": retained,
    }


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _summary_columns(
    rows: list[list[str]], summary_kind: SummaryKind
) -> tuple[int, int, tuple[int, ...]]:
    for row_index, row in enumerate(rows):
        normalized = [_normalize_header(field) for field in row]
        if summary_kind == "ncu":
            kernel_names = {"kernel", "kernelname"}
            measurement_names = {"metricvalue"}
        else:
            kernel_names = {"name", "kernel", "kernelname"}
            measurement_names = {
                name
                for name in normalized
                if name.startswith(
                    (
                        "time",
                        "totaltime",
                        "avg",
                        "average",
                        "med",
                        "min",
                        "max",
                        "stddev",
                        "duration",
                    )
                )
            }
        kernel_indexes = [index for index, name in enumerate(normalized) if name in kernel_names]
        measurement_indexes = tuple(
            index for index, name in enumerate(normalized) if name in measurement_names
        )
        if kernel_indexes and measurement_indexes:
            return row_index, kernel_indexes[0], measurement_indexes
    raise RuntimeError(f"profile summary has no recognized {summary_kind} measurement header")


def _has_finite_measurement(row: list[str], indexes: tuple[int, ...]) -> bool:
    for index in indexes:
        if index >= len(row):
            continue
        candidate = row[index].strip().strip('"').replace(",", "").rstrip("%")
        try:
            value = float(candidate)
        except ValueError:
            continue
        if math.isfinite(value):
            return True
    return False


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
    parser.add_argument("--nsys-summary", type=Path, required=True)
    parser.add_argument("--ncu-summary", type=Path, required=True)
    parser.add_argument("--nsys-report", type=Path, required=True)
    parser.add_argument("--nsys-sqlite", type=Path, required=True)
    parser.add_argument("--ncu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _write_atomic(
        arguments.output,
        validate_profiles(
            nsys_summary=arguments.nsys_summary,
            ncu_summary=arguments.ncu_summary,
            nsys_report=arguments.nsys_report,
            nsys_sqlite=arguments.nsys_sqlite,
            ncu_report=arguments.ncu_report,
        ),
    )


if __name__ == "__main__":
    main()
