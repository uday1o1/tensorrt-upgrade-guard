"""Validate exact host and worker SPDX documents before measured GPU gates."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from upgrade_guard.contracts.base import sha256_file

SCHEMA_VERSION = "upgradeguard.dev/sbom-validation/v1"


def validate_spdx(
    path: Path,
    *,
    expected_image: str | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, int | str]:
    """Require a populated SPDX 2.3 document bound to its source identity."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"SBOM is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"SBOM is not a JSON object: {path}")
    packages = value.get("packages")
    if (
        value.get("spdxVersion") != "SPDX-2.3"
        or not isinstance(packages, list)
        or not packages
        or not all(isinstance(package, dict) for package in packages)
    ):
        raise RuntimeError(f"SBOM is not a populated SPDX 2.3 document: {path}")
    comment = value.get("documentComment")
    if not isinstance(comment, str):
        raise RuntimeError(f"SBOM does not contain a document identity comment: {path}")
    if expected_image is not None and expected_image not in comment:
        raise RuntimeError(f"worker SBOM is not bound to its locked image: {path}")
    if expected_lock_sha256 is not None and expected_lock_sha256 not in comment:
        raise RuntimeError(f"host SBOM is not bound to its exact lock: {path}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "package_count": len(packages),
    }


def validate_documents(
    *,
    host: Path,
    baseline: Path,
    candidate: Path,
    baseline_image: str,
    candidate_image: str,
    lock: Path,
) -> dict[str, Any]:
    """Validate all three release SBOM scopes and retain their hashes."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "documents": {
            "host": validate_spdx(host, expected_lock_sha256=sha256_file(lock)),
            "baseline_worker": validate_spdx(baseline, expected_image=baseline_image),
            "candidate_worker": validate_spdx(candidate, expected_image=candidate_image),
        },
        "claim_scope": (
            "Worker SPDX files inventory the observed image packages. "
            "They do not claim that every preinstalled package is vulnerability-free."
        ),
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
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-image", required=True)
    parser.add_argument("--candidate-image", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _write_atomic(
        arguments.output,
        validate_documents(
            host=arguments.host,
            baseline=arguments.baseline,
            candidate=arguments.candidate,
            baseline_image=arguments.baseline_image,
            candidate_image=arguments.candidate_image,
            lock=arguments.lock,
        ),
    )


if __name__ == "__main__":
    main()
