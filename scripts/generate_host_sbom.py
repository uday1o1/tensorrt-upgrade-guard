"""Generate an SPDX inventory from the exact uv lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upgrade_guard.contracts.base import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    lock = arguments.lock.resolve(strict=True)
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    packages = data.get("package")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("uv lock contains no package inventory")
    entries = [_package(value) for value in packages]
    lock_digest = sha256_file(lock).removeprefix("sha256:")
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "TensorRT UpgradeGuard host lock inventory",
        "documentNamespace": f"https://upgradeguard.dev/spdx/host/{lock_digest}",
        "creationInfo": {
            "created": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "creators": ["Tool: tensorrt-upgrade-guard-0.1.0"],
        },
        "documentComment": f"Package inventory from uv.lock sha256:{lock_digest}",
        "packages": sorted(entries, key=lambda item: str(item["name"])),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package(value: Any) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("uv package entry is not an object")
    name = value.get("name")
    version = value.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise RuntimeError("uv package entry lacks a name or version")
    identifier = hashlib.sha256(f"{name}\0{version}".encode()).hexdigest()[:20]
    return {
        "SPDXID": f"SPDXRef-pypi-{identifier}",
        "name": name,
        "versionInfo": version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "supplier": "NOASSERTION",
        "primaryPackagePurpose": "LIBRARY",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:pypi/{name}@{version}",
            }
        ],
    }


if __name__ == "__main__":
    main()
