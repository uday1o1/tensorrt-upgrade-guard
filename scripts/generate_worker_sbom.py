"""Generate a deterministic SPDX inventory inside one exact worker image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _identifier(kind: str, name: str, version: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{name}\0{version}".encode()).hexdigest()[:20]
    return f"SPDXRef-{kind}-{digest}"


def _packages() -> list[dict[str, object]]:
    packages: set[tuple[str, str, str]] = set()
    result = subprocess.run(
        ("dpkg-query", "-W", "-f=${Package}\t${Version}\n"),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            name, separator, version = line.partition("\t")
            if separator:
                packages.add(("deb", name, version))
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.add(("python", name, distribution.version))
    return [
        {
            "SPDXID": _identifier(kind, name, version),
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
                    "referenceLocator": f"pkg:{kind}/{name}@{version}",
                }
            ],
        }
        for kind, name, version in sorted(packages)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", arguments.image):
        raise RuntimeError("worker SBOM requires an immutable image manifest")
    identity = hashlib.sha256(arguments.image.encode()).hexdigest()
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "TensorRT UpgradeGuard worker inventory",
        "documentNamespace": f"https://upgradeguard.dev/spdx/{identity}",
        "creationInfo": {
            "created": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "creators": ["Tool: tensorrt-upgrade-guard-0.1.0"],
        },
        "documentComment": f"Observed package inventory for {arguments.image}",
        "packages": _packages(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
