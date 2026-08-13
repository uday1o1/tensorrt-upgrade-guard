"""Shared strict contract behavior and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for artifact identity."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase, algorithm-qualified SHA-256 digest."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def model_sha256(model: BaseModel, *, exclude: set[str] | None = None) -> str:
    """Hash the complete JSON representation of a contract."""

    payload = model.model_dump(mode="json", exclude=exclude or set())
    return sha256_bytes(canonical_json_bytes(payload))
