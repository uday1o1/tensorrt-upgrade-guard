"""Content-addressed, append-only corpus publication helpers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CorpusKind = Literal["core", "plugin", "mobilenet"]
MATERIALIZER_NAME = "materializer.json"

COMMON_SOURCES = (
    "pyproject.toml",
    "uv.lock",
    "src/upgrade_guard/contracts/base.py",
    "src/upgrade_guard/corpus/generators.py",
    "src/upgrade_guard/corpus/reference.py",
)
KIND_SOURCES: dict[CorpusKind, tuple[str, ...]] = {
    "core": (
        "corpus/registry.yaml",
        "models/generators/tiny_transformer.py",
        "models/locks/tiny_transformer.lock.json",
        "src/upgrade_guard/corpus/materialize.py",
        "src/upgrade_guard/corpus/registry.py",
    ),
    "plugin": (
        "models/generators/plugin_micrograph.py",
        "models/locks/plugin_micrograph.lock.json",
        "scripts/materialize_plugin_corpus.py",
        "src/upgrade_guard/corpus/plugin.py",
    ),
    "mobilenet": (
        "corpus/attribution.yaml",
        "models/generators/derive_dynamic_mobilenet.py",
        "models/locks/mobilenetv3.lock.json",
        "scripts/materialize_mobilenet_corpus.py",
        "src/upgrade_guard/corpus/mobilenet.py",
    ),
}
LOCK_NAMES: dict[CorpusKind, str] = {
    "core": "corpus.lock.json",
    "plugin": "plugin-corpus.lock.json",
    "mobilenet": "mobilenet-corpus.lock.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def materializer_document(project: Path, kind: CorpusKind) -> dict[str, Any]:
    """Hash every checked-in producer and dependency identity for one corpus."""

    project = project.resolve(strict=True)
    source_paths = tuple(sorted((*COMMON_SOURCES, *KIND_SOURCES[kind])))
    sources: dict[str, dict[str, int | str]] = {}
    for relative in source_paths:
        path = (project / relative).resolve(strict=True)
        if not path.is_relative_to(project) or not path.is_file() or path.is_symlink():
            raise ValueError(f"materializer source is not a safe regular file: {relative}")
        sources[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    identity_payload = {
        "schema_version": "upgradeguard.dev/corpus-materializer/v1",
        "kind": kind,
        "sources": sources,
    }
    identity = f"sha256:{hashlib.sha256(_canonical(identity_payload)).hexdigest()}"
    return {**identity_payload, "materializer_sha256": identity}


def write_sidecar(root: Path, document: dict[str, Any]) -> Path:
    """Atomically bind a generated staging corpus to its materializer identity."""

    root = root.resolve(strict=True)
    destination = root / MATERIALIZER_NAME
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"materializer sidecar already exists: {destination}")
    _write_atomic(destination, document)
    return destination


def verify_sidecar(root: Path, expected: dict[str, Any]) -> None:
    root = root.resolve(strict=True)
    sidecar = root / MATERIALIZER_NAME
    try:
        observed = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"corpus materializer sidecar is invalid: {sidecar}") from error
    if observed != expected:
        raise ValueError(f"corpus materializer identity differs: {sidecar}")


def tree_inventory(root: Path) -> dict[str, dict[str, int | str]]:
    """Return a safe, deterministic complete regular-file inventory."""

    root = root.resolve(strict=True)
    inventory: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"corpus contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"corpus contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        inventory[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if MATERIALIZER_NAME not in inventory:
        raise ValueError("corpus lacks its materializer sidecar")
    return inventory


def publish(staging: Path, destination: Path, expected: dict[str, Any]) -> str:
    """Publish once under a global lock or verify an identical existing object."""

    staging = staging.resolve(strict=True)
    destination = destination.resolve()
    verify_sidecar(staging, expected)
    staged_inventory = tree_inventory(staging)
    store = destination.parent
    store.mkdir(parents=True, exist_ok=True)
    lock_path = store.parent / ".publish.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError(f"corpus identity collides with a non-directory: {destination}")
            verify_sidecar(destination, expected)
            if tree_inventory(destination) != staged_inventory:
                raise ValueError(f"corpus identity collision has different content: {destination}")
            preserved = _preserved_staging_path(store.parent, staging.name)
            staging.replace(preserved)
            return "reused"
        staging.replace(destination)
        _make_read_only(destination)
        return "published"


def corpus_index(
    project: Path,
    roots: dict[CorpusKind, Path],
) -> dict[str, Any]:
    """Build the source-run pointer to immutable content-addressed corpora."""

    project = project.resolve(strict=True)
    entries: dict[str, dict[str, str]] = {}
    for kind, authored_root in sorted(roots.items()):
        root = authored_root.resolve(strict=True)
        if not root.is_relative_to(project) or root.is_symlink():
            raise ValueError(f"corpus root is outside the project or is a symlink: {root}")
        expected = materializer_document(project, kind)
        verify_sidecar(root, expected)
        inventory = tree_inventory(root)
        lock_path = root / LOCK_NAMES[kind]
        if not lock_path.is_file() or lock_path.is_symlink():
            raise ValueError(f"corpus lock is absent: {lock_path}")
        entries[kind] = {
            "root": root.relative_to(project).as_posix(),
            "lock": lock_path.relative_to(project).as_posix(),
            "lock_sha256": _sha256(lock_path),
            "materializer_sha256": str(expected["materializer_sha256"]),
            "inventory_sha256": f"sha256:{hashlib.sha256(_canonical(inventory)).hexdigest()}",
        }
    return {"schema_version": "upgradeguard.dev/corpus-index/v1", "corpora": entries}


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError(f"corpus contains a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _preserved_staging_path(store: Path, name: str) -> Path:
    stale = store / "stale"
    stale.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return stale / f"{timestamp}-{name}"


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _kind(value: str) -> CorpusKind:
    if value not in KIND_SOURCES:
        raise argparse.ArgumentTypeError(f"unsupported corpus kind: {value}")
    return value  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    identity = subparsers.add_parser("identity")
    identity.add_argument("--project", type=Path, required=True)
    identity.add_argument("--kind", type=_kind, required=True)
    sidecar = subparsers.add_parser("write-sidecar")
    sidecar.add_argument("--project", type=Path, required=True)
    sidecar.add_argument("--kind", type=_kind, required=True)
    sidecar.add_argument("--root", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--project", type=Path, required=True)
    verify.add_argument("--kind", type=_kind, required=True)
    verify.add_argument("--root", type=Path, required=True)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--project", type=Path, required=True)
    publish_parser.add_argument("--kind", type=_kind, required=True)
    publish_parser.add_argument("--staging", type=Path, required=True)
    publish_parser.add_argument("--destination", type=Path, required=True)
    index_parser = subparsers.add_parser("write-index")
    index_parser.add_argument("--project", type=Path, required=True)
    index_parser.add_argument("--output", type=Path, required=True)
    index_parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="KIND=PATH",
    )
    arguments = parser.parse_args()
    if arguments.command == "write-index":
        roots: dict[CorpusKind, Path] = {}
        for authored in arguments.corpus:
            name, separator, path = authored.partition("=")
            if not separator:
                raise ValueError(f"invalid corpus index entry: {authored}")
            kind = _kind(name)
            if kind in roots:
                raise ValueError(f"duplicate corpus index entry: {kind}")
            roots[kind] = Path(path)
        if not roots:
            raise ValueError("corpus index requires at least one corpus")
        _write_atomic(arguments.output, corpus_index(arguments.project, roots))
        return 0
    document = materializer_document(arguments.project, arguments.kind)
    if arguments.command == "identity":
        print(json.dumps(document, allow_nan=False, sort_keys=True))
    elif arguments.command == "write-sidecar":
        write_sidecar(arguments.root, document)
    elif arguments.command == "verify":
        verify_sidecar(arguments.root, document)
        tree_inventory(arguments.root)
    else:
        print(publish(arguments.staging, arguments.destination, document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
