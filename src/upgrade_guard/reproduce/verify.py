"""Fail-closed directory, tar, and ZIP bundle verification."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from pydantic import ValidationError

from upgrade_guard.contracts.bundle import BundleManifest
from upgrade_guard.errors import InvalidInputError

ALLOWED_SUFFIXES = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".log",
        ".onnx",
        ".data",
        ".npy",
        ".npz",
        ".plan",
        ".cpp",
        ".cu",
        ".cuh",
        ".hpp",
        ".h",
        ".cmake",
        ".sh",
    }
)
DEFAULT_FILE_LIMIT = 512
DEFAULT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedBundle:
    """Verified identity without a trust claim about source code."""

    source: Path
    manifest: BundleManifest
    observed_files: tuple[str, ...]
    source_code_present: bool
    engine_present: bool


@dataclass(frozen=True, slots=True)
class BundleEntry:
    """One bounded regular-file reader."""

    path: str
    size: int
    read: Path | ArchiveReader


@dataclass(frozen=True, slots=True)
class ArchiveReader:
    """One archive member reference with an explicit archive kind."""

    archive_path: Path
    member_name: str
    kind: Literal["zip", "tar"]


class BytesReader(Protocol):
    """Bounded binary reader surface."""

    def read(self, amount: int = -1) -> bytes: ...


def verify_bundle(
    source: Path,
    *,
    maximum_files: int = DEFAULT_FILE_LIMIT,
    maximum_expanded_bytes: int = DEFAULT_SIZE_LIMIT,
) -> VerifiedBundle:
    """Verify names, types, limits, manifest, and every content digest."""

    if maximum_files <= 0 or maximum_expanded_bytes <= 0:
        raise InvalidInputError("bundle verification limits must be positive")
    entries = tuple(_entries(source))
    if len(entries) > maximum_files:
        raise InvalidInputError("bundle exceeds the verifier file-count limit")
    names = [entry.path for entry in entries]
    if len(names) != len(set(names)):
        raise InvalidInputError("bundle contains duplicate paths")
    total = sum(entry.size for entry in entries)
    if total > maximum_expanded_bytes:
        raise InvalidInputError("bundle exceeds the verifier expanded-size limit")
    by_name = {entry.path: entry for entry in entries}
    manifest_entry = by_name.get("bundle.json")
    if manifest_entry is None:
        raise InvalidInputError("bundle is missing bundle.json")
    if manifest_entry.size > MAX_MANIFEST_BYTES:
        raise InvalidInputError("bundle.json exceeds its size limit")
    try:
        payload = json.loads(_read_entry(manifest_entry, MAX_MANIFEST_BYTES))
        manifest = BundleManifest.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise InvalidInputError(
            "bundle.json does not satisfy the strict manifest schema",
            details={"reason": str(error)},
        ) from error
    if len(entries) > min(maximum_files, manifest.file_count_limit):
        raise InvalidInputError("bundle exceeds its declared file-count limit")
    if total > min(maximum_expanded_bytes, manifest.expanded_size_limit_bytes):
        raise InvalidInputError("bundle exceeds its declared expanded-size limit")

    declared = {artifact.path: artifact for artifact in manifest.files}
    actual_payload_names = set(by_name) - {"bundle.json"}
    if set(declared) != actual_payload_names:
        raise InvalidInputError(
            "bundle payload inventory differs from bundle.json",
            details={
                "missing": sorted(set(declared) - actual_payload_names),
                "undeclared": sorted(actual_payload_names - set(declared)),
            },
        )
    for path, artifact in declared.items():
        entry = by_name[path]
        if entry.size != artifact.bytes:
            raise InvalidInputError(f"bundle file size mismatch: {path}")
        digest = _hash_entry(entry)
        if digest != artifact.sha256:
            raise InvalidInputError(f"bundle file hash mismatch: {path}")
    if manifest.manifest_sha256 != manifest.computed_sha256():
        raise InvalidInputError("bundle manifest self-hash is invalid")
    return VerifiedBundle(
        source=source,
        manifest=manifest,
        observed_files=tuple(sorted(by_name)),
        source_code_present=manifest.source_build is not None,
        engine_present=manifest.included_engine is not None,
    )


def materialize_verified_bundle(source: Path, destination: Path) -> VerifiedBundle:
    """Copy a verified bundle into a new directory without archive extraction APIs."""

    if destination.exists() or destination.is_symlink():
        raise InvalidInputError("refusing to overwrite replay materialization")
    verified = verify_bundle(source)
    entries = tuple(_entries(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for entry in entries:
            target = staging / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_read_entry(entry))
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    materialized = verify_bundle(destination)
    if materialized.manifest.manifest_sha256 != verified.manifest.manifest_sha256:
        raise InvalidInputError("materialized bundle identity differs from its source")
    return materialized


def _entries(source: Path) -> Iterator[BundleEntry]:
    if source.is_symlink():
        raise InvalidInputError("bundle source cannot be a symlink")
    if source.is_dir():
        yield from _directory_entries(source)
        return
    if not source.is_file():
        raise InvalidInputError(f"bundle does not exist: {source}")
    if zipfile.is_zipfile(source):
        yield from _zip_entries(source)
        return
    try:
        if tarfile.is_tarfile(source):
            yield from _tar_entries(source)
            return
    except OSError as error:
        raise InvalidInputError("bundle archive could not be inspected") from error
    raise InvalidInputError("bundle must be a directory, ZIP, or tar archive")


def _directory_entries(root: Path) -> Iterator[BundleEntry]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise InvalidInputError(f"bundle contains a symlink: {path.relative_to(root)}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise InvalidInputError(f"bundle contains a non-regular file: {path.relative_to(root)}")
        relative = path.relative_to(root).as_posix()
        _validate_member_path(relative)
        _validate_suffix(relative)
        size = path.stat().st_size
        yield BundleEntry(relative, size, path)


def _zip_entries(source: Path) -> Iterator[BundleEntry]:
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise InvalidInputError(f"bundle ZIP contains a symlink: {info.filename}")
            _validate_member_path(info.filename)
            _validate_suffix(info.filename)
            yield BundleEntry(
                info.filename,
                info.file_size,
                ArchiveReader(source, info.filename, "zip"),
            )


def _tar_entries(source: Path) -> Iterator[BundleEntry]:
    with tarfile.open(source, mode="r:*") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise InvalidInputError(f"bundle tar contains a non-regular file: {member.name}")
            _validate_member_path(member.name)
            _validate_suffix(member.name)
            yield BundleEntry(
                member.name,
                member.size,
                ArchiveReader(source, member.name, "tar"),
            )


def _validate_member_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or "\x00" in value
    ):
        raise InvalidInputError(f"unsafe bundle path: {value!r}")


def _validate_suffix(value: str) -> None:
    if value in {"bundle.json", "SHA256SUMS"}:
        return
    if Path(value).suffix.lower() not in ALLOWED_SUFFIXES:
        raise InvalidInputError(f"unsupported bundle file type: {value}")


def _read_entry(entry: BundleEntry, maximum_bytes: int | None = None) -> bytes:
    if maximum_bytes is not None and entry.size > maximum_bytes:
        raise InvalidInputError(f"bundle entry exceeds size limit: {entry.path}")
    reader = entry.read
    if isinstance(reader, Path):
        return _read_bounded_file(reader, entry.size)
    if reader.kind == "zip":
        with (
            zipfile.ZipFile(reader.archive_path) as zip_archive,
            zip_archive.open(reader.member_name) as handle,
        ):
            return _read_bounded_handle(handle, entry.size)
    with tarfile.open(reader.archive_path, mode="r:*") as tar_archive:
        tar_handle = tar_archive.extractfile(reader.member_name)
        if tar_handle is None:
            raise InvalidInputError(f"bundle tar entry could not be read: {reader.member_name}")
        with tar_handle:
            return _read_bounded_handle(tar_handle, entry.size)


def _read_bounded_file(path: Path, expected_size: int) -> bytes:
    with path.open("rb") as handle:
        return _read_bounded_handle(handle, expected_size)


def _read_bounded_handle(handle: BytesReader, expected_size: int) -> bytes:
    content = handle.read(expected_size + 1)
    if len(content) != expected_size:
        raise InvalidInputError("bundle entry changed or expanded while being read")
    return content


def _hash_entry(entry: BundleEntry) -> str:
    digest = hashlib.sha256(_read_entry(entry)).hexdigest()
    return f"sha256:{digest}"
