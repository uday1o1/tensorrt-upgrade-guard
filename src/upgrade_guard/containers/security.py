"""Fail-closed GPU worker isolation policy."""

from __future__ import annotations

from pathlib import Path

from upgrade_guard.errors import InvalidInputError


def validated_mount(path: Path, *, must_exist: bool) -> Path:
    """Resolve a narrow mount path without symlinks or broad host roots."""

    if must_exist and not path.exists():
        raise InvalidInputError("container mount path does not exist", details={"path": str(path)})
    resolved = path.resolve(strict=must_exist)
    if resolved in (Path("/"), Path.home().resolve()) or len(resolved.parts) < 4:
        raise InvalidInputError("refusing broad container mount", details={"path": str(resolved)})
    if path.is_symlink():
        raise InvalidInputError("container mount cannot be a symlink", details={"path": str(path)})
    return resolved


def validate_locked_image(reference: str) -> str:
    """Require an exact OCI digest at the GPU execution boundary."""

    marker = "@sha256:"
    if marker not in reference:
        raise InvalidInputError("GPU worker image must use an immutable sha256 digest")
    digest = reference.rpartition(marker)[2]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise InvalidInputError("GPU worker image digest is malformed")
    return reference
