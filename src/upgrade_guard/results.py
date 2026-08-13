"""Safe result-directory creation and artifact retention."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from types import TracebackType

from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.errors import InfrastructureError, InvalidInputError

_RESULT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ResultTransaction(AbstractContextManager["ResultTransaction"]):
    """Construct a run directory privately and publish it atomically."""

    def __init__(self, root: Path, result_id: str) -> None:
        if not _RESULT_ID.fullmatch(result_id):
            raise InvalidInputError("result ID contains unsupported characters")
        self.root = root
        self.result_id = result_id
        self.destination = root / result_id
        if self.destination.exists() or self.destination.is_symlink():
            raise InvalidInputError(f"refusing to overwrite result directory: {result_id}")
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(tempfile.mkdtemp(prefix=f".{result_id}.", dir=root))
        self._published = False

    def __enter__(self) -> ResultTransaction:
        return self

    def write_bytes(self, relative_path: str, content: bytes, media_type: str) -> ArtifactReference:
        """Write one normalized relative artifact and return its identity."""

        path = _safe_relative_path(relative_path)
        destination = self.temporary.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise InvalidInputError(f"refusing to overwrite result artifact: {relative_path}")
        try:
            with destination.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise InfrastructureError(
                f"could not write result artifact: {relative_path}"
            ) from error
        return ArtifactReference(
            path=relative_path,
            sha256=sha256_bytes(content),
            bytes=len(content),
            media_type=media_type,
        )

    def write_text(
        self,
        relative_path: str,
        content: str,
        media_type: str = "text/plain; charset=utf-8",
    ) -> ArtifactReference:
        return self.write_bytes(relative_path, content.encode("utf-8"), media_type)

    def publish(self) -> Path:
        """Atomically expose a complete result directory."""

        if self._published:
            raise InvalidInputError("result transaction has already been published")
        try:
            self.temporary.replace(self.destination)
        except OSError as error:
            raise InfrastructureError("could not atomically publish result directory") from error
        self._published = True
        return self.destination

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        del exc_type, exc_value, traceback
        if not self._published:
            shutil.rmtree(self.temporary, ignore_errors=True)
        return None


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or "\x00" in value
    ):
        raise InvalidInputError("result artifact path must be normalized and relative")
    return path
