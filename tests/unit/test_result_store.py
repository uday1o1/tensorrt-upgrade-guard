"""Atomic result-directory tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from upgrade_guard.errors import InvalidInputError
from upgrade_guard.results import ResultTransaction


def test_result_transaction_publishes_complete_artifacts(tmp_path: Path) -> None:
    with ResultTransaction(tmp_path, "run-001") as transaction:
        artifact = transaction.write_text("logs/worker.log", "complete\n")
        destination = transaction.publish()
    assert destination == tmp_path / "run-001"
    assert (destination / artifact.path).read_text(encoding="utf-8") == "complete\n"
    assert artifact.bytes == 9


def test_unpublished_or_failed_transactions_are_removed(tmp_path: Path) -> None:
    with ResultTransaction(tmp_path, "run-001") as transaction:
        temporary = transaction.temporary
        transaction.write_text("partial.txt", "partial")
    assert not temporary.exists()
    assert not (tmp_path / "run-001").exists()


@pytest.mark.parametrize(
    "path",
    ["../escape", "/absolute", "nested/../escape", "bad\\path", "bad\x00path"],
)
def test_result_artifact_paths_fail_closed(tmp_path: Path, path: str) -> None:
    with ResultTransaction(tmp_path, "run-001") as transaction:
        with pytest.raises(InvalidInputError, match="normalized"):
            transaction.write_text(path, "content")


def test_result_transaction_refuses_overwrites(tmp_path: Path) -> None:
    with ResultTransaction(tmp_path, "run-001") as transaction:
        transaction.write_text("artifact.txt", "one")
        with pytest.raises(InvalidInputError, match="overwrite"):
            transaction.write_text("artifact.txt", "two")
        transaction.publish()
        with pytest.raises(InvalidInputError, match="already been published"):
            transaction.publish()
    with pytest.raises(InvalidInputError, match="overwrite"):
        ResultTransaction(tmp_path, "run-001")


def test_result_id_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="result ID"):
        ResultTransaction(tmp_path, "../bad")
