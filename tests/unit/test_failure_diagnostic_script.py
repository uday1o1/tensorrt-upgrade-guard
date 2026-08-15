"""Redaction and safe-inventory tests for failure diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.write_failure_diagnostic import classify_failure, write_diagnostic

SOURCE = "b" * 40
GPU = "GPU-12345678-abcd-1234-abcd-1234567890ab"


def _state(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    (state / "done").mkdir(parents=True)
    (state / "logs").mkdir()
    (state / "done" / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/qualification-step/v2",
                "step": "preflight",
            }
        ),
        encoding="utf-8",
    )
    (state / "done" / "broken.json").write_text("not json", encoding="utf-8")
    return state


def test_diagnostic_is_atomic_redacted_and_uses_only_local_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    secret = "registry-token-" + "super-secret"
    monkeypatch.setenv("UPGRADE_GUARD_REGISTRY_TOKEN", secret)
    (state / "logs" / "core-qualification.log").write_text(secret, encoding="utf-8")
    output = write_diagnostic(
        state=state,
        step="core-qualification",
        exit_code=4,
        source=SOURCE,
        mode="full",
        gpu=GPU,
    )
    text = output.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert secret not in text
    assert payload["classification"] == "INFRASTRUCTURE_INVALID"
    assert payload["markers"] == {
        "valid": ["preflight.json"],
        "invalid": ["broken.json"],
        "unsafe_entry_count": 0,
    }
    assert payload["logs"] == [{"path": "logs/core-qualification.log", "bytes": len(secret)}]
    assert payload["resume_command"] == ["bash", "scripts/run_gpu_qualification.sh"]
    assert not list((state / "diagnostics").glob(f".{output.name}.*"))


def test_unsafe_marker_and_log_paths_are_never_followed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    outside = tmp_path / "outside-secret"
    outside.write_text("do-not-read", encoding="utf-8")
    (state / "done" / "linked.json").symlink_to(outside)
    (state / "logs" / "linked.log").symlink_to(outside)
    (state / "logs" / "unsafe name.log").write_text("ignored", encoding="utf-8")
    output = write_diagnostic(state=state, step="profiles", exit_code=1)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "linked.json" in payload["markers"]["invalid"]
    assert payload["logs"] == []
    assert payload["unsafe_log_entry_count"] == 2
    assert "outside-secret" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("exit_code", "requested", "expected"),
    [
        (1, "auto", ("QUALIFICATION_FAILED", "failed")),
        (4, "auto", ("INFRASTRUCTURE_INVALID", "infrastructure_invalid")),
        (1, "enospc", ("ENOSPC", "infrastructure_invalid")),
        (4, "failed", ("QUALIFICATION_FAILED", "failed")),
    ],
)
def test_failure_classifications(exit_code: int, requested: str, expected: tuple[str, str]) -> None:
    assert classify_failure(exit_code, requested) == expected  # type: ignore[arg-type]


def test_state_and_identity_validation_prevent_unsafe_output(tmp_path: Path) -> None:
    state = _state(tmp_path)
    linked = tmp_path / "linked-state"
    linked.symlink_to(state, target_is_directory=True)
    with pytest.raises(ValueError, match="state"):
        write_diagnostic(state=linked, step="preflight", exit_code=1)
    with pytest.raises(ValueError, match="source"):
        write_diagnostic(
            state=state,
            step="preflight",
            exit_code=1,
            source="secret=credential",
        )
    assert not (state / "diagnostics").exists()


def test_smoke_resume_command_is_exact(tmp_path: Path) -> None:
    state = _state(tmp_path)
    output = write_diagnostic(state=state, step="gpu-smoke", exit_code=1, mode="smoke")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["resume_command"] == [
        "env",
        "UG_SMOKE_ONLY=1",
        "bash",
        "scripts/run_gpu_qualification.sh",
    ]
    assert os.path.commonpath((str(output.resolve()), str(state.resolve()))) == str(state.resolve())
