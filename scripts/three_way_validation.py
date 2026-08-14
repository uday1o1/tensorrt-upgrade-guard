"""Shared typed failure evidence for extended three-way validators."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from upgrade_guard.classify import status_for_failure
from upgrade_guard.compare.determinism import summarize_determinism
from upgrade_guard.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from upgrade_guard.contracts.common import (
    ArtifactReference,
    FailureRecord,
    NumericalTolerance,
    Phase,
    PrecisionMode,
    ResultStatus,
)
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode


class ThreeWayValidationResult(StrictModel):
    """Stable extended-case result with a typed terminal failure."""

    schema_version: Literal[
        "upgradeguard.dev/plugin-validation/v2",
        "upgradeguard.dev/mobilenet-validation/v2",
    ]
    status: ResultStatus
    failure_code: FailureCode | None
    failure: FailureRecord | None
    specification_sha256: Sha256Digest
    invocation_manifest: ArtifactReference
    invocation_manifest_sha256: Sha256Digest
    repetitions: int = Field(ge=20)
    cases: tuple[dict[str, Any], ...]

    @model_validator(mode="after")
    def validate_terminal_state(self) -> ThreeWayValidationResult:
        code = self.failure.code if self.failure else None
        if self.failure_code is not code or self.status is not status_for_failure(code):
            raise ValueError("validation status, failure code, and failure record must agree")
        return self


def worker_evidence_failure_code(
    result_path: Path,
    *,
    environment_id: str,
    error: RuntimeError,
) -> FailureCode:
    """Preserve a worker's typed code or classify host evidence validation."""

    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict) and value.get("status") == "failed":
        raw_code = value.get("failure_code")
        if not isinstance(raw_code, str):
            return FailureCode.INFRASTRUCTURE_INVALID
        try:
            return FailureCode(raw_code)
        except ValueError:
            return FailureCode.INFRASTRUCTURE_INVALID
    if is_output_schema_failure(error):
        return (
            FailureCode.CORPUS_INVALID
            if environment_id == "baseline"
            else FailureCode.OUTPUT_SCHEMA_CHANGED
        )
    return FailureCode.INFRASTRUCTURE_INVALID


def is_output_schema_failure(error: RuntimeError) -> bool:
    """Return whether host validation observed the worker's output contract drifting."""

    message = str(error)
    return any(
        fragment in message
        for fragment in (
            "worker output inventory changed",
            "worker output name changed",
            "worker output schema differs",
        )
    )


def worker_output_tolerance_stable(
    result_path: Path,
    *,
    runs_root: Path,
    policy: NumericalTolerance,
) -> bool:
    """Recompute determinism when only the reference-facing output schema differs."""

    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("schema-failed worker result is invalid") from error
    repetitions = value.get("repetitions") if isinstance(value, dict) else None
    if not isinstance(repetitions, list) or not repetitions:
        raise RuntimeError("schema-failed worker result has no repetitions")
    root = runs_root.resolve(strict=True)
    observed = []
    for repetition in repetitions:
        outputs = repetition.get("outputs") if isinstance(repetition, dict) else None
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
            raise RuntimeError("schema-failed worker output inventory is not promotable")
        artifact = outputs[0]
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            raise RuntimeError("schema-failed worker output path is invalid")
        container_path = PurePosixPath(raw_path)
        output_root = PurePosixPath("/output")
        if not container_path.is_relative_to(output_root) or ".." in container_path.parts:
            raise RuntimeError("schema-failed worker output path escaped /output")
        path = root / Path(*container_path.relative_to(output_root).parts)
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_relative_to(root)
            or path.is_symlink()
            or artifact.get("sha256") != sha256_file(resolved)
        ):
            raise RuntimeError("schema-failed worker output identity differs")
        array = np.load(resolved, allow_pickle=False)
        if artifact.get("dtype") != str(array.dtype) or artifact.get("shape") != list(array.shape):
            raise RuntimeError("schema-failed worker output metadata differs")
        observed.append(array)
    return summarize_determinism(
        tuple(observed),
        (),
        policy,
        input_hashes_stable=value.get("input_integrity_stable") is True,
    ).tolerance_stable


def failure_record(
    *,
    code: FailureCode,
    phase: Phase,
    environment_id: str,
    model_id: str,
    precision: PrecisionMode,
    case_id: str,
    output_name: str,
    gate: str,
    observed: str,
    threshold: str,
    runs_root: Path,
    evidence_paths: tuple[Path, ...],
) -> FailureRecord:
    """Create a stable failure record bound to retained worker evidence."""

    root = runs_root.resolve(strict=True)
    evidence = tuple(_artifact(path, root) for path in evidence_paths)
    predicate = {
        "code": code.value,
        "phase": phase.value,
        "environment_id": environment_id,
        "model_id": model_id,
        "precision": precision.value,
        "case_id": case_id,
        "output_name": output_name,
        "gate": gate,
        "observed": observed,
        "threshold": threshold,
        "evidence": [item.model_dump(mode="json") for item in evidence],
    }
    return FailureRecord(
        code=code,
        phase=phase,
        environment_id=environment_id,
        model_id=model_id,
        precision=precision,
        shape_id=case_id,
        input_fixture_id=case_id,
        output_name=output_name,
        gate=gate,
        observed=observed,
        threshold=threshold,
        evidence=evidence,
        signature_sha256=sha256_bytes(canonical_json_bytes(predicate)),
    )


def _artifact(path: Path, root: Path) -> ArtifactReference:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or path.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"validation failure evidence escaped the runs root: {path}")
    return ArtifactReference(
        path=resolved.relative_to(root).as_posix(),
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
        media_type="application/json",
    )
