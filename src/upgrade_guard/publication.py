"""Strict cross-file validation for a completed public qualification."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import ValidationError

from upgrade_guard.classify import exit_code_for_failure
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock
from upgrade_guard.errors import ExitCode, FailureCode
from upgrade_guard.gates import (
    MARKER_SCHEMA,
    POST_PUBLICATION_ARTIFACTS,
    STEP_ALIASES,
    STEP_OWNED_PATHS,
    direct_step_dependencies,
    expected_publication_steps,
    step_is_bound_to,
)
from upgrade_guard.report.model import ReportModel


class PublicationValidationError(ValueError):
    """A purported terminal publication is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class PublicationDecision:
    """Validated public decision and its stored machine-readable result table."""

    status: Literal["passed", "failed"]
    failure_codes: tuple[FailureCode, ...]
    core_qualification: dict[str, Any]
    results: dict[str, Any]


def validate_publication(
    directory: Path,
    *,
    expected_status: Literal["passed", "failed"] | None = None,
) -> PublicationDecision:
    """Validate every terminal publication file and its important hash links."""

    root = _safe_root(directory)
    paths = {
        name: _regular_file(root, name)
        for name in ("evidence.json", "results.json", "report-model.json", "report.md")
    }
    evidence = _json_object(paths["evidence.json"])
    results = _json_object(paths["results.json"])
    report_value = _json_object(paths["report-model.json"])
    try:
        report = ReportModel.model_validate(report_value)
    except ValidationError as error:
        raise PublicationValidationError("published report model is invalid") from error
    status = _status(evidence, results, report, expected_status=expected_status)
    failure_codes = _failure_codes(results, evidence, status=status)
    _schemas(evidence, results)
    _generated_artifacts(root, evidence)
    _artifact_reference(root, report.results_artifact, expected_path="results.json")
    for reference in (*report.evidence, *report.corpus_provenance):
        _artifact_reference(root, reference)
    core = _core_qualification(
        root,
        evidence,
        results,
        status=status,
        codes=failure_codes,
    )
    _inventory(root, evidence)
    _provenance(root, evidence, results, report)
    _gates(root, evidence, results, report, status=status)
    _failure_payload(root, evidence, results, report, status=status, codes=failure_codes)
    if not paths["report.md"].read_text(encoding="utf-8").strip():
        raise PublicationValidationError("published Markdown report is empty")
    return PublicationDecision(status, failure_codes, core, results)


def _safe_root(directory: Path) -> Path:
    if directory.is_symlink():
        raise PublicationValidationError("publication directory cannot be a symlink")
    try:
        root = directory.resolve(strict=True)
    except OSError as error:
        raise PublicationValidationError("publication directory is unavailable") from error
    if not root.is_dir():
        raise PublicationValidationError("publication path is not a directory")
    return root


def _regular_file(root: Path, relative: str) -> Path:
    if not _safe_relative(relative):
        raise PublicationValidationError(f"publication artifact path is unsafe: {relative}")
    path = root / relative
    current = root
    for part in PurePosixPath(relative).parts[:-1]:
        current /= part
        try:
            parent_mode = current.lstat().st_mode
        except OSError as error:
            raise PublicationValidationError(
                f"publication artifact parent is unavailable: {relative}"
            ) from error
        if not stat.S_ISDIR(parent_mode):
            raise PublicationValidationError(f"publication artifact parent is unsafe: {relative}")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise PublicationValidationError(f"publication is missing {relative}") from error
    if not stat.S_ISREG(mode):
        raise PublicationValidationError(f"publication artifact is not regular: {relative}")
    try:
        if not path.resolve(strict=True).is_relative_to(root):
            raise PublicationValidationError(f"publication artifact escaped its root: {relative}")
    except OSError as error:
        raise PublicationValidationError(
            f"publication artifact is unavailable: {relative}"
        ) from error
    return path


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationValidationError(f"publication JSON is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise PublicationValidationError(f"publication JSON is not an object: {path.name}")
    return value


def _status(
    evidence: dict[str, Any],
    results: dict[str, Any],
    report: ReportModel,
    *,
    expected_status: Literal["passed", "failed"] | None,
) -> Literal["passed", "failed"]:
    observed = (evidence.get("status"), results.get("status"), report.status.value)
    if len(set(observed)) != 1 or observed[0] not in {"passed", "failed"}:
        raise PublicationValidationError("publication status differs across terminal artifacts")
    status: Literal["passed", "failed"] = observed[0]
    if expected_status is not None and status != expected_status:
        raise PublicationValidationError("publication has the wrong terminal status")
    if not report.publication_complete:
        raise PublicationValidationError("published report model is incomplete")
    return status


def _failure_codes(
    results: dict[str, Any],
    evidence: dict[str, Any],
    *,
    status: Literal["passed", "failed"],
) -> tuple[FailureCode, ...]:
    raw = results.get("failure_codes")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise PublicationValidationError("published failure codes are invalid")
    try:
        codes = tuple(FailureCode(item) for item in raw)
    except ValueError as error:
        raise PublicationValidationError("published failure codes are unknown") from error
    if len(codes) != len(set(codes)):
        raise PublicationValidationError("published failure codes contain duplicates")
    if status == "passed":
        if codes or evidence.get("failure_codes") not in (None, []):
            raise PublicationValidationError("passing publication contains failure codes")
    elif (
        not codes
        or evidence.get("failure_codes") != raw
        or any(exit_code_for_failure(code) is not ExitCode.QUALIFICATION_FAILED for code in codes)
    ):
        raise PublicationValidationError("failed publication codes are not domain failures")
    return codes


def _schemas(evidence: dict[str, Any], results: dict[str, Any]) -> None:
    if evidence.get("schema_version") != "upgradeguard.dev/remote-evidence/v1":
        raise PublicationValidationError("remote evidence schema version is unsupported")
    if results.get("schema_version") != "upgradeguard.dev/published-result-table/v1":
        raise PublicationValidationError("published result schema version is unsupported")


def _generated_artifacts(root: Path, evidence: dict[str, Any]) -> None:
    generated = evidence.get("generated_artifacts")
    if not isinstance(generated, dict):
        raise PublicationValidationError("generated artifact inventory is missing")
    for name in ("results.json", "report-model.json", "report.md"):
        value = generated.get(name)
        path = _regular_file(root, name)
        if (
            not isinstance(value, dict)
            or value.get("sha256") != sha256_file(path)
            or value.get("bytes") != path.stat().st_size
        ):
            raise PublicationValidationError(f"generated artifact identity differs: {name}")


def _artifact_reference(
    root: Path,
    reference: ArtifactReference | None,
    *,
    expected_path: str | None = None,
) -> None:
    if reference is None:
        raise PublicationValidationError("publication lacks a required artifact reference")
    if expected_path is not None and reference.path != expected_path:
        raise PublicationValidationError("publication artifact path differs")
    path = _regular_file(root, reference.path)
    if reference.sha256 != sha256_file(path) or reference.bytes != path.stat().st_size:
        raise PublicationValidationError(f"publication artifact hash differs: {reference.path}")


def _inventory(root: Path, evidence: dict[str, Any]) -> None:
    inventory = evidence.get("artifacts")
    if not isinstance(inventory, dict) or not inventory:
        raise PublicationValidationError("evidence artifact inventory is missing")
    excluded = {"evidence.json", "results.json", "report-model.json", "report.md"}
    expected: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise PublicationValidationError("publication contains a symlink")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise PublicationValidationError("publication contains a special filesystem node")
        relative = path.relative_to(root)
        value = relative.as_posix()
        if (
            value not in excluded
            and relative.parts[0] not in {"stale", "diagnostics"}
            and value not in POST_PUBLICATION_ARTIFACTS
        ):
            expected.add(value)
    if set(inventory) != expected:
        raise PublicationValidationError("evidence artifact inventory is incomplete")
    for relative, identity in inventory.items():
        if not isinstance(relative, str) or not _safe_relative(relative):
            raise PublicationValidationError("evidence inventory path is unsafe")
        path = _regular_file(root, relative)
        if (
            not isinstance(identity, dict)
            or identity.get("sha256") != sha256_file(path)
            or identity.get("bytes") != path.stat().st_size
        ):
            raise PublicationValidationError(f"evidence inventory differs: {relative}")


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        value
        and not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and "\\" not in value
        and "\x00" not in value
    )


def _provenance(
    root: Path,
    evidence: dict[str, Any],
    results: dict[str, Any],
    report: ReportModel,
) -> None:
    try:
        matrix = MatrixLock.model_validate_json(
            _regular_file(root, "matrix.lock.json").read_text(encoding="utf-8")
        )
        reference = ReferenceEnvironmentLock.model_validate_json(
            _regular_file(root, "reference-environment.lock.json").read_text(encoding="utf-8")
        )
        source = _regular_file(root, "source.commit").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise PublicationValidationError("publication provenance lock is invalid") from error
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise PublicationValidationError("matrix lock self-hash differs")
    if reference.computed_sha256() != reference.lock_sha256:
        raise PublicationValidationError("reference environment lock self-hash differs")
    expected = {
        "source_git_commit": source,
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "reference_environment_lock_sha256": reference.lock_sha256,
    }
    for name, value in expected.items():
        if evidence.get(name) != value or results.get(name) != value:
            raise PublicationValidationError(f"publication provenance differs: {name}")
        if getattr(report, name) != value:
            raise PublicationValidationError(f"report provenance differs: {name}")
    images = {
        environment.id: environment.worker_image.canonical_reference
        for environment in matrix.environments
    }
    if (
        evidence.get("environment_images") != images
        or results.get("environment_images") != images
        or report.environment_images != images
    ):
        raise PublicationValidationError("publication worker image provenance differs")


def _gates(
    root: Path,
    evidence: dict[str, Any],
    results: dict[str, Any],
    report: ReportModel,
    *,
    status: Literal["passed", "failed"],
) -> None:
    result_status = results.get("gate_status")
    evidence_status = evidence.get("gate_status")
    report_status = {name: value.value for name, value in report.acceptance_gates.items()}
    failure_step = results.get("failure_step") if status == "failed" else None
    try:
        expected_steps = expected_publication_steps(status, failure_step=failure_step)
    except ValueError as error:
        raise PublicationValidationError("publication gate prefix is invalid") from error
    result_sequence = results.get("gate_sequence")
    evidence_sequence = evidence.get("gate_sequence")
    if (
        not isinstance(result_status, dict)
        or result_status != evidence_status
        or result_status != report_status
        or result_sequence != list(expected_steps)
        or evidence_sequence != list(expected_steps)
        or set(result_status) != set(expected_steps)
        or set(report_status) != set(expected_steps)
    ):
        raise PublicationValidationError("publication gate statuses differ")
    allowed = {"passed"} if status == "passed" else {"passed", "failed"}
    if not result_status or any(value not in allowed for value in result_status.values()):
        raise PublicationValidationError("publication gate status is invalid")
    if status == "failed" and list(result_status.values()).count("failed") != 1:
        raise PublicationValidationError("failed publication must have one terminal failed gate")
    gate_evidence = results.get("gate_evidence")
    if (
        not isinstance(gate_evidence, dict)
        or gate_evidence != evidence.get("gate_evidence")
        or set(gate_evidence) != set(expected_steps)
    ):
        raise PublicationValidationError("publication gate evidence differs")
    for name, outcome in result_status.items():
        value = gate_evidence.get(name)
        marker = _regular_file(root, f"done/{name}.json")
        if (
            not isinstance(value, dict)
            or value.get("status") != outcome
            or value.get("marker_sha256") != sha256_file(marker)
        ):
            raise PublicationValidationError(f"publication gate marker differs: {name}")
    _validate_marker_chain(
        root,
        expected_steps,
        result_status,
        source_git_commit=results.get("source_git_commit"),
        gpu_uuid=results.get("gpu_uuid"),
        matrix_lock_sha256=results.get("matrix_lock_sha256"),
        failure_step=failure_step,
    )


def _validate_marker_chain(
    root: Path,
    steps: tuple[str, ...],
    outcomes: dict[str, Any],
    *,
    source_git_commit: Any,
    gpu_uuid: Any,
    matrix_lock_sha256: Any,
    failure_step: Any,
) -> None:
    if not all(isinstance(item, str) and item for item in (source_git_commit, gpu_uuid)):
        raise PublicationValidationError("publication marker lineage identity is invalid")
    corpus_identities = _retained_corpus_identities(root)
    qualification_lineage = _retained_qualification_lineage(root)
    for step in steps:
        marker_path = _regular_file(root, f"done/{step}.json")
        marker = _json_object(marker_path)
        dependencies = direct_step_dependencies(step, failure_step=failure_step)
        expected_dependencies = {
            dependency: sha256_file(_regular_file(root, f"done/{dependency}.json"))
            for dependency in dependencies
        }
        matrix_bound = step == "matrix-lock" or step_is_bound_to(
            step,
            "matrix-lock",
            failure_step=failure_step,
        )
        corpus_bound = step == "corpus-materialization" or step_is_bound_to(
            step,
            "corpus-materialization",
            failure_step=failure_step,
        )
        expected = {
            "schema_version": MARKER_SCHEMA,
            "step": step,
            "source_git_commit": source_git_commit,
            "gpu_uuid": gpu_uuid,
            "mode": "full",
            "outcome": outcomes[step],
            "inventory": _retained_step_inventory(root, step),
            "direct_dependency_marker_sha256s": expected_dependencies,
            "matrix_lock_sha256": matrix_lock_sha256 if matrix_bound else None,
            "corpus_identities": corpus_identities if corpus_bound else [],
            "qualification_spec_lineage": qualification_lineage if matrix_bound else None,
        }
        if marker != expected:
            raise PublicationValidationError(f"publication marker lineage differs: {step}")


def _retained_step_inventory(root: Path, step: str) -> list[dict[str, int | str]]:
    entries: dict[str, dict[str, int | str]] = {}
    try:
        owned = STEP_OWNED_PATHS[step]
    except KeyError as error:
        raise PublicationValidationError(
            f"publication marker owns an unknown step: {step}"
        ) from error
    for authored in owned:
        relative = authored.removesuffix("/")
        path = root / relative
        if authored.endswith("/"):
            _inventory_retained_directory(root, path, entries)
        else:
            _inventory_retained_file(root, relative, entries)
    aliases = sorted(alias for alias, target in STEP_ALIASES.items() if target == step)
    log_names = (step, *aliases)
    observed_logs = [
        f"logs/{name}.log" for name in log_names if (root / f"logs/{name}.log").exists()
    ]
    if not observed_logs:
        raise PublicationValidationError(f"publication marker lacks a closed step log: {step}")
    for relative in observed_logs:
        _inventory_retained_file(root, relative, entries)
    return [entries[name] for name in sorted(entries)]


def _inventory_retained_directory(
    root: Path,
    path: Path,
    entries: dict[str, dict[str, int | str]],
) -> None:
    relative = path.relative_to(root).as_posix()
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise PublicationValidationError(
            f"publication marker directory is absent: {relative}"
        ) from error
    if not stat.S_ISDIR(mode):
        raise PublicationValidationError(f"publication marker directory is unsafe: {relative}")
    before = len(entries)
    for current_root, directory_names, file_names in os.walk(path, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            child = current / name
            if not stat.S_ISDIR(child.lstat().st_mode):
                child_relative = child.relative_to(root).as_posix()
                raise PublicationValidationError(
                    f"publication marker directory is unsafe: {child_relative}"
                )
        for name in file_names:
            _inventory_retained_file(root, (current / name).relative_to(root).as_posix(), entries)
    if len(entries) == before:
        raise PublicationValidationError(f"publication marker directory is empty: {relative}")


def _inventory_retained_file(
    root: Path,
    relative: str,
    entries: dict[str, dict[str, int | str]],
) -> None:
    path = _regular_file(root, relative)
    if relative in entries:
        raise PublicationValidationError(f"publication marker inventory duplicates: {relative}")
    entries[relative] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _retained_qualification_lineage(root: Path) -> dict[str, str]:
    locked_sha256 = sha256_file(_regular_file(root, "full.yaml"))
    marker = _json_object(_regular_file(root, "done/matrix-lock.json"))
    lineage = marker.get("qualification_spec_lineage")
    if not isinstance(lineage, dict) or set(lineage) != {
        "resolved_path",
        "source_sha256",
        "locked_sha256",
    }:
        raise PublicationValidationError("publication qualification lineage is invalid")
    resolved = lineage.get("resolved_path")
    if (
        not isinstance(resolved, str)
        or not Path(resolved).is_absolute()
        or not _sha256_string(lineage.get("source_sha256"))
        or lineage.get("locked_sha256") != locked_sha256
    ):
        raise PublicationValidationError("publication qualification lineage differs")
    return {name: str(value) for name, value in lineage.items()}


def _retained_corpus_identities(root: Path) -> list[dict[str, str]]:
    value = _json_object(_regular_file(root, "corpora.json"))
    if (
        set(value) != {"schema_version", "reference_environment_sha256", "corpora"}
        or value.get("schema_version") != "upgradeguard.dev/corpus-index/v1"
    ):
        raise PublicationValidationError("publication corpus index is invalid")
    reference = value.get("reference_environment_sha256")
    corpora = value.get("corpora")
    if (
        not _sha256_string(reference)
        or not isinstance(corpora, dict)
        or set(corpora)
        != {
            "core",
            "plugin",
            "mobilenet",
        }
    ):
        raise PublicationValidationError("publication corpus index is incomplete")
    fields = {
        "root",
        "lock",
        "lock_sha256",
        "materializer_sha256",
        "inventory_sha256",
        "reference_environment_sha256",
    }
    identities: list[dict[str, str]] = []
    for kind, raw in sorted(corpora.items()):
        if not isinstance(raw, dict) or set(raw) != fields:
            raise PublicationValidationError(f"publication corpus identity is invalid: {kind}")
        root_path = PurePosixPath(str(raw.get("root")))
        lock_path = PurePosixPath(str(raw.get("lock")))
        hashes = (
            raw.get("lock_sha256"),
            raw.get("materializer_sha256"),
            raw.get("inventory_sha256"),
            raw.get("reference_environment_sha256"),
        )
        if (
            not _safe_posix_path(root_path)
            or not _safe_posix_path(lock_path)
            or not lock_path.is_relative_to(root_path)
            or not all(_sha256_string(item) for item in hashes)
            or raw.get("reference_environment_sha256") != reference
        ):
            raise PublicationValidationError(f"publication corpus identity differs: {kind}")
        identities.append({"kind": kind, **{name: str(raw[name]) for name in sorted(fields)}})
    return identities


def _safe_posix_path(path: PurePosixPath) -> bool:
    return (
        not path.is_absolute()
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _sha256_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    prefix, separator, digest = value.partition(":")
    return (
        prefix == "sha256"
        and separator == ":"
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _core_qualification(
    root: Path,
    evidence: dict[str, Any],
    results: dict[str, Any],
    *,
    status: Literal["passed", "failed"],
    codes: tuple[FailureCode, ...],
) -> dict[str, Any]:
    core = _json_object(_regular_file(root, "core-run/qualification-summary.json"))
    if results.get("core_qualification") != core or evidence.get("core_qualification") != core:
        raise PublicationValidationError("published result and core qualification differ")
    if core.get("schema_version") != "upgradeguard.dev/qualification-summary/v1":
        raise PublicationValidationError("core qualification schema version is unsupported")
    raw_codes = core.get("failure_codes")
    if not isinstance(raw_codes, list) or not all(isinstance(item, str) for item in raw_codes):
        raise PublicationValidationError("core qualification failure codes are invalid")
    try:
        core_codes = tuple(FailureCode(item) for item in raw_codes)
    except ValueError as error:
        raise PublicationValidationError("core qualification failure code is unknown") from error
    if len(core_codes) != len(set(core_codes)) or core.get("status") != _summary_status(core_codes):
        raise PublicationValidationError("core qualification status and failure codes differ")
    failed_step = results.get("failure_step")
    if status == "passed":
        if core_codes or core["status"] != "passed":
            raise PublicationValidationError("passing publication has a non-passing core gate")
    elif failed_step == "core-qualification":
        if core["status"] != "failed" or core_codes != codes:
            raise PublicationValidationError("failed core gate differs from publication failure")
    elif failed_step in {"plugin-matrix", "mobilenet-matrix"}:
        if core_codes or core["status"] != "passed":
            raise PublicationValidationError("later failed gate has a non-passing core gate")
    else:
        raise PublicationValidationError("failed publication step is invalid")
    return core


def _summary_status(codes: tuple[FailureCode, ...]) -> str:
    if not codes:
        return "passed"
    if FailureCode.INFRASTRUCTURE_INVALID in codes:
        return "infrastructure_invalid"
    if FailureCode.INCONCLUSIVE in codes:
        return "inconclusive"
    return "failed"


def _failure_payload(
    root: Path,
    evidence: dict[str, Any],
    results: dict[str, Any],
    report: ReportModel,
    *,
    status: Literal["passed", "failed"],
    codes: tuple[FailureCode, ...],
) -> None:
    raw_failures = results.get("failures", [])
    if not isinstance(raw_failures, list) or evidence.get("failures", []) != raw_failures:
        raise PublicationValidationError("published failures are invalid")
    try:
        failures = tuple(FailureRecord.model_validate(value) for value in raw_failures)
    except ValidationError as error:
        raise PublicationValidationError("published failure record is invalid") from error
    if tuple(report.failures) != failures:
        raise PublicationValidationError("report failure records differ from results")
    if status == "passed":
        if failures:
            raise PublicationValidationError("passing publication contains failures")
        return
    if not failures or {failure.code for failure in failures} != set(codes):
        raise PublicationValidationError("failed publication lacks exact typed failure records")
    failed_gates = [name for name, value in results["gate_status"].items() if value == "failed"]
    failure_roots = {
        "core-qualification": "core-run",
        "plugin-matrix": "plugin-runs",
        "mobilenet-matrix": "mobilenet-runs",
    }
    try:
        failure_root = (root / failure_roots[failed_gates[0]]).resolve(strict=True)
    except (IndexError, KeyError, OSError) as error:
        raise PublicationValidationError("failed publication has no typed failure root") from error
    if not failure_root.is_relative_to(root):
        raise PublicationValidationError("typed failure root escaped the publication")
    for failure in failures:
        if not failure.evidence:
            raise PublicationValidationError("typed failure record lacks evidence")
        for reference in failure.evidence:
            _artifact_reference(failure_root, reference)
    if (
        evidence.get("failure_step") != failed_gates[0]
        or results.get("failure_step") != failed_gates[0]
    ):
        raise PublicationValidationError("failed publication step differs")
    failure_evidence = results.get("failure_evidence")
    if not isinstance(failure_evidence, dict):
        raise PublicationValidationError("failed publication evidence is missing")
    try:
        artifact = ArtifactReference.model_validate(failure_evidence.get("artifact"))
    except ValidationError as error:
        raise PublicationValidationError("failed publication artifact is invalid") from error
    _artifact_reference(root, artifact)
    if failure_evidence.get("value") != _json_object(root / artifact.path):
        raise PublicationValidationError("failed publication artifact value differs")
    _failure_disposition(root, evidence, results, report, failures, artifact)


def _failure_disposition(
    root: Path,
    evidence: dict[str, Any],
    results: dict[str, Any],
    report: ReportModel,
    failures: tuple[FailureRecord, ...],
    failure_artifact: ArtifactReference,
) -> None:
    from upgrade_guard.errors import InfrastructureError
    from upgrade_guard.reduce.public_failure import validate_public_failure_disposition

    disposition_path = _regular_file(root, "public-failure/disposition.json")
    try:
        disposition = validate_public_failure_disposition(
            disposition_path.parent,
            state=root,
        )
    except (OSError, ValueError, ValidationError, InfrastructureError) as error:
        raise PublicationValidationError("public failure disposition is invalid") from error
    disposition_artifact = ArtifactReference(
        path="public-failure/disposition.json",
        sha256=sha256_file(disposition_path),
        bytes=disposition_path.stat().st_size,
        media_type="application/json",
    )
    if (
        disposition.source_artifact != failure_artifact
        or tuple(item.failure for item in disposition.items) != failures
        or not any(reference == disposition_artifact for reference in report.evidence)
    ):
        raise PublicationValidationError("public failure disposition identity differs")
    reduced = tuple(item for item in disposition.items if item.disposition == "reduced_replayed")
    reason = (
        None
        if reduced
        else "; ".join(
            sorted(
                {
                    item.reason
                    for item in disposition.items
                    if item.disposition == "not_applicable" and item.reason is not None
                }
            )
        )
    )
    expected = {
        "status": "passed" if reduced else "not_applicable",
        "reason": reason,
        "disposition": disposition_artifact.model_dump(mode="json"),
        "disposition_sha256": disposition.disposition_sha256,
        "items": [item.model_dump(mode="json") for item in disposition.items],
    }
    for name in ("reduction", "reproduction"):
        if results.get(name) != expected or evidence.get(name) != expected:
            raise PublicationValidationError(f"failed publication {name} disposition differs")
