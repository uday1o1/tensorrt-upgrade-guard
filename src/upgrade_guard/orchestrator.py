"""Public full-V1 qualification orchestration."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import yaml

from upgrade_guard.errors import (
    FailureCode,
    InfrastructureError,
    InvalidInputError,
    UpgradeGuardError,
)
from upgrade_guard.publication import (
    PublicationDecision,
    PublicationValidationError,
    validate_publication,
)
from upgrade_guard.qualification import QualificationOutcome


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Bounded public-orchestrator process result."""

    returncode: int
    stdout: str
    stderr: str


class ProcessExecutor(Protocol):
    """Injectable full-run process boundary."""

    def execute(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        capture_output: bool,
    ) -> ProcessResult: ...


class PublicationValidator(Protocol):
    """Validate one completed publication independently of process status."""

    def __call__(
        self,
        directory: Path,
        *,
        expected_status: Literal["passed", "failed"] | None = None,
    ) -> PublicationDecision: ...


class SubprocessExecutor:
    """Execute the trusted checked-in runner without shell interpretation."""

    def execute(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        capture_output: bool,
    ) -> ProcessResult:
        output_target = subprocess.DEVNULL if capture_output else None
        try:
            completed = subprocess.run(  # noqa: S603 - validated authored command array
                tuple(command),
                cwd=cwd,
                env=env,
                check=False,
                stdout=output_target,
                stderr=output_target,
                text=True,
                shell=False,
            )
        except OSError as error:
            raise InfrastructureError("could not launch the full qualification runner") from error
        return ProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


class FullQualificationRunner:
    """Run every V1 gate through the single resumable production workflow."""

    def __init__(
        self,
        executor: ProcessExecutor | None = None,
        publication_validator: PublicationValidator | None = None,
    ) -> None:
        self.executor = executor or SubprocessExecutor()
        self.publication_validator = publication_validator or validate_publication

    def run(
        self,
        specification_path: Path,
        destination: Path,
        *,
        project_root: Path | None = None,
        matrix_path: Path | None = None,
        capture_output: bool = False,
    ) -> QualificationOutcome:
        """Validate authored roots, execute or resume, and return the stored outcome."""

        specification = specification_path.resolve(strict=True)
        project = _project_root(specification, project_root)
        runner = project / "scripts" / "run_cuda_pm_qualification.sh"
        if not runner.is_file() or runner.is_symlink():
            raise InvalidInputError("project root lacks the trusted qualification runner")
        selected_matrix = _matrix_path(specification, project, matrix_path)
        output = _safe_output(destination)
        environment = os.environ.copy()
        environment.update(
            {
                "UG_PUBLIC_ORCHESTRATOR": "1",
                "UG_QUALIFICATION_SPEC": str(specification),
                "UG_MATRIX_TEMPLATE": str(selected_matrix),
                "UG_STATE_ROOT": str(output),
            }
        )
        result = self.executor.execute(
            ("bash", str(runner)),
            cwd=project,
            env=environment,
            capture_output=capture_output,
        )
        if result.returncode == 0:
            self._publication(output, expected_status="passed")
            return QualificationOutcome(output, "passed", ())
        if result.returncode == 3:
            return QualificationOutcome(
                output,
                "unsupported",
                (FailureCode.PREFLIGHT_UNSUPPORTED,),
            )
        if result.returncode == 4:
            return _stored_nonpassing_outcome(output)
        if result.returncode == 1:
            published_codes = self._publication(
                output,
                expected_status="failed",
            ).failure_codes
            stored_codes = _stored_failure_codes(output, fallback=())
            if not stored_codes or stored_codes != published_codes:
                raise InfrastructureError(
                    "failed publication differs from its typed qualification artifact"
                )
            outcome = _classified_outcome(output, published_codes)
            if outcome.status != "failed":
                raise InfrastructureError("exit 1 did not retain a domain qualification failure")
            return outcome
        if result.returncode == 2:
            raise InvalidInputError("the full qualification runner rejected its authored inputs")
        raise UpgradeGuardError(
            "the full qualification runner failed internally",
            details={"exit_code": result.returncode},
        )

    def _publication(
        self,
        output: Path,
        *,
        expected_status: Literal["passed", "failed"],
    ) -> PublicationDecision:
        try:
            return self.publication_validator(output, expected_status=expected_status)
        except PublicationValidationError as error:
            raise InfrastructureError(
                "qualification exited without a valid complete publication"
            ) from error


def _project_root(specification: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve(strict=True)
        if not candidate.is_dir() or candidate.is_symlink():
            raise InvalidInputError("project root must be a real directory")
        return _validate_project(candidate)
    for candidate in (specification.parent, *specification.parents):
        if (candidate / "BUILD_PLAN.md").is_file() and (
            candidate / "scripts" / "run_cuda_pm_qualification.sh"
        ).is_file():
            return _validate_project(candidate)
    raise InvalidInputError(
        "could not locate the project root from the qualification specification; "
        "pass --project-root"
    )


def _validate_project(project: Path) -> Path:
    required = (
        "BUILD_PLAN.md",
        "pyproject.toml",
        "uv.lock",
        "CMakeLists.txt",
        "qualification",
        "corpus",
        "containers",
        "scripts",
        "src",
    )
    missing = [name for name in required if not (project / name).exists()]
    if missing:
        raise InvalidInputError(
            "project root is incomplete",
            details={"missing": missing},
        )
    return project


def _matrix_path(specification: Path, project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        selected = explicit.resolve(strict=True)
    else:
        try:
            value = yaml.safe_load(specification.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise InvalidInputError("qualification specification is invalid") from error
        authored = value.get("environment_matrix") if isinstance(value, dict) else None
        if not isinstance(authored, str) or not authored:
            raise InvalidInputError("full qualification requires environment_matrix or --matrix")
        selected = (specification.parent / authored).resolve(strict=True)
    if not selected.is_file() or selected.is_symlink():
        raise InvalidInputError("environment matrix must be a real file")
    if not selected.is_relative_to(project):
        raise InvalidInputError("environment matrix must be inside the project root")
    return selected


def _safe_output(destination: Path) -> Path:
    output = destination.absolute()
    if output.is_symlink():
        raise InvalidInputError("qualification output cannot be a symlink")
    if output.exists() and not output.is_dir():
        raise InvalidInputError("qualification output must be a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    return parent / output.name


def _stored_failure_codes(
    output: Path,
    *,
    fallback: tuple[FailureCode, ...],
) -> tuple[FailureCode, ...]:
    observed: list[FailureCode] = []
    artifacts = (
        (output / "core-run" / "qualification-summary.json", _summary_failure_codes),
        (output / "plugin-runs" / "validation.json", _validation_failure_codes),
        (output / "mobilenet-runs" / "validation.json", _validation_failure_codes),
    )
    for path, extract in artifacts:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for code in extract(value):
            if code not in observed:
                observed.append(code)
    return tuple(observed) or fallback


def _stored_nonpassing_outcome(output: Path) -> QualificationOutcome:
    artifacts = (
        (output / "core-run" / "qualification-summary.json", _summary_failure_codes),
        (output / "plugin-runs" / "validation.json", _validation_failure_codes),
        (output / "mobilenet-runs" / "validation.json", _validation_failure_codes),
    )
    for path, extract in artifacts:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        codes = extract(value)
        if not codes:
            continue
        if FailureCode.INFRASTRUCTURE_INVALID in codes:
            expected_status = "infrastructure_invalid"
        elif FailureCode.INCONCLUSIVE in codes:
            expected_status = "inconclusive"
        else:
            raise InfrastructureError(
                "exit 4 differs from the stored non-passing qualification decision"
            )
        if value.get("status") != expected_status:
            raise InfrastructureError(
                "exit 4 differs from the stored non-passing qualification decision"
            )
        if expected_status == "infrastructure_invalid":
            return QualificationOutcome(output, "infrastructure_invalid", codes)
        return QualificationOutcome(output, "inconclusive", codes)
    return QualificationOutcome(
        output,
        "infrastructure_invalid",
        (FailureCode.INFRASTRUCTURE_INVALID,),
    )


def _summary_failure_codes(value: object) -> tuple[FailureCode, ...]:
    if not isinstance(value, dict):
        return ()
    raw = value.get("failure_codes")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        return ()
    try:
        return tuple(FailureCode(item) for item in raw)
    except ValueError:
        return ()


def _validation_failure_codes(value: object) -> tuple[FailureCode, ...]:
    if not isinstance(value, dict):
        return ()
    raw_codes: list[str] = []
    top_level = value.get("failure_code")
    if top_level is not None:
        if not isinstance(top_level, str):
            return ()
        raw_codes.append(top_level)
    failure = value.get("failure")
    if failure is not None:
        if not isinstance(failure, dict):
            return ()
        nested = failure.get("code")
        if not isinstance(nested, str):
            return ()
        raw_codes.append(nested)
    if not raw_codes or len(set(raw_codes)) != 1:
        return ()
    try:
        return (FailureCode(raw_codes[0]),)
    except ValueError:
        return ()


def _classified_outcome(
    output: Path,
    failure_codes: tuple[FailureCode, ...],
) -> QualificationOutcome:
    nonterminal = tuple(
        code
        for code in failure_codes
        if code not in {FailureCode.INCONCLUSIVE, FailureCode.INFRASTRUCTURE_INVALID}
    )
    if nonterminal:
        return QualificationOutcome(output, "failed", failure_codes)
    if FailureCode.INFRASTRUCTURE_INVALID in failure_codes:
        return QualificationOutcome(output, "infrastructure_invalid", failure_codes)
    return QualificationOutcome(output, "inconclusive", failure_codes)
