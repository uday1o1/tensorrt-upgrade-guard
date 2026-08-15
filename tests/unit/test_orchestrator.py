"""Public full-qualification orchestrator tests."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from typer.testing import CliRunner

import upgrade_guard.orchestrator as orchestrator
from upgrade_guard.classify import exit_code_for_failure
from upgrade_guard.cli import app
from upgrade_guard.errors import (
    ExitCode,
    FailureCode,
    InfrastructureError,
    InvalidInputError,
    UpgradeGuardError,
)
from upgrade_guard.orchestrator import FullQualificationRunner, ProcessResult, SubprocessExecutor
from upgrade_guard.publication import (
    PublicationDecision,
    PublicationValidationError,
)


class FakeExecutor:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.command: tuple[str, ...] = ()
        self.environment: Mapping[str, str] = {}

    def execute(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        capture_output: bool,
    ) -> ProcessResult:
        del cwd, capture_output
        self.command = tuple(command)
        self.environment = env
        output = Path(env["UG_STATE_ROOT"])
        if self.returncode == 0:
            output.mkdir(parents=True)
            (output / "evidence.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            (output / "results.json").write_text(
                json.dumps({"status": "passed", "failure_codes": []}), encoding="utf-8"
            )
            (output / "report-model.json").write_text(
                json.dumps({"status": "passed", "publication_complete": True}),
                encoding="utf-8",
            )
            (output / "report.md").write_text("# Passed\n", encoding="utf-8")
        return ProcessResult(self.returncode, "", "")


class ArtifactExecutor(FakeExecutor):
    def __init__(self, documents: Mapping[str, object], returncode: int = 1) -> None:
        super().__init__(returncode)
        self.documents = documents

    def execute(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        capture_output: bool,
    ) -> ProcessResult:
        del command, cwd, capture_output
        root = Path(env["UG_STATE_ROOT"])
        for relative, value in self.documents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        codes = _document_failure_codes(self.documents)
        if self.returncode == 1 and codes:
            (root / "evidence.json").write_text(
                json.dumps({"status": "failed", "failure_codes": codes}),
                encoding="utf-8",
            )
            (root / "results.json").write_text(
                json.dumps({"status": "failed", "failure_codes": codes}),
                encoding="utf-8",
            )
            (root / "report-model.json").write_text(
                json.dumps({"status": "failed", "publication_complete": True}),
                encoding="utf-8",
            )
            (root / "report.md").write_text("# Failed\n", encoding="utf-8")
        return ProcessResult(self.returncode, "", "")


def _document_failure_codes(documents: Mapping[str, object]) -> list[str]:
    codes: list[str] = []
    ordered = (
        "core-run/qualification-summary.json",
        "plugin-runs/validation.json",
        "mobilenet-runs/validation.json",
    )
    for relative in (*ordered, *(name for name in documents if name not in ordered)):
        value = documents.get(relative)
        if not isinstance(value, dict):
            continue
        raw: object = value.get("failure_codes") if relative.endswith("summary.json") else None
        if raw is None:
            top = value.get("failure_code")
            nested = value.get("failure")
            nested_code = nested.get("code") if isinstance(nested, dict) else None
            raw = [top or nested_code] if top is None or nested_code in {None, top} else []
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str):
                continue
            try:
                FailureCode(item)
            except ValueError:
                continue
            if item not in codes:
                codes.append(item)
    return codes


def _fixture_publication(
    output: Path,
    *,
    expected_status: Literal["passed", "failed"] | None = None,
) -> PublicationDecision:
    try:
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationValidationError("fixture publication is unavailable") from error
    if not isinstance(results, dict) or results.get("status") not in {"passed", "failed"}:
        raise PublicationValidationError("fixture publication status is invalid")
    status: Literal["passed", "failed"] = results["status"]
    if expected_status is not None and status != expected_status:
        raise PublicationValidationError("fixture publication has the wrong status")
    raw_codes = results.get("failure_codes")
    if not isinstance(raw_codes, list):
        raise PublicationValidationError("fixture failure codes are invalid")
    try:
        codes = tuple(FailureCode(item) for item in raw_codes)
    except (TypeError, ValueError) as error:
        raise PublicationValidationError("fixture failure codes are invalid") from error
    if status == "passed" and codes:
        raise PublicationValidationError("fixture passing publication has failures")
    if status == "failed" and (
        not codes
        or any(exit_code_for_failure(code) is not ExitCode.QUALIFICATION_FAILED for code in codes)
    ):
        raise PublicationValidationError("fixture failed publication has non-domain failures")
    return PublicationDecision(status, codes, {}, results)


def _runner(executor: FakeExecutor) -> FullQualificationRunner:
    return FullQualificationRunner(executor, _fixture_publication)


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    for directory in ("qualification", "matrices", "corpus", "containers", "scripts", "src"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    for name in ("BUILD_PLAN.md", "pyproject.toml", "uv.lock", "CMakeLists.txt"):
        (project / name).write_text("fixture\n", encoding="utf-8")
    runner = project / "scripts" / "run_gpu_qualification.sh"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    matrix = project / "matrices" / "pair.yaml"
    matrix.write_text("kind: EnvironmentMatrix\n", encoding="utf-8")
    specification = project / "qualification" / "full.yaml"
    specification.write_text(
        "environment_matrix: ../matrices/pair.yaml\n",
        encoding="utf-8",
    )
    return project, specification, matrix


def test_full_runner_uses_explicit_project_matrix_and_resumable_output(tmp_path: Path) -> None:
    project, specification, matrix = _project(tmp_path)
    executor = FakeExecutor()
    output = tmp_path / "result"
    outcome = _runner(executor).run(
        specification,
        output,
        project_root=project,
        capture_output=True,
    )
    assert outcome.status == "passed"
    assert executor.command == ("bash", str(project / "scripts/run_gpu_qualification.sh"))
    assert executor.environment["UG_QUALIFICATION_SPEC"] == str(specification)
    assert executor.environment["UG_MATRIX_TEMPLATE"] == str(matrix)
    assert executor.environment["UG_STATE_ROOT"] == str(output)


@pytest.mark.parametrize(
    ("returncode", "status", "failure"),
    [
        (3, "unsupported", FailureCode.PREFLIGHT_UNSUPPORTED),
        (4, "infrastructure_invalid", FailureCode.INFRASTRUCTURE_INVALID),
    ],
)
def test_full_runner_preserves_nonpassing_exit_semantics(
    tmp_path: Path,
    returncode: int,
    status: str,
    failure: FailureCode,
) -> None:
    project, specification, _ = _project(tmp_path)
    outcome = _runner(FakeExecutor(returncode)).run(
        specification,
        tmp_path / "result",
        project_root=project,
    )
    assert outcome.status == status
    assert outcome.failure_codes == (failure,)


def test_exit_one_without_completed_failed_publication_is_infrastructure_invalid(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)
    with pytest.raises(InfrastructureError, match="complete publication"):
        _runner(FakeExecutor(1)).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


def test_full_runner_requires_a_complete_project_and_authored_matrix(tmp_path: Path) -> None:
    project, specification, _ = _project(tmp_path)
    (project / "BUILD_PLAN.md").unlink()
    with pytest.raises(InvalidInputError, match="incomplete"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


def test_json_mode_discards_long_child_streams_instead_of_buffering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: Sequence[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=None, stderr=None)

    monkeypatch.setattr("upgrade_guard.orchestrator.subprocess.run", fake_run)
    result = SubprocessExecutor().execute(
        ("bash", "runner.sh"),
        cwd=tmp_path,
        env={"PATH": "/usr/bin"},
        capture_output=True,
    )

    assert result == ProcessResult(0, "", "")
    assert observed["stdout"] is subprocess.DEVNULL
    assert observed["stderr"] is subprocess.DEVNULL
    assert "capture_output" not in observed


def test_subprocess_launch_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("not exposed")

    monkeypatch.setattr("upgrade_guard.orchestrator.subprocess.run", fail)
    with pytest.raises(InfrastructureError, match="could not launch"):
        SubprocessExecutor().execute(
            ("bash", "runner.sh"), cwd=tmp_path, env={}, capture_output=False
        )


@pytest.mark.parametrize(
    ("returncode", "error_type", "message"),
    [
        (2, InvalidInputError, "rejected its authored inputs"),
        (5, UpgradeGuardError, "failed internally"),
        (7, UpgradeGuardError, "failed internally"),
    ],
)
def test_full_runner_rejects_unmapped_process_failures(
    tmp_path: Path,
    returncode: int,
    error_type: type[Exception],
    message: str,
) -> None:
    project, specification, _ = _project(tmp_path)
    with pytest.raises(error_type, match=message):
        _runner(FakeExecutor(returncode)).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


@pytest.mark.parametrize("evidence", [None, "not-json", '{"status":"failed"}'])
def test_success_exit_requires_complete_passing_publication(
    tmp_path: Path, evidence: str | None
) -> None:
    project, specification, _ = _project(tmp_path)

    class IncompleteExecutor(FakeExecutor):
        def execute(
            self,
            command: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str],
            capture_output: bool,
        ) -> ProcessResult:
            del command, cwd, capture_output
            output = Path(env["UG_STATE_ROOT"])
            output.mkdir(parents=True)
            if evidence is not None:
                (output / "evidence.json").write_text(evidence, encoding="utf-8")
                for name in ("results.json", "report-model.json", "report.md"):
                    (output / name).write_text("{}\n", encoding="utf-8")
            return ProcessResult(0, "", "")

    with pytest.raises(InfrastructureError, match="complete publication"):
        FullQualificationRunner(IncompleteExecutor()).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


def test_full_runner_discovers_project_and_accepts_explicit_internal_matrix(
    tmp_path: Path,
) -> None:
    project, specification, matrix = _project(tmp_path)
    outcome = _runner(FakeExecutor()).run(
        specification,
        tmp_path / "result",
        matrix_path=matrix,
    )
    assert outcome.status == "passed"


def test_full_runner_rejects_missing_or_external_matrix_and_unsafe_output(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)
    specification.write_text("kind: Qualification\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="requires environment_matrix"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "missing-matrix",
            project_root=project,
        )

    external = tmp_path / "external.yaml"
    external.write_text("{}\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="inside the project root"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "external-matrix",
            project_root=project,
            matrix_path=external,
        )

    output_file = tmp_path / "output-file"
    output_file.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="must be a directory"):
        _runner(FakeExecutor()).run(
            specification,
            output_file,
            project_root=project,
            matrix_path=project / "matrices" / "pair.yaml",
        )


def test_failed_run_retains_specific_stored_failure_code(tmp_path: Path) -> None:
    project, specification, _ = _project(tmp_path)

    outcome = _runner(
        ArtifactExecutor(
            {"core-run/qualification-summary.json": {"failure_codes": ["NUMERICAL_REGRESSION"]}}
        )
    ).run(
        specification,
        tmp_path / "result",
        project_root=project,
    )
    assert outcome.failure_codes == (FailureCode.NUMERICAL_REGRESSION,)


@pytest.mark.parametrize(
    "failure_codes",
    [
        ["INCONCLUSIVE"],
        ["INFRASTRUCTURE_INVALID"],
        ["NUMERICAL_REGRESSION", "INCONCLUSIVE"],
    ],
)
def test_child_exit_one_rejects_non_domain_failure_publication(
    tmp_path: Path,
    failure_codes: list[str],
) -> None:
    project, specification, _ = _project(tmp_path)
    executor = ArtifactExecutor(
        {"core-run/qualification-summary.json": {"failure_codes": failure_codes}}
    )
    with pytest.raises(InfrastructureError, match="complete publication"):
        _runner(executor).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


@pytest.mark.parametrize(
    ("relative", "payload", "status", "exit_code", "failure_code"),
    [
        (
            "plugin-runs/validation.json",
            {"failure": {"code": "NUMERICAL_REGRESSION"}},
            "failed",
            1,
            FailureCode.NUMERICAL_REGRESSION,
        ),
        (
            "mobilenet-runs/validation.json",
            {
                "failure_code": "NONDETERMINISM_REGRESSION",
                "failure": {"code": "NONDETERMINISM_REGRESSION"},
            },
            "failed",
            1,
            FailureCode.NONDETERMINISM_REGRESSION,
        ),
        (
            "plugin-runs/validation.json",
            {"status": "inconclusive", "failure_code": "INCONCLUSIVE"},
            "inconclusive",
            4,
            FailureCode.INCONCLUSIVE,
        ),
        (
            "mobilenet-runs/validation.json",
            {
                "status": "infrastructure_invalid",
                "failure": {"code": "INFRASTRUCTURE_INVALID"},
            },
            "infrastructure_invalid",
            4,
            FailureCode.INFRASTRUCTURE_INVALID,
        ),
    ],
)
def test_extended_validation_failure_drives_public_exit_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative: str,
    payload: dict[str, object],
    status: str,
    exit_code: int,
    failure_code: FailureCode,
) -> None:
    project, specification, _ = _project(tmp_path)
    qualification_runner = _runner(ArtifactExecutor({relative: payload}, returncode=exit_code))
    monkeypatch.setattr(
        "upgrade_guard.orchestrator.FullQualificationRunner",
        lambda: qualification_runner,
    )

    result = CliRunner().invoke(
        app,
        [
            "qualify",
            str(specification),
            "--out",
            str(tmp_path / "result"),
            "--project-root",
            str(project),
            "--json",
        ],
    )

    assert result.exit_code == exit_code
    response = json.loads(result.stdout)
    assert response["status"] == status
    assert response["failure_codes"] == [failure_code.value]


@pytest.mark.parametrize(
    ("status", "codes"),
    [
        ("inconclusive", ["NUMERICAL_REGRESSION", "INCONCLUSIVE"]),
        (
            "infrastructure_invalid",
            ["NUMERICAL_REGRESSION", "INFRASTRUCTURE_INVALID"],
        ),
    ],
)
def test_exit_four_preserves_stored_mixed_nonpassing_status(
    tmp_path: Path,
    status: str,
    codes: list[str],
) -> None:
    project, specification, _ = _project(tmp_path)
    outcome = _runner(
        ArtifactExecutor(
            {
                "core-run/qualification-summary.json": {
                    "status": status,
                    "failure_codes": codes,
                }
            },
            returncode=4,
        )
    ).run(specification, tmp_path / "result", project_root=project)

    assert outcome.status == status
    assert outcome.failure_codes == tuple(FailureCode(code) for code in codes)


def test_extended_failures_are_collected_in_deterministic_step_order(tmp_path: Path) -> None:
    project, specification, _ = _project(tmp_path)
    outcome = _runner(
        ArtifactExecutor(
            {
                "mobilenet-runs/validation.json": {"failure_code": "NUMERICAL_REGRESSION"},
                "plugin-runs/validation.json": {"failure_code": "NONDETERMINISM_REGRESSION"},
            }
        )
    ).run(specification, tmp_path / "result", project_root=project)

    assert outcome.status == "failed"
    assert outcome.failure_codes == (
        FailureCode.NONDETERMINISM_REGRESSION,
        FailureCode.NUMERICAL_REGRESSION,
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"failure_code": 1},
        {"failure": "INCONCLUSIVE"},
        {
            "failure_code": "INCONCLUSIVE",
            "failure": {"code": "INFRASTRUCTURE_INVALID"},
        },
        {"failure_code": "NOT_A_FAILURE_CODE"},
    ],
)
def test_malformed_extended_validation_is_not_trusted(
    tmp_path: Path,
    payload: object,
) -> None:
    project, specification, _ = _project(tmp_path)
    with pytest.raises(InfrastructureError, match="complete publication"):
        _runner(ArtifactExecutor({"plugin-runs/validation.json": payload})).run(
            specification, tmp_path / "result", project_root=project
        )


def test_subprocess_passthrough_retains_child_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: Sequence[str], **kwargs: object) -> SimpleNamespace:
        del command
        assert kwargs["stdout"] is None
        assert kwargs["stderr"] is None
        return SimpleNamespace(returncode=5, stdout="child out", stderr="child err")

    monkeypatch.setattr("upgrade_guard.orchestrator.subprocess.run", fake_run)

    result = SubprocessExecutor().execute(
        ("bash", "runner.sh"), cwd=tmp_path, env={}, capture_output=False
    )

    assert result == ProcessResult(5, "child out", "child err")


def test_full_runner_rejects_untrusted_project_and_runner_paths(tmp_path: Path) -> None:
    project, specification, _ = _project(tmp_path)
    project_file = tmp_path / "not-a-project"
    project_file.write_text("fixture\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="real directory"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "file-project-result",
            project_root=project_file,
        )

    runner = project / "scripts" / "run_gpu_qualification.sh"
    replacement = tmp_path / "replacement-runner.sh"
    replacement.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner.unlink()
    runner.symlink_to(replacement)
    with pytest.raises(InvalidInputError, match="trusted qualification runner"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "symlink-runner-result",
            project_root=project,
        )


def test_full_runner_requires_a_discoverable_project_root(tmp_path: Path) -> None:
    specification = tmp_path / "isolated" / "qualification.yaml"
    specification.parent.mkdir()
    specification.write_text("environment_matrix: matrix.yaml\n", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="could not locate the project root"):
        _runner(FakeExecutor()).run(specification, tmp_path / "result")


def test_full_runner_rejects_malformed_specification_and_directory_matrix(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)
    specification.write_text("environment_matrix: [\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="specification is invalid"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "malformed-spec-result",
            project_root=project,
        )

    with pytest.raises(InvalidInputError, match="matrix must be a real file"):
        _runner(FakeExecutor()).run(
            specification,
            tmp_path / "directory-matrix-result",
            project_root=project,
            matrix_path=project / "matrices",
        )


def test_full_runner_rejects_symlink_output(tmp_path: Path) -> None:
    project, specification, matrix = _project(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(InvalidInputError, match="output cannot be a symlink"):
        _runner(FakeExecutor()).run(
            specification,
            linked_output,
            project_root=project,
            matrix_path=matrix,
        )


def test_exit_one_rejects_publication_and_stored_code_disagreement(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)
    executor = ArtifactExecutor(
        {"core-run/qualification-summary.json": {"failure_codes": ["OUTPUT_SCHEMA_CHANGED"]}}
    )

    def mismatched_publication(
        output: Path,
        *,
        expected_status: Literal["passed", "failed"] | None = None,
    ) -> PublicationDecision:
        del output
        assert expected_status == "failed"
        return PublicationDecision(
            "failed",
            (FailureCode.NUMERICAL_REGRESSION,),
            {},
            {"status": "failed"},
        )

    with pytest.raises(InfrastructureError, match="differs from its typed"):
        FullQualificationRunner(executor, mismatched_publication).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


@pytest.mark.parametrize(
    "failure_code",
    [FailureCode.INCONCLUSIVE, FailureCode.INFRASTRUCTURE_INVALID],
)
def test_exit_one_rejects_nonfailed_stored_classification(
    tmp_path: Path,
    failure_code: FailureCode,
) -> None:
    project, specification, _ = _project(tmp_path)
    executor = ArtifactExecutor(
        {"core-run/qualification-summary.json": {"failure_codes": [failure_code.value]}}
    )

    def permissive_publication(
        output: Path,
        *,
        expected_status: Literal["passed", "failed"] | None = None,
    ) -> PublicationDecision:
        del output
        assert expected_status == "failed"
        return PublicationDecision(
            "failed",
            (failure_code,),
            {},
            {"status": "failed"},
        )

    with pytest.raises(InfrastructureError, match="did not retain a domain"):
        FullQualificationRunner(executor, permissive_publication).run(
            specification,
            tmp_path / "result",
            project_root=project,
        )


def test_stored_failure_codes_skip_unsafe_and_malformed_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    (output / "core-run").mkdir(parents=True)
    (output / "plugin-runs").mkdir()
    (output / "mobilenet-runs").mkdir()
    (output / "core-run" / "qualification-summary.json").write_text("not json", encoding="utf-8")
    unsafe_target = tmp_path / "unsafe-validation.json"
    unsafe_target.write_text(
        json.dumps({"failure_code": "OUTPUT_SCHEMA_CHANGED"}), encoding="utf-8"
    )
    (output / "plugin-runs" / "validation.json").symlink_to(unsafe_target)
    (output / "mobilenet-runs" / "validation.json").write_text(
        json.dumps({"failure_code": "NONDETERMINISM_REGRESSION"}), encoding="utf-8"
    )

    assert orchestrator._stored_failure_codes(output, fallback=()) == (
        FailureCode.NONDETERMINISM_REGRESSION,
    )


def test_stored_failure_codes_deduplicate_and_fall_back(tmp_path: Path) -> None:
    output = tmp_path / "result"
    (output / "core-run").mkdir(parents=True)
    (output / "plugin-runs").mkdir()
    (output / "core-run" / "qualification-summary.json").write_text(
        json.dumps({"failure_codes": ["NUMERICAL_REGRESSION"]}), encoding="utf-8"
    )
    (output / "plugin-runs" / "validation.json").write_text(
        json.dumps({"failure_code": "NUMERICAL_REGRESSION"}), encoding="utf-8"
    )

    assert orchestrator._stored_failure_codes(output, fallback=()) == (
        FailureCode.NUMERICAL_REGRESSION,
    )
    assert orchestrator._stored_failure_codes(
        tmp_path / "absent", fallback=(FailureCode.EXECUTION_FAILED,)
    ) == (FailureCode.EXECUTION_FAILED,)


def test_exit_four_skips_malformed_artifacts_before_typed_decision(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)

    class MalformedArtifactExecutor(FakeExecutor):
        def execute(
            self,
            command: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str],
            capture_output: bool,
        ) -> ProcessResult:
            del command, cwd, capture_output
            output = Path(env["UG_STATE_ROOT"])
            (output / "core-run").mkdir(parents=True)
            (output / "plugin-runs").mkdir()
            (output / "mobilenet-runs").mkdir()
            (output / "core-run" / "qualification-summary.json").write_text(
                "not json", encoding="utf-8"
            )
            (output / "plugin-runs" / "validation.json").write_text("[]", encoding="utf-8")
            (output / "mobilenet-runs" / "validation.json").write_text(
                json.dumps(
                    {
                        "status": "inconclusive",
                        "failure_code": "INCONCLUSIVE",
                    }
                ),
                encoding="utf-8",
            )
            return ProcessResult(4, "", "")

    outcome = _runner(MalformedArtifactExecutor(4)).run(
        specification,
        tmp_path / "result",
        project_root=project,
    )

    assert outcome.status == "inconclusive"


def test_exit_four_skips_empty_summary_before_typed_extended_decision(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)
    outcome = _runner(
        ArtifactExecutor(
            {
                "core-run/qualification-summary.json": {},
                "plugin-runs/validation.json": {
                    "status": "inconclusive",
                    "failure_code": "INCONCLUSIVE",
                },
            },
            returncode=4,
        )
    ).run(specification, tmp_path / "result", project_root=project)

    assert outcome.status == "inconclusive"
    assert outcome.failure_codes == (FailureCode.INCONCLUSIVE,)


def test_exit_four_rejects_domain_failure_or_contradictory_status(
    tmp_path: Path,
) -> None:
    project, specification, _ = _project(tmp_path)
    with pytest.raises(InfrastructureError, match="stored non-passing"):
        _runner(
            ArtifactExecutor(
                {
                    "core-run/qualification-summary.json": {
                        "status": "failed",
                        "failure_codes": ["NUMERICAL_REGRESSION"],
                    }
                },
                returncode=4,
            )
        ).run(specification, tmp_path / "domain-result", project_root=project)

    with pytest.raises(InfrastructureError, match="stored non-passing"):
        _runner(
            ArtifactExecutor(
                {
                    "core-run/qualification-summary.json": {
                        "status": "inconclusive",
                        "failure_codes": ["INFRASTRUCTURE_INVALID"],
                    }
                },
                returncode=4,
            )
        ).run(specification, tmp_path / "status-result", project_root=project)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"failure_codes": "NUMERICAL_REGRESSION"},
        {"failure_codes": [1]},
        {"failure_codes": ["NOT_A_FAILURE_CODE"]},
    ],
)
def test_summary_failure_code_parser_rejects_malformed_payloads(payload: object) -> None:
    assert orchestrator._summary_failure_codes(payload) == ()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"failure_code": 1},
        {"failure": "NUMERICAL_REGRESSION"},
        {"failure": {"code": 1}},
        {},
        {
            "failure_code": "NUMERICAL_REGRESSION",
            "failure": {"code": "OUTPUT_SCHEMA_CHANGED"},
        },
        {"failure_code": "NOT_A_FAILURE_CODE"},
    ],
)
def test_validation_failure_code_parser_rejects_malformed_payloads(payload: object) -> None:
    assert orchestrator._validation_failure_codes(payload) == ()
