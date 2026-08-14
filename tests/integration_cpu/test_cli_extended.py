"""Additional public CLI dispatch coverage without Docker or GPU access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.factories import (
    FIXED_TIME,
    digest,
    environment_lock,
    reference_environment_lock,
    supported_doctor,
)
from upgrade_guard.cli import app
from upgrade_guard.contracts.doctor import DoctorIssue
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode, InvalidInputError
from upgrade_guard.qualification import QualificationOutcome

runner = CliRunner()


def _qualification_file(root: Path) -> Path:
    path = root / "qualification.yaml"
    path.write_text("kind: Qualification\n", encoding="utf-8")
    return path


def _matrix_lock() -> MatrixLock:
    lock = MatrixLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="EnvironmentLock",
        source_matrix_sha256=digest("1"),
        gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        created_at=FIXED_TIME,
        environments=(
            environment_lock(environment_id="baseline", worker_manifest_character="2"),
            environment_lock(environment_id="candidate", worker_manifest_character="3"),
        ),
        lock_sha256=digest("4"),
    )
    return lock.model_copy(update={"lock_sha256": lock.computed_sha256()})


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("unsupported", 3),
        ("inconclusive", 4),
        ("infrastructure_invalid", 4),
    ],
)
def test_qualify_preserves_nonresult_exit_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    exit_code: int,
) -> None:
    specification = _qualification_file(tmp_path)

    class StatusRunner:
        def run(self, qualification: Path, output: Path, **kwargs: object) -> QualificationOutcome:
            del qualification, kwargs
            return QualificationOutcome(output, status, ())  # type: ignore[arg-type]

    monkeypatch.setattr("upgrade_guard.orchestrator.FullQualificationRunner", StatusRunner)
    result = runner.invoke(
        app,
        ["qualify", str(specification), "--out", str(tmp_path / "run"), "--json"],
    )

    assert result.exit_code == exit_code
    assert json.loads(result.stdout)["status"] == status


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("passed", 0),
        ("failed", 1),
        ("unsupported", 3),
        ("inconclusive", 4),
        ("infrastructure_invalid", 4),
    ],
)
def test_hidden_core_qualification_preserves_exit_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    exit_code: int,
) -> None:
    specification = _qualification_file(tmp_path)
    failure_codes = (FailureCode.NUMERICAL_REGRESSION,) if status == "failed" else ()

    class StatusRunner:
        def __init__(self, *, source_root: Path) -> None:
            assert source_root == tmp_path

        def run(self, qualification: Path, output: Path) -> QualificationOutcome:
            assert qualification == specification
            return QualificationOutcome(output, status, failure_codes)  # type: ignore[arg-type]

    monkeypatch.setattr("upgrade_guard.qualification.QualificationRunner", StatusRunner)
    result = runner.invoke(
        app,
        [
            "dev",
            "qualify-core",
            str(specification),
            "--out",
            str(tmp_path / "core"),
            "--project-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == exit_code
    payload = json.loads(result.stdout)
    assert payload["status"] == status
    assert payload["failure_codes"] == (["NUMERICAL_REGRESSION"] if status == "failed" else [])


def test_hidden_core_qualification_maps_corpus_invalid_to_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specification = _qualification_file(tmp_path)

    class CorpusInvalidRunner:
        def __init__(self, *, source_root: Path) -> None:
            assert source_root == tmp_path

        def run(self, qualification: Path, output: Path) -> QualificationOutcome:
            assert qualification == specification
            return QualificationOutcome(
                output,
                "failed",
                (FailureCode.CORPUS_INVALID,),
            )

    monkeypatch.setattr(
        "upgrade_guard.qualification.QualificationRunner",
        CorpusInvalidRunner,
    )
    result = runner.invoke(
        app,
        [
            "dev",
            "qualify-core",
            str(specification),
            "--out",
            str(tmp_path / "core"),
            "--project-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["failure_codes"] == ["CORPUS_INVALID"]


def test_hidden_core_qualification_human_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    specification = _qualification_file(tmp_path)

    class PassingRunner:
        def __init__(self, *, source_root: Path) -> None:
            assert source_root == tmp_path

        def run(self, qualification: Path, output: Path) -> QualificationOutcome:
            assert qualification == specification
            return QualificationOutcome(output, "passed", ())

    monkeypatch.setattr("upgrade_guard.qualification.QualificationRunner", PassingRunner)
    result = runner.invoke(
        app,
        [
            "dev",
            "qualify-core",
            str(specification),
            "--out",
            str(tmp_path / "core"),
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Core qualification passed" in result.stdout


@pytest.mark.parametrize("json_output", [False, True])
def test_matrix_verify_dispatches_valid_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    json_output: bool,
) -> None:
    lock = _matrix_lock()
    lock_path = tmp_path / "matrix.lock.json"
    lock_path.write_text(lock.model_dump_json(indent=2), encoding="utf-8")
    observed: list[MatrixLock] = []

    class VerifyingLocker:
        def verify(self, value: MatrixLock) -> None:
            observed.append(value)

    monkeypatch.setattr("upgrade_guard.cli.MatrixLocker", VerifyingLocker)
    arguments = ["matrix", "verify", str(lock_path)]
    if json_output:
        arguments.append("--json")
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert observed == [lock]
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["status"] == "passed"
        assert payload["lock_sha256"] == lock.lock_sha256
    else:
        assert f"Verified current environment against lock: {lock.lock_sha256}" in result.stdout


def test_matrix_verify_rejects_malformed_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "matrix.lock.json"
    lock_path.write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["matrix", "verify", str(lock_path), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error_code"] == "INVALID_INPUT"
    assert payload["message"] == "environment lock is invalid"


@pytest.mark.parametrize("json_output", [False, True])
def test_reference_environment_lock_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    json_output: bool,
) -> None:
    lock = reference_environment_lock()
    destination = tmp_path / "reference.lock.json"

    class FakeReferenceLocker:
        def lock(self, image_reference: str, output: Path) -> object:
            assert image_reference == "registry.example/reference:v1"
            assert output == destination
            return lock

    monkeypatch.setattr(
        "upgrade_guard.reference_environment.ReferenceEnvironmentLocker",
        FakeReferenceLocker,
    )
    arguments = [
        "dev",
        "lock-reference",
        "registry.example/reference:v1",
        "--out",
        str(destination),
    ]
    if json_output:
        arguments.append("--json")
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    if json_output:
        assert json.loads(result.stdout)["lock_sha256"] == lock.lock_sha256
    else:
        assert f"Wrote independent reference lock: {destination}" in result.stdout


def test_doctor_human_issues_preserve_expected_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue = DoctorIssue(
        code="DOCKER_UNAVAILABLE",
        category="unsupported",
        message="Docker is unavailable",
    )
    result_model = supported_doctor().model_copy(
        update={"outcome": "unsupported", "issues": (issue,)}
    )
    monkeypatch.setattr("upgrade_guard.cli.run_doctor", lambda: result_model)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 3
    assert "[DOCKER_UNAVAILABLE] Docker is unavailable" in result.stdout


@pytest.mark.parametrize("json_output", [False, True])
def test_outer_guard_maps_expected_doctor_error(
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    def fail_doctor() -> object:
        raise InvalidInputError("injected doctor input error")

    monkeypatch.setattr("upgrade_guard.cli.run_doctor", fail_doctor)
    arguments = ["doctor"]
    if json_output:
        arguments.append("--json")
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    output = result.stdout if json_output else result.stderr
    assert "injected doctor input error" in output


def test_corpus_materialize_rejects_bad_reference_lock_before_generation(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("kind: CorpusRecipe\n", encoding="utf-8")
    malformed = tmp_path / "malformed.lock.json"
    malformed.write_text("{", encoding="utf-8")
    stale = tmp_path / "stale.lock.json"
    stale.write_text(
        reference_environment_lock()
        .model_copy(update={"lock_sha256": digest("f")})
        .model_dump_json(indent=2),
        encoding="utf-8",
    )

    malformed_result = runner.invoke(
        app,
        [
            "corpus",
            "materialize",
            str(recipe),
            "--out",
            str(tmp_path / "malformed-output"),
            "--reference-lock",
            str(malformed),
            "--json",
        ],
    )
    stale_result = runner.invoke(
        app,
        [
            "corpus",
            "materialize",
            str(recipe),
            "--out",
            str(tmp_path / "stale-output"),
            "--reference-lock",
            str(stale),
            "--json",
        ],
    )

    assert malformed_result.exit_code == 2
    assert json.loads(malformed_result.stdout)["message"] == (
        "reference environment lock is invalid"
    )
    assert stale_result.exit_code == 2
    assert json.loads(stale_result.stdout)["message"] == (
        "reference environment lock self-hash differs"
    )


def test_expected_command_errors_and_json_reduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stored = tmp_path / "stored"
    stored.mkdir()
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"fixture")

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise InvalidInputError("injected expected error")

    monkeypatch.setattr("upgrade_guard.qualification.compare_stored_run", fail)
    compared = runner.invoke(app, ["compare", str(stored), "--json"])

    monkeypatch.setattr(
        "upgrade_guard.reduce.session.reduce_failure_directory",
        lambda *args, **kwargs: {"failure_code": "NUMERICAL_REGRESSION", "status": "passed"},
    )
    reduced = runner.invoke(
        app,
        ["reduce", str(stored), "--out", str(tmp_path / "reduced"), "--json"],
    )
    monkeypatch.setattr("upgrade_guard.reduce.session.reduce_failure_directory", fail)
    reduction_error = runner.invoke(
        app,
        ["reduce", str(stored), "--out", str(tmp_path / "error"), "--json"],
    )

    monkeypatch.setattr("upgrade_guard.cli.parse_image_reference", fail)
    resolved = runner.invoke(app, ["dev", "resolve-image", "bad", "--json"])
    monkeypatch.setattr("upgrade_guard.cli.verify_bundle", fail)
    verified = runner.invoke(app, ["reproduce", "verify", str(bundle), "--json"])
    monkeypatch.setattr("upgrade_guard.cli.observe_replay_target", fail)
    replayed = runner.invoke(
        app,
        ["reproduce", "run", str(bundle), "--out", str(tmp_path / "replay"), "--json"],
    )

    for result in (compared, reduction_error, resolved, verified, replayed):
        assert result.exit_code == 2
        assert json.loads(result.stdout)["message"] == "injected expected error"
    assert reduced.exit_code == 0
    assert json.loads(reduced.stdout)["status"] == "passed"


def test_matrix_lock_and_image_resolver_hide_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("{}\n", encoding="utf-8")

    class CrashingLocker:
        def lock(self, matrix_path: Path, output: Path) -> object:
            del matrix_path, output
            raise RuntimeError("private matrix failure")

    monkeypatch.setattr("upgrade_guard.cli.MatrixLocker", CrashingLocker)
    locked = runner.invoke(app, ["matrix", "lock", str(matrix), "--json"])

    def crash(reference: str) -> object:
        del reference
        raise RuntimeError("private resolver failure")

    monkeypatch.setattr("upgrade_guard.cli.parse_image_reference", crash)
    resolved = runner.invoke(app, ["dev", "resolve-image", "bad", "--json"])

    for result in (locked, resolved):
        assert result.exit_code == 5
        assert json.loads(result.stdout)["message"] == "unexpected internal tool failure"
        assert "private" not in result.stdout


def test_report_rejects_malformed_model(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "report-model.json").write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["report", str(run)])

    assert result.exit_code == 2
    assert "run directory does not contain a valid report-model.json" in result.stderr


def test_doctor_infrastructure_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    result_model = supported_doctor().model_copy(update={"outcome": "infrastructure_invalid"})
    monkeypatch.setattr("upgrade_guard.cli.run_doctor", lambda: result_model)

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 4
    assert json.loads(result.stdout)["outcome"] == "infrastructure_invalid"
