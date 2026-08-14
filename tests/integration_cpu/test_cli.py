"""Public CLI behavior without importing NVIDIA Python software."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tests.factories import FIXED_TIME, digest, run_result, supported_doctor
from upgrade_guard.cli import app
from upgrade_guard.contracts.environment import PlatformIdentity, ResolvedImage
from upgrade_guard.errors import FailureCode, InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.qualification import QualificationOutcome
from upgrade_guard.report.model import build_report_model
from upgrade_guard.reproduce.run import ReplayResult

runner = CliRunner()


def test_doctor_json_reports_an_injected_supported_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upgrade_guard.cli.run_doctor", supported_doctor)
    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "upgradeguard.dev/doctor/v1"
    assert result.exit_code == 0
    assert payload["outcome"] == "supported"
    assert payload["docker"]["available"] is True
    assert payload["gpus"]
    assert not payload["issues"]
    assert "tensorrt" not in sys.modules
    assert "cuda" not in sys.modules


def test_matrix_lock_stops_at_injected_preflight_and_creates_no_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        """
api_version: upgradeguard.dev/v1alpha1
kind: EnvironmentMatrix
gpu_uuid: GPU-11111111-1111-1111-1111-111111111111
environments:
  - id: baseline
    base_image: registry.example/base:v1
    worker_image: registry.example/worker:v1
  - id: candidate
    base_image: registry.example/base:v2
    worker_image: registry.example/worker:v2
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "matrix.lock.json"

    class UnsupportedLocker:
        def lock(self, matrix: Path, output: Path) -> FakeLock:
            del matrix, output
            raise UnsupportedEnvironmentError("GPU unavailable")

    monkeypatch.setattr("upgrade_guard.cli.MatrixLocker", UnsupportedLocker)
    result = runner.invoke(
        app,
        ["matrix", "lock", str(matrix), "--out", str(output), "--json"],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "upgradeguard.dev/error/v1"
    assert payload["error_code"] == "PREFLIGHT_UNSUPPORTED"
    assert not output.exists()


def test_help_exposes_only_public_milestone_zero_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "matrix" in result.stdout


def test_doctor_human_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upgrade_guard.cli.run_doctor", supported_doctor)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Host preflight: supported" in result.stdout
    assert "NVIDIA GPUs: 1" in result.stdout


class FakeLock:
    lock_sha256 = "sha256:" + ("a" * 64)

    def model_dump_json(self, *, indent: int) -> str:
        return json.dumps({"lock_sha256": self.lock_sha256}, indent=indent)


class SuccessfulLocker:
    def lock(self, matrix: Path, output: Path) -> FakeLock:
        del matrix, output
        return FakeLock()


class FailingLocker:
    def lock(self, matrix: Path, output: Path) -> FakeLock:
        del matrix, output
        raise InvalidInputError("bad matrix")


def test_matrix_human_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("upgrade_guard.cli.MatrixLocker", SuccessfulLocker)
    result = runner.invoke(app, ["matrix", "lock", str(matrix)])
    assert result.exit_code == 0
    assert "Wrote immutable environment lock" in result.stdout
    assert FakeLock.lock_sha256 in result.stdout


def test_matrix_json_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("upgrade_guard.cli.MatrixLocker", SuccessfulLocker)
    result = runner.invoke(app, ["matrix", "lock", str(matrix), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["lock_sha256"] == FakeLock.lock_sha256


def test_matrix_human_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("upgrade_guard.cli.MatrixLocker", FailingLocker)
    result = runner.invoke(app, ["matrix", "lock", str(matrix)])
    assert result.exit_code == 2
    assert "INVALID_INPUT: bad matrix" in result.stderr


class FakeResolvedArtifact:
    image = ResolvedImage(
        authored_reference="registry.example/team/image:v1",
        registry="registry.example",
        repository="team/image",
        authored_tag="v1",
        requested_digest=None,
        index_digest=digest("1"),
        manifest_digest=digest("2"),
        config_digest=digest("3"),
        manifest_media_type="application/vnd.oci.image.manifest.v1+json",
        config_media_type="application/vnd.oci.image.config.v1+json",
        platform=PlatformIdentity(os="linux", architecture="amd64"),
    )


class FakeRegistryClient:
    def __init__(self, *, credentials: object) -> None:
        assert credentials is None

    def resolve_linux_amd64(self, reference: str) -> FakeResolvedArtifact:
        assert reference == "registry.example/team/image:v1"
        return FakeResolvedArtifact()


def test_hidden_dev_resolver_bootstraps_worker_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upgrade_guard.cli.RegistryClient", FakeRegistryClient)
    result = runner.invoke(
        app,
        ["dev", "resolve-image", "registry.example/team/image:v1", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["manifest_digest"] == digest("2")
    assert payload["config_digest"] == digest("3")


def test_corpus_materialize_cli_human_and_json(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """api_version: upgradeguard.dev/v1alpha1
kind: CorpusRecipe
id: cli-smoke
generator_version: tiny-transformer-v1
precisions: [fp32]
expected_model_sha256:
  fp32: sha256:16dd39f7df92632a0d9268b0b669ee8e110d0bce6b2da189fd046e3b4d2e71b4
transformer_shapes:
  - {id: b1_s8, batch: 1, sequence: 8, weight: 1.0}
""",
        encoding="utf-8",
    )
    first = runner.invoke(
        app, ["corpus", "materialize", str(recipe), "--out", str(tmp_path / "first")]
    )
    assert first.exit_code == 0
    assert "Materialized immutable corpus" in first.stdout
    second = runner.invoke(
        app,
        [
            "corpus",
            "materialize",
            str(recipe),
            "--out",
            str(tmp_path / "second"),
            "--json",
        ],
    )
    assert second.exit_code == 0
    assert json.loads(second.stdout)["id"] == "cli-smoke"


class StubQualificationRunner:
    status = "passed"

    def run(self, qualification: Path, output: Path) -> QualificationOutcome:
        del qualification
        failure_codes = (FailureCode.NUMERICAL_REGRESSION,) if self.status == "failed" else ()
        return QualificationOutcome(output, self.status, failure_codes)  # type: ignore[arg-type]


def test_qualify_and_compare_cli_statuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")
    monkeypatch.setattr("upgrade_guard.cli.QualificationRunner", StubQualificationRunner)
    output = tmp_path / "run"
    result = runner.invoke(app, ["qualify", str(specification), "--out", str(output), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "passed"

    StubQualificationRunner.status = "failed"
    result = runner.invoke(app, ["qualify", str(specification), "--out", str(output)])
    assert result.exit_code == 1
    assert "Qualification failed" in result.stdout
    StubQualificationRunner.status = "passed"

    output.mkdir()
    (output / "qualification-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": "passed",
                "failure_codes": [],
            }
        ),
        encoding="utf-8",
    )
    human = runner.invoke(app, ["compare", str(output)])
    machine = runner.invoke(app, ["compare", str(output), "--json"])
    assert human.exit_code == machine.exit_code == 0
    assert "Failure codes: none" in human.stdout
    assert json.loads(machine.stdout)["status"] == "passed"


@pytest.mark.parametrize(
    ("status", "failure_codes", "exit_code"),
    [
        ("passed", [], 0),
        ("failed", ["NUMERICAL_REGRESSION"], 1),
        ("inconclusive", ["INCONCLUSIVE"], 4),
        ("infrastructure_invalid", ["INFRASTRUCTURE_INVALID"], 4),
    ],
)
def test_compare_preserves_stored_status_exit_semantics(
    tmp_path: Path,
    status: str,
    failure_codes: list[str],
    exit_code: int,
) -> None:
    run = tmp_path / status
    run.mkdir()
    (run / "qualification-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": status,
                "failure_codes": failure_codes,
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["compare", str(run), "--json"])
    assert result.exit_code == exit_code
    assert json.loads(result.stdout)["status"] == status


def test_report_and_reproduce_verify_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report_directory = tmp_path / "run"
    report_directory.mkdir()
    report = build_report_model(
        title="CLI report",
        generated_at=FIXED_TIME,
        baseline_environment_id="baseline",
        candidate_environment_id="candidate",
        results=(run_result(),),
    )
    (report_directory / "report-model.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    rendered = runner.invoke(app, ["report", str(report_directory), "--format", "html"])
    assert rendered.exit_code == 0
    assert "<!doctype html>" in rendered.stdout.lower()
    invalid = runner.invoke(app, ["report", str(report_directory), "--format", "xml"])
    assert invalid.exit_code == 2

    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"fixture")
    verified = SimpleNamespace(
        manifest=SimpleNamespace(id="bundle-001", manifest_sha256=digest("9")),
        source_code_present=False,
        engine_present=False,
        observed_files=("bundle.json",),
    )
    monkeypatch.setattr("upgrade_guard.cli.verify_bundle", lambda path: verified)
    result = runner.invoke(app, ["reproduce", "verify", str(bundle), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["bundle_id"] == "bundle-001"

    human = runner.invoke(app, ["reproduce", "verify", str(bundle)])
    assert human.exit_code == 0
    assert "Verified reproduction bundle" in human.stdout


def test_cli_expected_errors_and_reduction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recipe = tmp_path / "bad-recipe.yaml"
    recipe.write_text("bad: recipe\n", encoding="utf-8")
    corpus = runner.invoke(
        app,
        ["corpus", "materialize", str(recipe), "--out", str(tmp_path / "corpus"), "--json"],
    )
    assert corpus.exit_code == 2
    assert json.loads(corpus.stdout)["error_code"] == "INVALID_INPUT"

    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")

    class FailingQualificationRunner:
        def run(self, qualification: Path, output: Path) -> QualificationOutcome:
            del qualification, output
            raise InvalidInputError("bad qualification")

    monkeypatch.setattr("upgrade_guard.cli.QualificationRunner", FailingQualificationRunner)
    qualified = runner.invoke(
        app,
        ["qualify", str(specification), "--out", str(tmp_path / "run"), "--json"],
    )
    assert qualified.exit_code == 2

    failure = tmp_path / "failure"
    failure.mkdir()
    (failure / "reduction-request.json").write_text(
        json.dumps(
            {
                "api_version": "upgradeguard.dev/v1alpha1",
                "kind": "ReductionRequest",
                "failure_code": "PROFILE_REJECTED",
                "signature_sha256": digest("5"),
                "confirmation_count": 2,
                "maximum_trials": 20,
                "maximum_seconds": 60,
                "predicate": {
                    "kind": "profile",
                    "input_name": "tokens",
                    "observed_shape": [9, 8, 256],
                    "minimum_shape": [1, 8, 256],
                    "maximum_shape": [8, 512, 256],
                },
            }
        ),
        encoding="utf-8",
    )
    reduced = runner.invoke(app, ["reduce", str(failure), "--out", str(tmp_path / "reduced")])
    assert reduced.exit_code == 0
    assert "Reduced PROFILE_REJECTED evidence" in reduced.stdout


def test_cli_human_resolver_and_typed_replay_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("upgrade_guard.cli.RegistryClient", FakeRegistryClient)
    resolved = runner.invoke(app, ["dev", "resolve-image", "registry.example/team/image:v1"])
    assert resolved.exit_code == 0
    assert "Selected manifest" in resolved.stdout
    assert "Canonical reference" in resolved.stdout

    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"fixture")
    replay = ReplayResult(
        schema_version="upgradeguard.dev/replay-result/v1",
        status="passed",
        bundle_id="bundle",
        bundle_manifest_sha256=digest("b"),
        worker_image="registry.example/worker@" + digest("c"),
        selected_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        expected_failure_code="PROFILE_REJECTED",
        step_results=("build-engine", "seeded-failure"),
    )
    monkeypatch.setattr("upgrade_guard.cli.execute_replay", lambda *args, **kwargs: replay)
    result = runner.invoke(
        app,
        ["reproduce", "run", str(bundle), "--out", str(tmp_path / "replay"), "--json"],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["expected_failure_code"] == "PROFILE_REJECTED"
    human = runner.invoke(
        app,
        ["reproduce", "run", str(bundle), "--out", str(tmp_path / "human-replay")],
    )
    assert human.exit_code == 0
    assert "Reproduced PROFILE_REJECTED: bundle" in human.stdout
    assert "replay-result.json" in human.stdout
