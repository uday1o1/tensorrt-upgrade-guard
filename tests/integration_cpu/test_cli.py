"""Public CLI behavior without importing NVIDIA Python software."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tests.factories import (
    FIXED_TIME,
    digest,
    reference_environment_lock,
    run_result,
    supported_doctor,
)
from upgrade_guard.cli import app
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.contracts.environment import PlatformIdentity, ResolvedImage
from upgrade_guard.errors import FailureCode, InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.qualification import QualificationOutcome
from upgrade_guard.report.model import build_report_model
from upgrade_guard.reproduce.run import ReplayResult, ReplayTarget

runner = CliRunner()


def _write_reference_lock(root: Path) -> Path:
    path = root / "reference-environment.lock.json"
    path.write_text(reference_environment_lock().model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


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
    reference_lock = _write_reference_lock(tmp_path)
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
        app,
        [
            "corpus",
            "materialize",
            str(recipe),
            "--out",
            str(tmp_path / "first"),
            "--reference-lock",
            str(reference_lock),
        ],
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
            "--reference-lock",
            str(reference_lock),
            "--json",
        ],
    )
    assert second.exit_code == 0
    assert json.loads(second.stdout)["id"] == "cli-smoke"


class StubQualificationRunner:
    status = "passed"

    def run(
        self,
        qualification: Path,
        output: Path,
        **kwargs: object,
    ) -> QualificationOutcome:
        del qualification, kwargs
        failure_codes = (FailureCode.NUMERICAL_REGRESSION,) if self.status == "failed" else ()
        return QualificationOutcome(output, self.status, failure_codes)  # type: ignore[arg-type]


def test_qualify_and_compare_cli_statuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")
    monkeypatch.setattr(
        "upgrade_guard.orchestrator.FullQualificationRunner", StubQualificationRunner
    )
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


def test_compare_rejects_incomplete_passing_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")

    class PublishedQualificationRunner:
        def run(
            self,
            qualification: Path,
            output: Path,
            **kwargs: object,
        ) -> QualificationOutcome:
            del qualification, kwargs
            summary = {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": "passed",
                "failure_codes": [],
            }
            core = output / "core-run"
            core.mkdir(parents=True)
            (core / "qualification-summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (output / "results.json").write_text(
                json.dumps(
                    {
                        "schema_version": "upgradeguard.dev/published-result-table/v1",
                        "status": "passed",
                        "failure_codes": [],
                        "core_qualification": summary,
                    }
                ),
                encoding="utf-8",
            )
            return QualificationOutcome(output, "passed", ())

    monkeypatch.setattr(
        "upgrade_guard.orchestrator.FullQualificationRunner",
        PublishedQualificationRunner,
    )
    output = tmp_path / "public-run"
    qualified = runner.invoke(
        app,
        ["qualify", str(specification), "--out", str(output), "--json"],
    )
    compared = runner.invoke(app, ["compare", str(output), "--json"])

    assert qualified.exit_code == 0
    assert compared.exit_code == 2
    assert json.loads(compared.stdout)["error_code"] == "INVALID_INPUT"


def test_compare_rejects_incomplete_failed_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")

    class FailedQualificationRunner:
        def run(
            self,
            qualification: Path,
            output: Path,
            **kwargs: object,
        ) -> QualificationOutcome:
            del qualification, kwargs
            summary = {
                "schema_version": "upgradeguard.dev/qualification-summary/v1",
                "status": "passed",
                "failure_codes": [],
            }
            published = {
                "schema_version": "upgradeguard.dev/published-result-table/v1",
                "status": "failed",
                "failure_codes": ["OUTPUT_SCHEMA_CHANGED"],
                "core_qualification": summary,
            }
            core = output / "core-run"
            core.mkdir(parents=True)
            (core / "qualification-summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (output / "results.json").write_text(json.dumps(published), encoding="utf-8")
            return QualificationOutcome(
                output,
                "failed",
                (FailureCode.OUTPUT_SCHEMA_CHANGED,),
            )

    monkeypatch.setattr(
        "upgrade_guard.orchestrator.FullQualificationRunner",
        FailedQualificationRunner,
    )
    output = tmp_path / "public-failed-run"
    qualified = runner.invoke(
        app,
        ["qualify", str(specification), "--out", str(output), "--json"],
    )
    compared = runner.invoke(app, ["compare", str(output), "--json"])

    assert qualified.exit_code == 1
    assert compared.exit_code == 2
    assert json.loads(compared.stdout)["error_code"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    ("status", "failure_codes", "exit_code"),
    [
        ("passed", [], 0),
        ("failed", ["NUMERICAL_REGRESSION"], 1),
        ("failed", ["CORPUS_INVALID"], 2),
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
    reference_lock = _write_reference_lock(tmp_path)
    recipe = tmp_path / "bad-recipe.yaml"
    recipe.write_text("bad: recipe\n", encoding="utf-8")
    corpus = runner.invoke(
        app,
        [
            "corpus",
            "materialize",
            str(recipe),
            "--out",
            str(tmp_path / "corpus"),
            "--reference-lock",
            str(reference_lock),
            "--json",
        ],
    )
    assert corpus.exit_code == 2
    assert json.loads(corpus.stdout)["error_code"] == "INVALID_INPUT"

    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")

    class FailingQualificationRunner:
        def run(
            self,
            qualification: Path,
            output: Path,
            **kwargs: object,
        ) -> QualificationOutcome:
            del qualification, output, kwargs
            raise InvalidInputError("bad qualification")

    monkeypatch.setattr(
        "upgrade_guard.orchestrator.FullQualificationRunner", FailingQualificationRunner
    )
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
        worker_rebuild_recipe_sha256=digest("d"),
        worker_build_log_sha256=digest("e"),
        worker_build_log=ArtifactReference(
            path="logs/worker-build.log",
            sha256=digest("e"),
            bytes=5,
            media_type="text/plain",
        ),
        original_gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
        selected_gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        expected_failure_code=FailureCode.PROFILE_REJECTED,
        observed_failure_code=FailureCode.PROFILE_REJECTED,
        step_results=("build-engine", "seeded-failure"),
    )
    monkeypatch.setattr("upgrade_guard.cli.execute_replay", lambda *args, **kwargs: replay)
    monkeypatch.setattr(
        "upgrade_guard.cli.observe_replay_target",
        lambda gpu_uuid: ReplayTarget(
            gpu_uuid=gpu_uuid or "GPU-11111111-1111-1111-1111-111111111111",
            compute_capability="8.9",
            driver_version="610.0",
            vram_mib=24576,
        ),
    )
    target_arguments = [
        "--gpu",
        "GPU-11111111-1111-1111-1111-111111111111",
        "--local-registry",
        "127.0.0.1:5500",
    ]
    result = runner.invoke(
        app,
        [
            "reproduce",
            "run",
            str(bundle),
            "--out",
            str(tmp_path / "replay"),
            *target_arguments,
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["expected_failure_code"] == "PROFILE_REJECTED"
    human = runner.invoke(
        app,
        [
            "reproduce",
            "run",
            str(bundle),
            "--out",
            str(tmp_path / "human-replay"),
            *target_arguments,
        ],
    )
    assert human.exit_code == 0
    assert "Reproduced PROFILE_REJECTED: bundle" in human.stdout
    assert "replay-result.json" in human.stdout


def test_public_command_groups_map_unexpected_failures_to_exit_five(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def crash(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("private implementation detail")

    specification = tmp_path / "qualification.yaml"
    specification.write_text("kind: Qualification\n", encoding="utf-8")

    class CrashingQualificationRunner:
        run = crash

    monkeypatch.setattr(
        "upgrade_guard.orchestrator.FullQualificationRunner",
        CrashingQualificationRunner,
    )
    qualified = runner.invoke(
        app,
        ["qualify", str(specification), "--out", str(tmp_path / "run"), "--json"],
    )
    assert qualified.exit_code == 5
    assert json.loads(qualified.stdout)["error_code"] == "INTERNAL_TOOL_FAILURE"
    assert "private implementation detail" not in qualified.stdout

    stored = tmp_path / "stored"
    stored.mkdir()
    monkeypatch.setattr("upgrade_guard.qualification.compare_stored_run", crash)
    compared = runner.invoke(app, ["compare", str(stored), "--json"])
    assert compared.exit_code == 5
    assert json.loads(compared.stdout)["error_code"] == "INTERNAL_TOOL_FAILURE"

    monkeypatch.setattr("upgrade_guard.reduce.session.reduce_failure_directory", crash)
    reduced = runner.invoke(
        app,
        ["reduce", str(stored), "--out", str(tmp_path / "reduced"), "--json"],
    )
    assert reduced.exit_code == 5
    assert json.loads(reduced.stdout)["error_code"] == "INTERNAL_TOOL_FAILURE"

    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"fixture")
    monkeypatch.setattr("upgrade_guard.cli.verify_bundle", crash)
    verified = runner.invoke(app, ["reproduce", "verify", str(bundle), "--json"])
    assert verified.exit_code == 5
    assert json.loads(verified.stdout)["error_code"] == "INTERNAL_TOOL_FAILURE"

    report_directory = tmp_path / "report"
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
    monkeypatch.setattr("upgrade_guard.cli.render_text", crash)
    rendered = runner.invoke(app, ["report", str(report_directory)])
    assert rendered.exit_code == 5
    assert "INTERNAL_TOOL_FAILURE" in rendered.stderr
    assert "private implementation detail" not in rendered.stderr
