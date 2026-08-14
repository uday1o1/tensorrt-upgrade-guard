"""Stored-result report rendering tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tests.factories import FIXED_TIME, failure_record, run_result
from upgrade_guard.contracts.common import ArtifactReference, ResultStatus
from upgrade_guard.errors import FailureCode
from upgrade_guard.report.html_report import render_html
from upgrade_guard.report.json_report import render_json
from upgrade_guard.report.model import ReportModel, build_report_model
from upgrade_guard.report.text import render_text


def _report_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "api_version": "upgradeguard.dev/report/v1",
        "title": "Report contract fixture",
        "generated_at": FIXED_TIME,
        "status": ResultStatus.PASSED,
        "baseline_environment_id": "baseline",
        "candidate_environment_id": "candidate",
        "stack_attribution": "Complete locked stacks",
        "result_count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "unsupported_count": 0,
        "infrastructure_invalid_count": 0,
        "inconclusive_count": 0,
        "failures": (),
        "evidence": (),
        "warnings": (),
    }
    payload.update(updates)
    return payload


def _artifact() -> ArtifactReference:
    return ArtifactReference(
        path="results.json",
        sha256="sha256:" + "1" * 64,
        bytes=2,
        media_type="application/json",
    )


def _published_report_payload(**updates: object) -> dict[str, object]:
    artifact = _artifact()
    payload = _report_payload(
        result_count=1,
        passed_count=1,
        evidence=(artifact,),
        publication_complete=True,
        source_git_commit="1" * 40,
        gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        matrix_lock_sha256="sha256:" + "2" * 64,
        reference_environment_lock_sha256="sha256:" + "3" * 64,
        environment_images={"baseline": "registry.example/base@sha256:" + "4" * 64},
        acceptance_gates={"core": ResultStatus.PASSED},
        corpus_provenance=(artifact,),
        results_artifact=artifact,
        measured_sections={"numerical": "results.json#/numerical"},
        reproduction_commands=(("upgrade-guard", "qualify"),),
        methodology=("Frozen inputs",),
        limitations=("One selected GPU",),
    )
    payload.update(updates)
    return payload


def test_report_keeps_stack_attribution_and_failure_evidence() -> None:
    failure = failure_record(FailureCode.NUMERICAL_REGRESSION)
    report = build_report_model(
        title="UpgradeGuard qualification",
        generated_at=FIXED_TIME,
        baseline_environment_id="baseline",
        candidate_environment_id="candidate",
        results=(
            run_result(),
            run_result(status=ResultStatus.FAILED, failure=failure),
        ),
    )
    assert report.status is ResultStatus.FAILED
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert "not attributed to TensorRT alone" in report.stack_attribution
    assert report.failures == (failure,)

    text = render_text(report)
    assert "NUMERICAL_REGRESSION" in text
    assert "not attributed to TensorRT alone" in text
    assert json.loads(render_json(report))["status"] == "failed"


def test_html_report_escapes_untrusted_values() -> None:
    report = build_report_model(
        title="<script>alert(1)</script>",
        generated_at=FIXED_TIME,
        baseline_environment_id="<baseline>",
        candidate_environment_id="candidate",
        results=(run_result(),),
        warnings=("<img src=x>",),
    )
    rendered = render_html(report)
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img src=x>" not in rendered
    assert "<h2>Failures</h2><p>None.</p>" in rendered


def test_report_status_precedence() -> None:
    for status in (
        ResultStatus.UNSUPPORTED,
        ResultStatus.INCONCLUSIVE,
        ResultStatus.INFRASTRUCTURE_INVALID,
        ResultStatus.FAILED,
    ):
        report = build_report_model(
            title="report",
            generated_at=FIXED_TIME,
            baseline_environment_id="baseline",
            candidate_environment_id="candidate",
            results=(run_result(status=status),),
        )
        assert report.status is status


@pytest.mark.parametrize(
    "field",
    [
        "result_count",
        "passed_count",
        "failed_count",
        "unsupported_count",
        "infrastructure_invalid_count",
        "inconclusive_count",
    ],
)
def test_report_rejects_negative_counts(field: str) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ReportModel.model_validate(_report_payload(**{field: -1}))


def test_report_rejects_count_total_status_and_untyped_failure_contradictions() -> None:
    with pytest.raises(ValidationError, match="sum to result_count"):
        ReportModel.model_validate(_report_payload(result_count=1))
    with pytest.raises(ValidationError, match="status must agree"):
        ReportModel.model_validate(
            _report_payload(
                status=ResultStatus.FAILED,
                result_count=1,
                passed_count=1,
            )
        )
    with pytest.raises(ValidationError, match="typed failure records"):
        ReportModel.model_validate(
            _report_payload(
                status=ResultStatus.FAILED,
                result_count=1,
                failed_count=1,
                failures=(),
            )
        )


def test_report_typed_failures_cover_each_nonpassing_status() -> None:
    unsupported = failure_record(FailureCode.PREFLIGHT_UNSUPPORTED)
    inconclusive = failure_record(FailureCode.INCONCLUSIVE)
    with pytest.raises(ValidationError, match="typed failure records"):
        ReportModel.model_validate(_report_payload(failures=(unsupported,)))

    report = ReportModel.model_validate(
        _report_payload(
            status=ResultStatus.INCONCLUSIVE,
            result_count=2,
            unsupported_count=1,
            inconclusive_count=1,
            failures=(unsupported, inconclusive),
        )
    )
    assert report.failures == (unsupported, inconclusive)

    with pytest.raises(ValidationError, match="typed failure records"):
        ReportModel.model_validate(
            _report_payload(
                status=ResultStatus.FAILED,
                result_count=1,
                failed_count=1,
                failures=(unsupported,),
            )
        )


def test_one_failed_gate_can_retain_multiple_typed_failure_records() -> None:
    numerical = failure_record(FailureCode.NUMERICAL_REGRESSION)
    nondeterminism = failure_record(FailureCode.NONDETERMINISM_REGRESSION)
    report = ReportModel.model_validate(
        _report_payload(
            status=ResultStatus.FAILED,
            result_count=1,
            failed_count=1,
            failures=(numerical, nondeterminism),
            acceptance_gates={"core": ResultStatus.FAILED},
        )
    )

    assert report.failed_count == 1
    assert report.failures == (numerical, nondeterminism)


def test_acceptance_gates_must_be_typed_and_match_report_counts() -> None:
    with pytest.raises(ValidationError, match="acceptance_gates"):
        ReportModel.model_validate(_report_payload(acceptance_gates={"core": "not-a-status"}))
    with pytest.raises(ValidationError, match="gate statuses"):
        ReportModel.model_validate(
            _report_payload(
                result_count=1,
                passed_count=1,
                acceptance_gates={"core": ResultStatus.FAILED},
            )
        )


def test_incomplete_report_preserves_skeleton_without_publication_provenance() -> None:
    report = ReportModel.model_validate(
        _report_payload(result_count=1, passed_count=1, publication_complete=False)
    )
    assert report.publication_complete is False
    assert report.acceptance_gates == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("acceptance_gates", {}),
        ("evidence", ()),
        ("results_artifact", None),
        ("source_git_commit", None),
        ("matrix_lock_sha256", None),
        ("reference_environment_lock_sha256", None),
    ],
)
def test_complete_report_requires_decision_and_source_lock_provenance(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="full provenance"):
        ReportModel.model_validate(_published_report_payload(**{field: value}))


def test_complete_publication_renders_provenance_and_reproduction() -> None:
    report = ReportModel.model_validate(
        _published_report_payload(
            title="Published qualification",
            warnings=("Scoped result",),
        )
    )

    text = render_text(report)
    html = render_html(report)
    assert "Source commit" in text
    assert "upgrade-guard qualify" in text
    assert "Acceptance gates" in html
    assert "One selected GPU" in html
