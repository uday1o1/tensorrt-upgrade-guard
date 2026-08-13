"""Stored-result report rendering tests."""

from __future__ import annotations

import json

from tests.factories import FIXED_TIME, failure_record, run_result
from upgrade_guard.contracts.common import ResultStatus
from upgrade_guard.errors import FailureCode
from upgrade_guard.report.html_report import render_html
from upgrade_guard.report.json_report import render_json
from upgrade_guard.report.model import build_report_model
from upgrade_guard.report.text import render_text


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
