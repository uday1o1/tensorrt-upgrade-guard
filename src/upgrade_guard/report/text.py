"""Human-readable text report."""

from __future__ import annotations

from upgrade_guard.report.model import ReportModel


def render_text(report: ReportModel) -> str:
    """Render a compact evidence-oriented report."""

    lines = [
        report.title,
        "=" * len(report.title),
        f"Status: {report.status.value}",
        f"Baseline: {report.baseline_environment_id}",
        f"Candidate: {report.candidate_environment_id}",
        f"Results: {report.result_count}",
        (
            "Counts: "
            f"{report.passed_count} passed, "
            f"{report.failed_count} failed, "
            f"{report.unsupported_count} unsupported, "
            f"{report.infrastructure_invalid_count} infrastructure-invalid, "
            f"{report.inconclusive_count} inconclusive"
        ),
        "",
        "Attribution",
        "-----------",
        report.stack_attribution,
    ]
    if report.failures:
        lines.extend(("", "Failures", "--------"))
        for failure in report.failures:
            lines.append(
                f"- {failure.code.value}: phase={failure.phase.value}, "
                f"gate={failure.gate}, observed={failure.observed}, "
                f"threshold={failure.threshold or 'not applicable'}"
            )
    if report.warnings:
        lines.extend(("", "Warnings", "--------"))
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"
