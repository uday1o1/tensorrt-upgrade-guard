"""Machine-readable static report."""

from __future__ import annotations

from upgrade_guard.report.model import ReportModel


def render_json(report: ReportModel) -> str:
    """Render stable schema-valid JSON."""

    return report.model_dump_json(indent=2) + "\n"
