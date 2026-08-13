"""Dependency-free static HTML report."""

from __future__ import annotations

import html

from upgrade_guard.report.model import ReportModel


def render_html(report: ReportModel) -> str:
    """Render a self-contained accessible HTML report."""

    failures = "".join(
        (
            "<tr>"
            f"<td>{html.escape(failure.code.value)}</td>"
            f"<td>{html.escape(failure.phase.value)}</td>"
            f"<td>{html.escape(failure.gate)}</td>"
            f"<td>{html.escape(failure.observed)}</td>"
            f"<td>{html.escape(failure.threshold or 'not applicable')}</td>"
            "</tr>"
        )
        for failure in report.failures
    )
    failure_section = (
        "<h2>Failures</h2>"
        "<table><thead><tr><th>Code</th><th>Phase</th><th>Gate</th>"
        "<th>Observed</th><th>Threshold</th></tr></thead>"
        f"<tbody>{failures}</tbody></table>"
        if failures
        else "<h2>Failures</h2><p>None.</p>"
    )
    warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in report.warnings)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report.title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 72rem; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #bbb; padding: .5rem; text-align: left; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{html.escape(report.title)}</h1>
<dl>
<dt>Status</dt><dd>{html.escape(report.status.value)}</dd>
<dt>Baseline</dt><dd><code>{html.escape(report.baseline_environment_id)}</code></dd>
<dt>Candidate</dt><dd><code>{html.escape(report.candidate_environment_id)}</code></dd>
<dt>Results</dt><dd>{report.result_count}</dd>
</dl>
<h2>Attribution</h2>
<p>{html.escape(report.stack_attribution)}</p>
{failure_section}
<h2>Warnings</h2>
<ul>{warnings}</ul>
</body>
</html>
"""
