"""Normalized report input independent of a live worker."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import AwareDatetime

from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord, ResultStatus
from upgrade_guard.contracts.results import RunResult


class ReportModel(StrictModel):
    """Small static report model with evidence links."""

    api_version: Literal["upgradeguard.dev/report/v1"]
    title: str
    generated_at: AwareDatetime
    status: ResultStatus
    baseline_environment_id: str
    candidate_environment_id: str
    stack_attribution: str
    result_count: int
    passed_count: int
    failed_count: int
    unsupported_count: int
    infrastructure_invalid_count: int
    inconclusive_count: int
    failures: tuple[FailureRecord, ...]
    evidence: tuple[ArtifactReference, ...]
    warnings: tuple[str, ...]


def build_report_model(
    *,
    title: str,
    generated_at: AwareDatetime,
    baseline_environment_id: str,
    candidate_environment_id: str,
    results: tuple[RunResult, ...],
    evidence: tuple[ArtifactReference, ...] = (),
    warnings: tuple[str, ...] = (),
) -> ReportModel:
    """Summarize stored typed results without claiming component causality."""

    counts = Counter(result.status for result in results)
    failures = tuple(result.failure for result in results if result.failure is not None)
    status = _overall_status(results)
    return ReportModel(
        api_version="upgradeguard.dev/report/v1",
        title=title,
        generated_at=generated_at,
        status=status,
        baseline_environment_id=baseline_environment_id,
        candidate_environment_id=candidate_environment_id,
        stack_attribution=(
            "Observed changes belong to the compared locked stacks. "
            "They are not attributed to TensorRT alone without a smaller controlled experiment."
        ),
        result_count=len(results),
        passed_count=counts[ResultStatus.PASSED],
        failed_count=counts[ResultStatus.FAILED],
        unsupported_count=counts[ResultStatus.UNSUPPORTED],
        infrastructure_invalid_count=counts[ResultStatus.INFRASTRUCTURE_INVALID],
        inconclusive_count=counts[ResultStatus.INCONCLUSIVE],
        failures=failures,
        evidence=evidence,
        warnings=warnings,
    )


def _overall_status(results: tuple[RunResult, ...]) -> ResultStatus:
    statuses = {result.status for result in results}
    if ResultStatus.FAILED in statuses:
        return ResultStatus.FAILED
    if ResultStatus.INFRASTRUCTURE_INVALID in statuses:
        return ResultStatus.INFRASTRUCTURE_INVALID
    if ResultStatus.INCONCLUSIVE in statuses:
        return ResultStatus.INCONCLUSIVE
    if ResultStatus.UNSUPPORTED in statuses:
        return ResultStatus.UNSUPPORTED
    return ResultStatus.PASSED
