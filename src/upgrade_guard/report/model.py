"""Normalized report input independent of a live worker."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from upgrade_guard.classify import status_for_failure
from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.common import ArtifactReference, FailureRecord, ResultStatus
from upgrade_guard.contracts.environment import Sha256Digest
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
    result_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    infrastructure_invalid_count: int = Field(ge=0)
    inconclusive_count: int = Field(ge=0)
    failures: tuple[FailureRecord, ...]
    evidence: tuple[ArtifactReference, ...]
    warnings: tuple[str, ...]
    publication_complete: bool = False
    source_git_commit: str | None = None
    gpu_uuid: str | None = None
    matrix_lock_sha256: Sha256Digest | None = None
    reference_environment_lock_sha256: Sha256Digest | None = None
    environment_images: dict[str, str] = Field(default_factory=dict)
    acceptance_gates: dict[str, ResultStatus] = Field(default_factory=dict)
    corpus_provenance: tuple[ArtifactReference, ...] = ()
    results_artifact: ArtifactReference | None = None
    measured_sections: dict[str, str] = Field(default_factory=dict)
    reproduction_commands: tuple[tuple[str, ...], ...] = ()
    methodology: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_consistent_result_inventory(self) -> ReportModel:
        counts = {
            ResultStatus.PASSED: self.passed_count,
            ResultStatus.FAILED: self.failed_count,
            ResultStatus.UNSUPPORTED: self.unsupported_count,
            ResultStatus.INFRASTRUCTURE_INVALID: self.infrastructure_invalid_count,
            ResultStatus.INCONCLUSIVE: self.inconclusive_count,
        }
        if sum(counts.values()) != self.result_count:
            raise ValueError("report result counts must sum to result_count")
        expected_failure_statuses = {
            status
            for status, count in counts.items()
            if status is not ResultStatus.PASSED and count > 0
        }
        observed_failure_statuses = {status_for_failure(failure.code) for failure in self.failures}
        if observed_failure_statuses != expected_failure_statuses:
            raise ValueError("typed failure records must cover every non-passing result status")
        expected_status = _overall_status_from_counts(counts)
        if self.status is not expected_status:
            raise ValueError("report status must agree with result counts")
        if self.acceptance_gates:
            if any(not name for name in self.acceptance_gates):
                raise ValueError("acceptance gate names cannot be empty")
            gate_counts = Counter(self.acceptance_gates.values())
            if any(gate_counts[status] != count for status, count in counts.items()):
                raise ValueError("acceptance gate statuses must match report result counts")
        if self.publication_complete and (
            not self.source_git_commit
            or self.gpu_uuid is None
            or self.matrix_lock_sha256 is None
            or self.reference_environment_lock_sha256 is None
            or not self.environment_images
            or not self.acceptance_gates
            or not self.evidence
            or not self.corpus_provenance
            or self.results_artifact is None
            or not self.measured_sections
            or not self.reproduction_commands
            or not self.methodology
            or not self.limitations
        ):
            raise ValueError("complete public reports require full provenance and methodology")
        return self


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
    return _overall_status_from_counts(Counter(result.status for result in results))


def _overall_status_from_counts(
    counts: dict[ResultStatus, int] | Counter[ResultStatus],
) -> ResultStatus:
    if counts[ResultStatus.FAILED]:
        return ResultStatus.FAILED
    if counts[ResultStatus.INFRASTRUCTURE_INVALID]:
        return ResultStatus.INFRASTRUCTURE_INVALID
    if counts[ResultStatus.INCONCLUSIVE]:
        return ResultStatus.INCONCLUSIVE
    if counts[ResultStatus.UNSUPPORTED]:
        return ResultStatus.UNSUPPORTED
    return ResultStatus.PASSED
