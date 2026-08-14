"""Typed three-way numerical predicate used by source-bearing replay bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from upgrade_guard.compare.numerical import ThreeWayPrecedenceError, decide_three_way
from upgrade_guard.contracts.base import StrictModel, model_sha256, sha256_file
from upgrade_guard.contracts.common import NumericalPolicy
from upgrade_guard.contracts.environment import Sha256Digest
from upgrade_guard.errors import FailureCode


class PublicNumericalReplayPredicate(StrictModel):
    """Hash-bound predicate identity carried by a portable numerical replay."""

    schema_version: str = "upgradeguard.dev/public-numerical-replay-predicate/v1"
    failure_code: FailureCode
    failure_signature_sha256: Sha256Digest
    output_name: str = Field(min_length=1, max_length=256)
    semantics: Literal["tensor", "classification"]
    indexes: tuple[int, ...] = Field(min_length=1)
    reference_sha256: Sha256Digest
    baseline_sha256: Sha256Digest
    policy_sha256: Sha256Digest
    predicate_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_predicate(self) -> PublicNumericalReplayPredicate:
        if self.failure_code is not FailureCode.NUMERICAL_REGRESSION:
            raise ValueError("public numerical replay requires NUMERICAL_REGRESSION")
        if self.semantics not in {"tensor", "classification"}:
            raise ValueError("public numerical replay semantics are invalid")
        if tuple(sorted(set(self.indexes))) != self.indexes or any(
            index < 0 for index in self.indexes
        ):
            raise ValueError("public numerical replay indexes are invalid")
        if self.predicate_sha256 != self.computed_sha256():
            raise ValueError("public numerical replay predicate self-hash differs")
        return self

    def computed_sha256(self) -> str:
        return model_sha256(self, exclude={"predicate_sha256"})


def build_replay_predicate(
    *,
    failure_signature_sha256: str,
    output_name: str,
    semantics: Literal["tensor", "classification"],
    indexes: tuple[int, ...],
    reference: Path,
    baseline: Path,
    policy: Path,
) -> PublicNumericalReplayPredicate:
    """Build and validate one replay predicate from exact retained artifacts."""

    unvalidated = PublicNumericalReplayPredicate.model_construct(
        failure_code=FailureCode.NUMERICAL_REGRESSION,
        failure_signature_sha256=failure_signature_sha256,
        output_name=output_name,
        semantics=semantics,
        indexes=indexes,
        reference_sha256=sha256_file(reference),
        baseline_sha256=sha256_file(baseline),
        policy_sha256=sha256_file(policy),
        predicate_sha256="sha256:" + "0" * 64,
    )
    return PublicNumericalReplayPredicate.model_validate(
        unvalidated.model_copy(update={"predicate_sha256": unvalidated.computed_sha256()})
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--predicate", type=Path, required=True)
    arguments = parser.parse_args()
    policy = NumericalPolicy.model_validate_json(arguments.policy.read_text(encoding="utf-8"))
    predicate = PublicNumericalReplayPredicate.model_validate_json(
        arguments.predicate.read_text(encoding="utf-8")
    )
    if (
        sha256_file(arguments.reference) != predicate.reference_sha256
        or sha256_file(arguments.baseline) != predicate.baseline_sha256
        or sha256_file(arguments.policy) != predicate.policy_sha256
    ):
        raise RuntimeError("replay numerical predicate artifacts changed")
    values = tuple(
        np.load(path, allow_pickle=False)
        for path in (arguments.reference, arguments.baseline, arguments.candidate)
    )
    indexes = predicate.indexes
    if not indexes or len(indexes) != len(set(indexes)) or any(index < 0 for index in indexes):
        raise RuntimeError("replay numerical indexes are outside the retained output")
    index_array = np.asarray(indexes, dtype=np.int64)
    if predicate.semantics == "classification":
        if any(
            value.ndim != 2
            or value.shape[0] != values[0].shape[0]
            or value.shape[1] <= int(index_array[-1])
            for value in values
        ):
            raise RuntimeError("replay classification indexes are outside the class dimension")
        reference, baseline, candidate = (value[:, index_array] for value in values)
    else:
        if any(value.size <= int(index_array[-1]) for value in values):
            raise RuntimeError("replay numerical indexes are outside the retained output")
        reference, baseline, candidate = (value.reshape(-1)[index_array] for value in values)
    try:
        decision = decide_three_way(
            predicate.output_name,
            reference,
            baseline,
            candidate,
            policy=policy,
            semantics="classification" if predicate.semantics == "classification" else None,
        )
    except ThreeWayPrecedenceError as error:
        raise RuntimeError("replay three-way precedence changed") from error
    if decision.failure_code is not FailureCode.NUMERICAL_REGRESSION:
        raise RuntimeError("replay did not preserve the numerical regression")
    print(
        json.dumps(
            {
                "status": "failed",
                "failure_code": FailureCode.NUMERICAL_REGRESSION.value,
                "signature_sha256": predicate.failure_signature_sha256,
                "predicate_sha256": predicate.predicate_sha256,
                "indexes": indexes,
                "semantics": predicate.semantics,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
