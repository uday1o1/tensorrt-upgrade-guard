"""Public offline evidence reducer for numerical and profile failures."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.errors import InvalidInputError
from upgrade_guard.reduce.inputs import reduce_numerical_failure
from upgrade_guard.reduce.predicate import NumericalPredicate, ReductionRequest
from upgrade_guard.reduce.shapes import reduce_profile_failure


def reduce_failure_directory(source: Path, destination: Path) -> dict[str, Any]:
    """Reduce hash-addressed stored evidence without executing bundled code."""

    if destination.exists() or destination.is_symlink():
        raise InvalidInputError("refusing to overwrite reduction output")
    request_path = source / "reduction-request.json"
    try:
        request = ReductionRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise InvalidInputError("failure directory lacks a valid reduction request") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        if isinstance(request.predicate, NumericalPredicate):
            summary = _reduce_numerical(source, staging, request)
        else:
            reduced = reduce_profile_failure(request.predicate)
            summary = {
                "kind": "profile",
                "failure_code": request.failure_code.value,
                "signature_sha256": request.signature_sha256,
                "confirmation_count": request.confirmation_count,
                "reduced": {
                    "observed_shape": reduced.observed_shape,
                    "minimum_shape": reduced.minimum_shape,
                    "optimum_shape": reduced.optimum_shape,
                    "maximum_shape": reduced.maximum_shape,
                    "violating_dimension": reduced.violating_dimension,
                    "direction": reduced.direction,
                },
            }
        (staging / "reduction-result.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reduce_numerical(source: Path, staging: Path, request: ReductionRequest) -> dict[str, Any]:
    predicate = request.predicate
    if not isinstance(predicate, NumericalPredicate):
        raise AssertionError("numerical reducer received a profile predicate")
    reference_path = _safe_source_path(source, predicate.reference_path)
    candidate_path = _safe_source_path(source, predicate.candidate_path)
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    reduced = reduce_numerical_failure(
        reference,
        candidate,
        atol=predicate.atol,
        rtol=predicate.rtol,
    )
    reduced_reference = staging / "reference.npy"
    reduced_candidate = staging / "candidate.npy"
    np.save(reduced_reference, reduced.reference, allow_pickle=False)
    np.save(reduced_candidate, reduced.candidate, allow_pickle=False)
    return {
        "kind": "numerical",
        "failure_code": request.failure_code.value,
        "signature_sha256": request.signature_sha256,
        "confirmation_count": request.confirmation_count,
        "original_shape": reduced.original_shape,
        "flat_index": reduced.flat_index,
        "multidimensional_index": reduced.multidimensional_index,
        "absolute_error": reduced.absolute_error,
        "threshold": reduced.threshold,
        "reference_sha256": sha256_file(reduced_reference),
        "candidate_sha256": sha256_file(reduced_candidate),
    }


def _safe_source_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".npy":
        raise InvalidInputError("reduction array path must be a relative NPY file")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise InvalidInputError("reduction array escaped the failure directory")
    return resolved
