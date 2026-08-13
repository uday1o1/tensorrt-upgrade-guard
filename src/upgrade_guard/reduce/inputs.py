"""Numerical input and output evidence reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from upgrade_guard.errors import InvalidInputError

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class ReducedNumericalFailure:
    """A smallest elementwise slice retaining the threshold violation."""

    original_shape: tuple[int, ...]
    flat_index: int
    multidimensional_index: tuple[int, ...]
    reference: Array
    candidate: Array
    absolute_error: float
    threshold: float


def reduce_numerical_failure(
    reference: Array,
    candidate: Array,
    *,
    atol: float,
    rtol: float,
) -> ReducedNumericalFailure:
    """Select the strongest finite elementwise failure and retain a scalar array."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if (
        reference_array.shape != candidate_array.shape
        or reference_array.dtype != candidate_array.dtype
    ):
        raise InvalidInputError("numerical reduction requires matching array schema")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(candidate_array)):
        raise InvalidInputError("nonfinite evidence belongs to NONFINITE_OUTPUT reduction")
    absolute = np.abs(candidate_array.astype(np.float64) - reference_array.astype(np.float64))
    thresholds = atol + rtol * np.abs(reference_array.astype(np.float64))
    margin = absolute - thresholds
    flat_index = int(np.argmax(margin))
    if float(margin.reshape(-1)[flat_index]) <= 0:
        raise InvalidInputError("stored arrays do not satisfy the numerical failure predicate")
    index = tuple(int(item) for item in np.unravel_index(flat_index, reference_array.shape))
    return ReducedNumericalFailure(
        original_shape=tuple(int(item) for item in reference_array.shape),
        flat_index=flat_index,
        multidimensional_index=index,
        reference=np.asarray([reference_array[index]], dtype=reference_array.dtype),
        candidate=np.asarray([candidate_array[index]], dtype=candidate_array.dtype),
        absolute_error=float(absolute[index]),
        threshold=float(thresholds[index]),
    )
