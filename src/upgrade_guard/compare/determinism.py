"""Repeated-output determinism evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.common import NumericalTolerance
from upgrade_guard.contracts.results import DeterminismSummary
from upgrade_guard.errors import InvalidInputError

Array = npt.NDArray[Any]


def summarize_determinism(
    outputs: tuple[Array, ...],
    input_hashes: tuple[str, ...],
    policy: NumericalTolerance,
) -> DeterminismSummary:
    """Keep bitwise and tolerance stability as separate observations."""

    if not outputs:
        raise InvalidInputError("determinism requires at least one output")
    first = np.asarray(outputs[0])
    if any(np.asarray(output).shape != first.shape for output in outputs[1:]):
        raise InvalidInputError("determinism repetitions changed output shape")
    hashes = tuple(sha256_bytes(np.asarray(output).tobytes(order="C")) for output in outputs)
    finite = all(bool(np.all(np.isfinite(output))) for output in outputs)
    tolerance_stable = finite and all(
        bool(np.allclose(first, output, rtol=policy.rtol, atol=policy.atol, equal_nan=False))
        for output in outputs[1:]
    )
    return DeterminismSummary(
        repetitions=len(outputs),
        unique_output_hashes=tuple(sorted(set(hashes))),
        bitwise_stable=len(set(hashes)) == 1,
        tolerance_stable=tolerance_stable,
        input_hashes_stable=len(set(input_hashes)) <= 1,
        nonfinite_observed=not finite,
    )
