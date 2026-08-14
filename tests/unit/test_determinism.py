"""Repeated execution evidence tests."""

from __future__ import annotations

import numpy as np
import pytest

from upgrade_guard.compare.determinism import summarize_determinism
from upgrade_guard.contracts.common import NumericalTolerance
from upgrade_guard.errors import InvalidInputError


def test_bitwise_and_tolerance_stability_remain_separate() -> None:
    first = np.ones(8, dtype=np.float32)
    second = first.copy()
    second[0] = np.nextafter(second[0], np.float32(2.0))
    result = summarize_determinism(
        (first, second),
        ("sha256:" + "1" * 64,) * 2,
        NumericalTolerance(atol=1e-6, rtol=1e-6),
    )
    assert not result.bitwise_stable
    assert result.tolerance_stable
    assert result.input_hashes_stable


def test_nonfinite_is_not_mislabeled_as_floating_variation() -> None:
    output = np.asarray([np.inf], dtype=np.float32)
    result = summarize_determinism(
        (output, output.copy()),
        ("sha256:" + "1" * 64,) * 2,
        NumericalTolerance(atol=1e-6, rtol=1e-6),
    )
    assert result.nonfinite_observed
    assert not result.tolerance_stable


def test_determinism_rejects_empty_and_shape_changing_repetitions() -> None:
    policy = NumericalTolerance(atol=0, rtol=0)
    with pytest.raises(InvalidInputError, match="at least one"):
        summarize_determinism((), (), policy)
    with pytest.raises(InvalidInputError, match="changed output shape"):
        summarize_determinism(
            (np.ones((1,), dtype=np.float32), np.ones((2,), dtype=np.float32)),
            ("one",),
            policy,
        )


def test_per_repetition_input_integrity_overrides_legacy_flat_hashes() -> None:
    output = np.ones((2,), dtype=np.float32)
    result = summarize_determinism(
        (output, output.copy()),
        ("sha256:" + "1" * 64, "sha256:" + "2" * 64),
        NumericalTolerance(atol=0.0, rtol=0.0),
        input_hashes_stable=True,
    )
    assert result.input_hashes_stable is True
