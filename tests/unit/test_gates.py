"""Qualification gate authority contracts."""

from __future__ import annotations

import pytest

from upgrade_guard.gates import (
    direct_step_dependencies,
    expected_publication_steps,
    step_is_bound_to,
)


def test_publication_step_contract_rejects_inconsistent_requests() -> None:
    with pytest.raises(ValueError, match="passing publication"):
        expected_publication_steps("passed", failure_step="core-qualification")
    with pytest.raises(ValueError, match="failed publication"):
        expected_publication_steps("failed")
    with pytest.raises(ValueError, match="requires one failed"):
        direct_step_dependencies("public-failure")
    with pytest.raises(ValueError, match="unknown qualification step"):
        direct_step_dependencies("unknown")


def test_transitive_binding_handles_shared_dependencies_once() -> None:
    assert step_is_bound_to("profiles", "matrix-lock")
    assert not step_is_bound_to("preflight", "matrix-lock")
    assert step_is_bound_to(
        "public-failure",
        "corpus-materialization",
        failure_step="mobilenet-matrix",
    )
