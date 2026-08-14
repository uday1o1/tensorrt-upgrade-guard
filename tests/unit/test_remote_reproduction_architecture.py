"""Locked CUDA architecture derivation for source-bearing remote bundles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.create_remote_reproductions import _locked_cuda_architecture


@pytest.mark.parametrize(
    ("compute_capability", "cmake_architecture"),
    (("8.9", "89"), ("12.0", "120")),
)
def test_remote_bundle_architecture_comes_from_candidate_environment_lock(
    compute_capability: str, cmake_architecture: str
) -> None:
    matrix = SimpleNamespace(
        environments=(
            SimpleNamespace(probe=SimpleNamespace(gpu=SimpleNamespace(compute_capability="7.5"))),
            SimpleNamespace(
                probe=SimpleNamespace(gpu=SimpleNamespace(compute_capability=compute_capability))
            ),
        )
    )

    assert _locked_cuda_architecture(matrix) == (
        compute_capability,
        cmake_architecture,
    )


@pytest.mark.parametrize("compute_capability", ["", "8.90", "sm_89"])
def test_remote_bundle_rejects_noncanonical_locked_capability(
    compute_capability: str,
) -> None:
    matrix = SimpleNamespace(
        environments=(
            SimpleNamespace(probe=SimpleNamespace(gpu=SimpleNamespace(compute_capability="7.5"))),
            SimpleNamespace(
                probe=SimpleNamespace(gpu=SimpleNamespace(compute_capability=compute_capability))
            ),
        )
    )

    with pytest.raises(ValueError, match="canonical major.minor"):
        _locked_cuda_architecture(matrix)  # type: ignore[arg-type]
