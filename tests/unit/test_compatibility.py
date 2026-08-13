"""Compatibility policy positive and negative fixtures."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.factories import worker_probe
from upgrade_guard.contracts.environment import ToolObservation
from upgrade_guard.contracts.matrix import CapabilityPolicy
from upgrade_guard.matrix.compatibility import evaluate_compatibility

GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"


def test_current_full_v1_probe_is_compatible() -> None:
    result = evaluate_compatibility(
        worker_probe(),
        CapabilityPolicy(),
        expected_gpu_uuid=GPU_UUID,
    )
    assert result.compatible
    assert result.reasons == ()
    assert result.minimum_driver == "580.0"


def test_old_driver_is_rejected() -> None:
    result = evaluate_compatibility(
        worker_probe(driver="579.99"),
        CapabilityPolicy(),
        expected_gpu_uuid=GPU_UUID,
    )
    assert not result.compatible
    assert any("older than required" in reason for reason in result.reasons)


def test_unknown_cuda_major_is_rejected() -> None:
    result = evaluate_compatibility(
        worker_probe(cuda_runtime="14.0"),
        CapabilityPolicy(),
        expected_gpu_uuid=GPU_UUID,
    )
    assert not result.compatible
    assert result.minimum_driver == "unsupported"


def test_missing_extended_v1_tool_is_rejected() -> None:
    result = evaluate_compatibility(
        worker_probe(compute_sanitizer=False),
        CapabilityPolicy(),
        expected_gpu_uuid=GPU_UUID,
    )
    assert not result.compatible
    assert "required tool is unavailable: Compute Sanitizer" in result.reasons


def test_authored_matrix_cannot_disable_extended_v1_tools() -> None:
    with pytest.raises(ValidationError, match="True"):
        CapabilityPolicy(compute_sanitizer=False)


def test_wrong_gpu_and_old_compute_capability_are_rejected() -> None:
    result = evaluate_compatibility(
        worker_probe(
            gpu_uuid="GPU-22222222-2222-2222-2222-222222222222",
            compute_capability="7.0",
        ),
        CapabilityPolicy(),
        expected_gpu_uuid=GPU_UUID,
    )
    assert not result.compatible
    assert any("expected selected GPU" in reason for reason in result.reasons)
    assert any("below 7.5" in reason for reason in result.reasons)


def test_missing_header_and_version_evidence_is_rejected() -> None:
    probe = worker_probe().model_copy(
        update={
            "tensorrt": "",
            "cuda_headers": (),
            "nsight_compute": ToolObservation(available=False),
        }
    )
    result = evaluate_compatibility(
        probe,
        CapabilityPolicy(),
        expected_gpu_uuid=GPU_UUID,
    )
    assert not result.compatible
    assert "TensorRT version was not observed" in result.reasons
    assert "required CUDA headers are unavailable" in result.reasons
    assert "required tool is unavailable: Nsight Compute" in result.reasons
