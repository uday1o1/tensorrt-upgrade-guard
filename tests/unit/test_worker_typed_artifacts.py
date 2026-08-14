"""Strict worker artifact and stable-manifest adapter tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from tests.factories import failure_record
from upgrade_guard.contracts.base import model_sha256
from upgrade_guard.contracts.build import (
    BuildManifestAdapterContext,
    WorkerBuildResult,
    adapt_worker_build,
)
from upgrade_guard.contracts.case import (
    CaseManifest,
    ReferenceCapability,
    SourceAttribution,
    adapt_case_manifest,
)
from upgrade_guard.contracts.common import (
    ArtifactReference,
    DeterminismPolicy,
    FailureRecord,
    NumericalPolicy,
    NumericalTolerance,
    ResultStatus,
    TensorContract,
)
from upgrade_guard.contracts.results import (
    HardwareObservation,
    RunResult,
    RunResultAdapterContext,
    TimingBlock,
    WorkerCorrectnessResult,
    adapt_worker_run,
)
from upgrade_guard.errors import FailureCode


def digest(character: str) -> str:
    return "sha256:" + character * 64


def _worker_run_payload() -> dict[str, object]:
    return {
        "schema_version": "upgradeguard.dev/worker-correctness/v1",
        "status": "passed",
        "command": ["python3", "-m", "upgrade_guard.worker.run_correctness"],
        "command_sha256": digest("4"),
        "engine_sha256": digest("5"),
        "input_sha256": {"x": digest("1")},
        "repetitions": [
            {
                "index": 0,
                "inputs": [
                    {
                        "name": "x",
                        "source_sha256": digest("1"),
                        "host_value_sha256": digest("2"),
                        "device_value_sha256": digest("2"),
                        "stable": True,
                    }
                ],
                "outputs": [
                    {
                        "name": "y",
                        "path": "/output/y.repetition-00.npy",
                        "sha256": digest("3"),
                        "bytes": 16,
                        "dtype": "float32",
                        "shape": [1, 2],
                    }
                ],
            }
        ],
        "input_integrity_stable": True,
        "memory_diagnostics": {"execution_context_device_memory_bytes": 64},
        "tensorrt_version": "11.2.1",
        "started_unix_seconds": 1.0,
        "ended_unix_seconds": 2.0,
        "duration_seconds": 1.0,
    }


def _worker_build_payload() -> dict[str, object]:
    return {
        "schema_version": "upgradeguard.dev/worker-build/v1",
        "status": "passed",
        "command": ["python3", "-m", "upgrade_guard.worker.build_engine"],
        "command_sha256": digest("1"),
        "model": {"path": "/corpus/model.onnx", "sha256": digest("2"), "bytes": 10},
        "engine": {
            "path": "/output/engine.plan",
            "sha256": digest("3"),
            "bytes": 20,
            "device_memory_bytes": 30,
        },
        "memory_diagnostics": {},
        "inspector": {
            "path": "/output/inspector.json",
            "sha256": digest("4"),
            "bytes": 40,
        },
        "timing_cache": {
            "path": "/output/timing.cache",
            "input_sha256": None,
            "output_sha256": digest("5"),
            "bytes": 50,
        },
        "builder_configuration": {"strongly_typed": "true"},
        "timing_cache_state": "cold",
        "tensorrt_version": "11.2.1",
        "started_unix_seconds": 1.0,
        "ended_unix_seconds": 2.0,
        "duration_seconds": 1.0,
        "strongly_typed": True,
    }


def _run_context(*, failure: FailureRecord | None = None) -> RunResultAdapterContext:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    environment_hash = digest("7")
    return RunResultAdapterContext(
        id="run",
        case_manifest_sha256=digest("8"),
        build_manifest_sha256=digest("9"),
        environment_lock_sha256=environment_hash,
        hardware_sha256=digest("a"),
        hardware=HardwareObservation(
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            driver="580.80.01",
            environment_lock_sha256=environment_hash,
            valid=True,
            invalid_reasons=(),
        ),
        started_at=now,
        ended_at=now,
        serialized_engine_bytes=20,
        engine_device_memory_bytes=30,
        determinism_tolerance_stable=True,
        failure=failure,
    )


def test_case_adapter_requires_a_valid_self_hash() -> None:
    artifact = ArtifactReference(
        path="model.onnx", sha256=digest("1"), bytes=1, media_type="application/onnx"
    )
    tolerance = NumericalTolerance(atol=1e-5, rtol=1e-4)
    case = CaseManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="CaseManifest",
        id="case",
        model_id="model",
        source=SourceAttribution(
            name="model",
            source_url="https://example.invalid/model",
            source_revision="1",
            license_name="Apache-2.0",
            license_url="https://example.invalid/license",
            redistribution_allowed=True,
        ),
        model=artifact,
        opset=17,
        ir_version=10,
        exporter_environment_sha256=digest("2"),
        precision="fp32",
        profile_id="profile",
        shape_id="shape",
        inputs=(TensorContract(name="x", dtype="float32", shape=(1, 2)),),
        input_fixtures=(
            ArtifactReference(
                path="x.npy", sha256=digest("3"), bytes=1, media_type="application/x-npy"
            ),
        ),
        outputs=(TensorContract(name="y", dtype="float32", shape=(1, 2)),),
        reference_runner="onnxruntime_cpu",
        reference_environment_sha256=digest("4"),
        reference_capability=ReferenceCapability(
            supported=True,
            execution_provider="CPUExecutionProvider",
            observed_input_dtypes={"x": "float32"},
            observed_output_dtypes={"y": "float32"},
        ),
        numerical=NumericalPolicy(
            baseline_to_reference=tolerance,
            candidate_to_reference=tolerance,
            candidate_to_baseline=tolerance,
        ),
        determinism=DeterminismPolicy(repetitions=20, require_bitwise=False, tolerance=tolerance),
        workload_weight=1.0,
        manifest_sha256=digest("0"),
    )
    case = case.model_copy(update={"manifest_sha256": case.computed_sha256()})
    assert adapt_case_manifest(case) == case
    with pytest.raises(ValueError, match="self-hash"):
        adapt_case_manifest(case.model_copy(update={"manifest_sha256": digest("f")}))


def test_worker_build_is_strict_and_promotes_to_build_manifest() -> None:
    worker = WorkerBuildResult.model_validate(_worker_build_payload())
    manifest = adapt_worker_build(
        worker,
        BuildManifestAdapterContext(
            id="build",
            case_manifest_sha256=digest("6"),
            environment_lock_sha256=digest("7"),
        ),
    )
    assert manifest.engine is not None and manifest.engine.bytes == 20
    assert manifest.engine_device_memory_bytes == 30
    with pytest.raises(ValidationError, match="extra"):
        WorkerBuildResult.model_validate({**worker.model_dump(), "extra": True})


@pytest.mark.parametrize(
    "timestamps",
    [
        {"started_unix_seconds": float("nan")},
        {"ended_unix_seconds": 0.5},
        {"duration_seconds": 0.5},
    ],
)
def test_worker_build_rejects_invalid_chronology(timestamps: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="timestamps and duration"):
        WorkerBuildResult.model_validate({**_worker_build_payload(), **timestamps})


def test_stable_build_manifest_rejects_reversed_or_inconsistent_time() -> None:
    worker = WorkerBuildResult.model_validate(_worker_build_payload())
    manifest = adapt_worker_build(
        worker,
        BuildManifestAdapterContext(
            id="build",
            case_manifest_sha256=digest("6"),
            environment_lock_sha256=digest("7"),
        ),
    )
    with pytest.raises(ValidationError, match="timestamps and duration"):
        type(manifest).model_validate(
            {
                **manifest.model_dump(),
                "ended_at": manifest.started_at - timedelta(seconds=1),
            }
        )
    with pytest.raises(ValidationError, match="timestamps and duration"):
        type(manifest).model_validate({**manifest.model_dump(), "duration_seconds": 0.5})


def test_worker_run_promotes_input_and_selected_tactic_evidence() -> None:
    input_hash = digest("1")
    value_hash = digest("2")
    output_hash = digest("3")
    worker = WorkerCorrectnessResult.model_validate(
        {
            "schema_version": "upgradeguard.dev/worker-correctness/v1",
            "status": "passed",
            "command": ["python3", "-m", "upgrade_guard.worker.run_correctness"],
            "command_sha256": digest("4"),
            "engine_sha256": digest("5"),
            "input_sha256": {"x": input_hash},
            "repetitions": [
                {
                    "index": 0,
                    "inputs": [
                        {
                            "name": "x",
                            "source_sha256": input_hash,
                            "host_value_sha256": value_hash,
                            "device_value_sha256": value_hash,
                            "stable": True,
                        }
                    ],
                    "outputs": [
                        {
                            "name": "y",
                            "path": "/output/y.repetition-00.npy",
                            "sha256": output_hash,
                            "bytes": 16,
                            "dtype": "float32",
                            "shape": [1, 2],
                        }
                    ],
                }
            ],
            "input_integrity_stable": True,
            "tactic_diagnostic": {
                "path": "/output/tactic-diagnostics.jsonl",
                "sha256": digest("6"),
                "bytes": 32,
                "engine_sha256": digest("5"),
                "selected_tactic": "kVECTORIZED_WARP",
                "rows": 1,
                "hidden": 2,
                "enqueue_count": 1,
            },
            "memory_diagnostics": {"execution_context_device_memory_bytes": 64},
            "tensorrt_version": "11.2.1",
            "started_unix_seconds": 1.0,
            "ended_unix_seconds": 2.0,
            "duration_seconds": 1.0,
        }
    )
    now = datetime(2026, 8, 13, tzinfo=UTC)
    environment_hash = digest("7")
    result = adapt_worker_run(
        worker,
        RunResultAdapterContext(
            id="run",
            case_manifest_sha256=digest("8"),
            build_manifest_sha256=digest("9"),
            environment_lock_sha256=environment_hash,
            hardware_sha256=digest("a"),
            hardware=HardwareObservation(
                gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
                driver="580.80.01",
                environment_lock_sha256=environment_hash,
                valid=True,
                invalid_reasons=(),
            ),
            started_at=now,
            ended_at=now,
            serialized_engine_bytes=20,
            engine_device_memory_bytes=30,
            determinism_tolerance_stable=True,
        ),
    )
    assert result.determinism is not None and result.determinism.input_hashes_stable
    assert result.diagnostics[0].path == "tactic-diagnostics.jsonl"
    assert model_sha256(result).startswith("sha256:")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(engine_sha256=None), "complete typed evidence"),
        (
            lambda value: value["repetitions"][0].update(index=1),
            "indexes must be contiguous",
        ),
        (
            lambda value: value["repetitions"][0]["inputs"][0].update(stable=False),
            "input integrity failed",
        ),
        (
            lambda value: value["repetitions"][0]["inputs"][0].update(source_sha256=digest("f")),
            "source input identity changed",
        ),
        (
            lambda value: value["repetitions"][0].update(
                outputs=value["repetitions"][0]["outputs"] * 2
            ),
            "unique names",
        ),
    ],
)
def test_worker_correctness_contract_rejects_inconsistent_passes(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = _worker_run_payload()
    mutate(payload)
    with pytest.raises(ValidationError, match=message):
        WorkerCorrectnessResult.model_validate(payload)


def test_failed_worker_requires_matching_typed_host_failure() -> None:
    payload = _worker_run_payload()
    payload.update(
        status="failed",
        engine_sha256=None,
        input_sha256={},
        repetitions=[],
        input_integrity_stable=None,
        memory_diagnostics=None,
        tensorrt_version=None,
        failure_code=FailureCode.EXECUTION_FAILED,
        error_type="RuntimeError",
        message="execution failed",
    )
    worker = WorkerCorrectnessResult.model_validate(payload)
    with pytest.raises(ValueError, match="classification must agree"):
        adapt_worker_run(worker, _run_context())

    failure = failure_record(FailureCode.EXECUTION_FAILED)
    result = adapt_worker_run(worker, _run_context(failure=failure))
    assert result.status is ResultStatus.FAILED
    assert result.memory is None
    assert result.determinism is None


def test_timing_blocks_require_complete_pass_or_rejection_evidence() -> None:
    base = {
        "pair_index": 0,
        "environment_id": "baseline",
        "order_in_pair": 0,
        "accepted": True,
        "rejection_reasons": [],
        "iteration_count": 1,
        "median_milliseconds": None,
        "mean_milliseconds": None,
        "coefficient_of_variation": 0.0,
        "raw_timings": None,
        "temperature_celsius": None,
        "graphics_clock_mhz": None,
        "memory_clock_mhz": None,
        "power_watts": None,
        "utilization_percent": None,
        "competing_compute_processes": [],
    }
    with pytest.raises(ValidationError, match="accepted timing blocks"):
        TimingBlock.model_validate(base)
    base.update(accepted=False)
    with pytest.raises(ValidationError, match="rejected timing blocks"):
        TimingBlock.model_validate(base)


def test_run_result_requires_matching_status_and_hardware_identity() -> None:
    worker = WorkerCorrectnessResult.model_validate(_worker_run_payload())
    result = adapt_worker_run(worker, _run_context())
    with pytest.raises(ValidationError, match="status and failure"):
        RunResult.model_validate({**result.model_dump(), "status": ResultStatus.FAILED})
    wrong_hardware = result.hardware.model_copy(update={"environment_lock_sha256": digest("f")})
    with pytest.raises(ValidationError, match="hardware environment"):
        RunResult.model_validate({**result.model_dump(), "hardware": wrong_hardware})
    with pytest.raises(ValidationError, match="timestamps must be chronological"):
        RunResult.model_validate(
            {
                **result.model_dump(),
                "ended_at": result.started_at - timedelta(seconds=1),
            }
        )
