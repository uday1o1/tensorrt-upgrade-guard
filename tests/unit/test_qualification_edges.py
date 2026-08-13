"""Qualification evidence validation and fail-closed edge tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tests.factories import digest
from upgrade_guard.compare.performance import GateOutcome
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.common import PrecisionMode
from upgrade_guard.contracts.qualification import QualificationSpec, ShapeRange
from upgrade_guard.errors import FailureCode, InfrastructureError, InvalidInputError
from upgrade_guard.qualification import (
    _block_variation_reasons,
    _jsonable,
    _observe_validity,
    _optional_float,
    _optional_int,
    _output_paths,
    _read_json,
    _status,
    _translate_container_paths,
    compare_stored_run,
)


class ObservationRunner:
    def __init__(self, gpu_result: CommandResult, processes: str = "") -> None:
        self.gpu_result = gpu_result
        self.processes = processes

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        if "--query-compute-apps=gpu_uuid,pid,process_name" in command:
            return CommandResult(command, 0, self.processes, "", 0.01)
        return self.gpu_result


def _specification() -> QualificationSpec:
    return QualificationSpec.model_validate(
        {
            "api_version": "upgradeguard.dev/v1alpha1",
            "kind": "Qualification",
            "baseline_environment_id": "baseline",
            "candidate_environment_id": "candidate",
            "environment_lock": "matrix.json",
            "corpus_lock_id": "fixture",
            "required_cases": ["tiny-transformer"],
            "precision_modes": ["fp32"],
            "optimization_profiles": [
                {
                    "id": "profile",
                    "inputs": {"x": {"minimum": [1], "optimum": [1], "maximum": [1]}},
                }
            ],
            "concrete_shapes": [{"id": "one", "inputs": {"x": [1]}}],
            "input_fixture_ids": ["input"],
            "builder": {
                "strongly_typed": True,
                "timing_cache": "disabled",
                "workspace_limit_bytes": 1,
                "optimization_level": 0,
            },
            "numerical": {
                "baseline_to_reference": {"atol": 0, "rtol": 0},
                "candidate_to_reference": {"atol": 0, "rtol": 0},
                "candidate_to_baseline": {"atol": 0, "rtol": 0},
            },
            "determinism": {
                "repetitions": 20,
                "require_bitwise": True,
                "tolerance": {"atol": 0, "rtol": 0},
            },
            "performance": {
                "warmup_milliseconds": 0,
                "measurement_milliseconds": 1,
                "minimum_accepted_pairs": 20,
                "bootstrap_replicates": 1000,
                "bootstrap_seed": 1,
                "practical_allowance": 0.05,
                "shape_allowances": {"one": 0.05},
                "shape_weights": {"one": 1.0},
                "workload_provenance": "fixture",
                "one_inference_stream": True,
                "cuda_graph": False,
            },
            "memory": {"confirmation_builds": 3},
            "hardware_validity": {
                "selected_gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111",
                "maximum_temperature_celsius": 85,
                "maximum_clock_variation_ratio": 0.1,
                "maximum_power_variation_ratio": 0.1,
                "maximum_gpu_utilization_before_block": 5,
                "reject_competing_compute_processes": True,
                "require_stable_power_limit": True,
            },
            "required_confirmations": 2,
            "reduction_budget": {
                "maximum_trials": 1,
                "maximum_seconds": 1,
                "confirmation_count": 2,
            },
            "retention": {},
        }
    )


def test_precision_specific_numerical_policy_is_bounded() -> None:
    specification = _specification()
    assert specification.numerical_policy(PrecisionMode.FP32) == specification.numerical

    payload = specification.model_dump(mode="json")
    fp16_policy = {
        "baseline_to_reference": {"atol": 0.005, "rtol": 0.005},
        "candidate_to_reference": {"atol": 0.005, "rtol": 0.005},
        "candidate_to_baseline": {"atol": 0.005, "rtol": 0.005},
    }
    payload["precision_numerical"] = {"explicit_fp16": fp16_policy}
    accepted = QualificationSpec.model_validate(payload)
    assert (
        accepted.numerical_policy(PrecisionMode.EXPLICIT_FP16).baseline_to_reference.atol == 0.005
    )

    fp16_policy["candidate_to_reference"]["atol"] = 0.02
    with pytest.raises(ValidationError, match="fp16 numerical policy exceeds"):
        QualificationSpec.model_validate(payload)

    payload["precision_numerical"] = {
        "fp32": {
            "baseline_to_reference": {"atol": 0, "rtol": 0},
            "candidate_to_reference": {"atol": 0, "rtol": 0.002},
            "candidate_to_baseline": {"atol": 0, "rtol": 0},
        }
    }
    with pytest.raises(ValidationError, match="fp32 numerical policy exceeds"):
        QualificationSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"minimum": [1], "optimum": [1, 2], "maximum": [2]}, "identical rank"),
        ({"minimum": [], "optimum": [], "maximum": []}, "rank cannot be zero"),
        (
            {"minimum": [0], "optimum": [1], "maximum": [2]},
            "0 < min <= opt <= max",
        ),
    ],
)
def test_shape_range_rejects_invalid_profiles(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ShapeRange.model_validate(payload)


def test_stored_summary_and_machine_json_fail_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    summary = run / "qualification-summary.json"
    summary.write_text("[]", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="object"):
        compare_stored_run(run)
    summary.write_text('{"schema_version":"wrong","status":"passed"}', encoding="utf-8")
    with pytest.raises(InvalidInputError, match="schema version"):
        compare_stored_run(run)
    summary.write_text(
        '{"schema_version":"upgradeguard.dev/qualification-summary/v1","status":"wrong"}',
        encoding="utf-8",
    )
    with pytest.raises(InvalidInputError, match="status"):
        compare_stored_run(run)
    summary.write_text("{", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="valid machine JSON"):
        _read_json(summary)


def test_output_evidence_rejects_schema_escape_and_hash_mutation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    value = output / "value.npy"
    np.save(value, np.ones(1, dtype=np.float32), allow_pickle=False)

    with pytest.raises(InvalidInputError, match="schema changed"):
        _output_paths({"repetitions": [{"outputs": []}]}, "output", output)
    escaped = tmp_path / "escaped.npy"
    np.save(escaped, np.ones(1, dtype=np.float32), allow_pickle=False)
    record = {
        "repetitions": [
            {"outputs": [{"name": "output", "path": str(escaped), "sha256": sha256_file(escaped)}]}
        ]
    }
    with pytest.raises(InvalidInputError, match="escaped"):
        _output_paths(record, "output", output)
    record["repetitions"][0]["outputs"][0] = {
        "name": "output",
        "path": str(value),
        "sha256": digest("f"),
    }
    with pytest.raises(InvalidInputError, match="hash differs"):
        _output_paths(record, "output", output)


def test_hardware_observation_errors_and_optional_parsing() -> None:
    specification = _specification()
    failed = CommandResult(("nvidia-smi",), 1, "", "driver error", 0.01)
    observed, reasons = _observe_validity(
        ObservationRunner(failed), specification.hardware_validity.selected_gpu_uuid, specification
    )
    assert observed == {"query_error": "driver error"}
    assert reasons == ("gpu_observation_failed",)

    malformed = CommandResult(("nvidia-smi",), 0, "too,few", "", 0.01)
    observed, reasons = _observe_validity(
        ObservationRunner(malformed),
        specification.hardware_validity.selected_gpu_uuid,
        specification,
    )
    assert observed == {"raw": "too,few"}
    assert reasons == ("gpu_observation_malformed",)
    assert _optional_float("not-a-number") is None
    assert _optional_int("not-a-number") is None
    assert _optional_float("1.25") == 1.25
    assert _optional_int("12") == 12


def test_status_json_and_path_translation_helpers(tmp_path: Path) -> None:
    assert _status((FailureCode.INFRASTRUCTURE_INVALID,)) == "infrastructure_invalid"
    assert _status((FailureCode.INCONCLUSIVE,)) == "inconclusive"
    assert _status((FailureCode.NUMERICAL_REGRESSION,)) == "failed"
    assert _status(()) == "passed"
    assert _jsonable({"outcome": GateOutcome.PASSED, "items": (GateOutcome.REGRESSION,)}) == {
        "outcome": "passed",
        "items": ["regression"],
    }
    translated = _translate_container_paths(
        {"paths": ["/output/a", "/corpus/b", "unchanged"]},
        tmp_path / "output",
        tmp_path / "corpus",
    )
    assert translated == {
        "paths": [
            str(tmp_path / "output" / "a"),
            str(tmp_path / "corpus" / "b"),
            "unchanged",
        ]
    }
    specification = _specification()
    reasons = _block_variation_reasons(
        {"graphics_clock_mhz": 2000, "power_watts": 100, "power_limit_watts": 300},
        {"graphics_clock_mhz": 1500, "power_watts": 130, "power_limit_watts": 250},
        specification,
    )
    assert reasons == (
        "graphics_clock_variation_exceeded",
        "power_variation_exceeded",
        "power_limit_changed",
    )
