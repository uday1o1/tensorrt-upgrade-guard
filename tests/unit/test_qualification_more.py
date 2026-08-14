"""Additional deterministic qualification helper and error-branch tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from tests.factories import digest, run_result
from upgrade_guard.compare.numerical import compare_arrays
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.contracts.common import NumericalTolerance, Phase, PrecisionMode
from upgrade_guard.contracts.results import RunResult
from upgrade_guard.corpus.registry import CorpusLock, MaterializedArtifact
from upgrade_guard.errors import (
    FailureCode,
    InfrastructureError,
    InvalidInputError,
    UnsupportedEnvironmentError,
)
from upgrade_guard.qualification import (
    QualificationRunner,
    _block_variation_reasons,
    _compare_correctness_case,
    _corpus_root,
    _finalize_run_result,
    _load_model,
    _load_specification,
    _materialize_case_manifests,
    _observe_validity,
    _qualification_project_root,
    _read_json_array,
    _repetition_input_hashes_stable,
    _tensor_dtype,
    _verify_corpus,
    _worker_failure_record,
    _worker_tolerance_stable,
)


def _specification():
    return _load_specification(Path("qualification/full.yaml"))


def _input_run() -> dict[str, object]:
    return {
        "input_sha256": {"x": digest("1"), "mask": digest("2")},
        "repetitions": [
            {
                "inputs": [
                    {
                        "name": "x",
                        "source_sha256": digest("1"),
                        "host_value_sha256": digest("3"),
                        "device_value_sha256": digest("3"),
                        "stable": True,
                    },
                    {
                        "name": "mask",
                        "source_sha256": digest("2"),
                        "host_value_sha256": digest("4"),
                        "device_value_sha256": digest("4"),
                        "stable": True,
                    },
                ]
            }
        ],
    }


def test_repetition_input_hashes_require_complete_named_integrity() -> None:
    valid = _input_run()
    assert _repetition_input_hashes_stable(valid)

    mutations = []
    missing_top_level = copy.deepcopy(valid)
    missing_top_level["input_sha256"] = {}
    mutations.append(missing_top_level)
    missing_inputs = copy.deepcopy(valid)
    missing_inputs["repetitions"] = [{}]
    mutations.append(missing_inputs)
    wrong_count = copy.deepcopy(valid)
    wrong_count["repetitions"][0]["inputs"].pop()  # type: ignore[index,union-attr]
    mutations.append(wrong_count)
    duplicate = copy.deepcopy(valid)
    duplicate["repetitions"][0]["inputs"][1]["name"] = "x"  # type: ignore[index]
    mutations.append(duplicate)
    source_changed = copy.deepcopy(valid)
    source_changed["repetitions"][0]["inputs"][0]["source_sha256"] = digest("f")  # type: ignore[index]
    mutations.append(source_changed)
    device_changed = copy.deepcopy(valid)
    device_changed["repetitions"][0]["inputs"][0]["device_value_sha256"] = digest("f")  # type: ignore[index]
    mutations.append(device_changed)

    assert all(not _repetition_input_hashes_stable(item) for item in mutations)


def test_worker_tolerance_requires_outputs_and_uses_integrity(tmp_path: Path) -> None:
    tolerance = NumericalTolerance(atol=0, rtol=0)
    assert not _worker_tolerance_stable({}, tmp_path, tolerance)
    assert not _worker_tolerance_stable({"repetitions": [{}]}, tmp_path, tolerance)
    assert not _worker_tolerance_stable(
        {"repetitions": [{"outputs": [{"name": "y"}, {"name": "y"}]}]},
        tmp_path,
        tolerance,
    )

    run = _input_run()
    for index, repetition in enumerate(run["repetitions"]):  # type: ignore[union-attr]
        path = tmp_path / f"y-{index}.npy"
        np.save(path, np.asarray([1.0], dtype=np.float32), allow_pickle=False)
        repetition["outputs"] = [  # type: ignore[index]
            {"name": "y", "path": str(path), "sha256": _sha256_file(path)}
        ]
    assert _worker_tolerance_stable(run, tmp_path, tolerance)

    run["repetitions"][0]["inputs"][0]["stable"] = False  # type: ignore[index]
    assert _worker_tolerance_stable(run, tmp_path, tolerance)
    assert not _repetition_input_hashes_stable(run)


def test_worker_failure_record_rejects_inconsistent_or_escaped_evidence(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    assert (
        _worker_failure_record(
            None,
            None,
            phase=Phase.BUILD,
            environment_id="candidate",
            precision="fp32",
            shape_id=None,
            result_path=result,
            output_root=tmp_path,
        )
        is None
    )
    with pytest.raises(InfrastructureError, match="unexpectedly contains"):
        _worker_failure_record(
            None,
            "error",
            phase=Phase.BUILD,
            environment_id="candidate",
            precision="fp32",
            shape_id=None,
            result_path=result,
            output_root=tmp_path,
        )
    with pytest.raises(InfrastructureError, match="omitted"):
        _worker_failure_record(
            FailureCode.ENGINE_BUILD_FAILED,
            None,
            phase=Phase.BUILD,
            environment_id="candidate",
            precision="fp32",
            shape_id=None,
            result_path=result,
            output_root=tmp_path,
        )
    outside = tmp_path.parent / "outside-worker-result.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="escaped"):
        _worker_failure_record(
            FailureCode.EXECUTION_FAILED,
            "execution failed",
            phase=Phase.CORRECTNESS,
            environment_id="candidate",
            precision="fp32",
            shape_id="b1_s8",
            result_path=outside,
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("code", "phase", "gate"),
    [
        (FailureCode.NONDETERMINISM_REGRESSION, "determinism", "determinism"),
        (FailureCode.NUMERICAL_REGRESSION, "correctness", "three_way_numerical"),
    ],
)
def test_finalize_run_result_records_specific_failure_phase(
    tmp_path: Path, code: FailureCode, phase: str, gate: str
) -> None:
    tolerance = NumericalTolerance(atol=0, rtol=0)
    metric = compare_arrays(
        "output",
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        tolerance,
        relative_error_guard=1e-12,
    )
    record = {
        "environment_id": "candidate",
        "run_result": run_result().model_dump(mode="json"),
    }
    finalized = _finalize_run_result(
        record,
        numerical=(metric,),
        failure_code=code,
        precision="fp32",
        shape_id="b1_s8",
        output_root=tmp_path,
    )
    typed = RunResult.model_validate(finalized)
    assert typed.failure is not None
    assert typed.failure.phase.value == phase
    assert typed.failure.gate == gate
    assert (tmp_path / "candidate/fp32/b1_s8/run-result.json").is_file()


def test_project_root_resolution_rejects_invalid_and_discovers_checkout(tmp_path: Path) -> None:
    specification = tmp_path / "project" / "nested" / "qualification.yaml"
    specification.parent.mkdir(parents=True)
    specification.write_text("kind: Qualification\n", encoding="utf-8")
    (tmp_path / "project" / "BUILD_PLAN.md").write_text("plan\n", encoding="utf-8")
    (tmp_path / "project" / "src").mkdir()
    assert _qualification_project_root(specification, None) == tmp_path / "project"

    with pytest.raises(InvalidInputError, match="real directory"):
        _qualification_project_root(specification, specification)
    isolated = tmp_path / "isolated" / "qualification.yaml"
    isolated.parent.mkdir()
    isolated.write_text("kind: Qualification\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="could not locate"):
        _qualification_project_root(isolated, None)


def test_reference_array_and_authored_model_loaders_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="reference evidence"):
        _read_json_array(path)
    path.write_text('[{"ok": true}, 1]', encoding="utf-8")
    with pytest.raises(InvalidInputError, match="object array"):
        _read_json_array(path)
    with pytest.raises(InvalidInputError, match="qualification specification"):
        _load_specification(path)
    with pytest.raises(InvalidInputError, match="environment lock"):
        _load_model(path, CorpusLock, "environment lock")


def test_verify_corpus_rejects_materializer_inventory_and_hash_drift(tmp_path: Path) -> None:
    root = tmp_path / ("a" * 64)
    root.mkdir()
    artifact = root / "model.bin"
    artifact.write_bytes(b"model")
    lock = CorpusLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="CorpusLock",
        id="fixture",
        recipe_sha256=digest("1"),
        reference_environment_sha256=digest("2"),
        artifacts=(
            MaterializedArtifact(
                path="model.bin",
                sha256=_sha256_file(artifact),
                bytes=artifact.stat().st_size,
                media_type="application/octet-stream",
            ),
        ),
    )
    _verify_corpus(root, lock)

    (root / "extra.bin").write_bytes(b"extra")
    with pytest.raises(InvalidInputError, match="inventory"):
        _verify_corpus(root, lock)
    (root / "extra.bin").unlink()
    artifact.write_bytes(b"other")
    with pytest.raises(InvalidInputError, match="artifact differs"):
        _verify_corpus(root, lock)
    artifact.write_bytes(b"model")
    (root / "materializer.json").write_text(
        json.dumps({"materializer_sha256": digest("f")}), encoding="utf-8"
    )
    with pytest.raises(InvalidInputError, match="materializer identity"):
        _verify_corpus(root, lock)


def test_case_manifest_materialization_rejects_invalid_lock_and_missing_model(
    tmp_path: Path,
) -> None:
    specification = _specification()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus.lock.json").write_text("{}", encoding="utf-8")
    shape_map = {shape.id: shape for shape in specification.concrete_shapes}
    with pytest.raises(InvalidInputError, match="invalid for case manifests"):
        _materialize_case_manifests(
            specification,
            corpus,
            tmp_path / "output",
            PrecisionMode.FP32,
            "fp32",
            shape_map,
        )

    lock = CorpusLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="CorpusLock",
        id="fixture",
        recipe_sha256=digest("1"),
        reference_environment_sha256=digest("2"),
        artifacts=(),
    )
    (corpus / "corpus.lock.json").write_text(lock.model_dump_json(), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="model is absent"):
        _materialize_case_manifests(
            specification,
            corpus,
            tmp_path / "output",
            PrecisionMode.FP32,
            "fp32",
            shape_map,
        )


def test_execute_rejects_unsupported_case_shape_and_precision_before_workers(
    tmp_path: Path,
) -> None:
    specification = _specification()
    runner = QualificationRunner(source_root=tmp_path)
    with pytest.raises(UnsupportedEnvironmentError, match="tiny-transformer"):
        runner._execute(specification, None, {}, tmp_path, tmp_path / "out")  # type: ignore[arg-type]

    tiny_only = specification.model_copy(update={"required_cases": ("tiny-transformer",)})
    mismatched = tiny_only.model_copy(
        update={
            "performance": tiny_only.performance.model_copy(
                update={"shape_weights": {"missing": 1.0}}
            )
        }
    )
    with pytest.raises(InvalidInputError, match="performance weights"):
        runner._execute(mismatched, None, {}, tmp_path, tmp_path / "out")  # type: ignore[arg-type]

    qdq = tiny_only.model_copy(update={"precision_modes": (PrecisionMode.QDQ,)})
    with pytest.raises(UnsupportedEnvironmentError, match="Q/DQ"):
        runner._execute(qdq, None, {}, tmp_path, tmp_path / "out")  # type: ignore[arg-type]


def test_corpus_root_rejects_unsafe_missing_and_non_directory_paths(tmp_path: Path) -> None:
    specification = _specification()
    default = specification.model_copy(update={"corpus_root": None})
    with pytest.raises(InvalidInputError, match="unavailable"):
        _corpus_root(tmp_path, default)

    for authored in ("../escape", str(tmp_path / "absolute")):
        unsafe = specification.model_copy(update={"corpus_root": authored})
        with pytest.raises(InvalidInputError, match="project-relative"):
            _corpus_root(tmp_path, unsafe)

    file_path = tmp_path / "corpus-file"
    file_path.write_text("not a directory", encoding="utf-8")
    not_directory = specification.model_copy(update={"corpus_root": "corpus-file"})
    with pytest.raises(InvalidInputError, match="escaped"):
        _corpus_root(tmp_path, not_directory)


def test_correctness_comparison_applies_integrity_before_reference_loading(
    tmp_path: Path,
) -> None:
    specification = _specification()
    baseline = {
        "worker": {"input_integrity_stable": False},
        "run_result": {},
    }
    candidate = {
        "worker": {"input_integrity_stable": True},
        "run_result": {},
    }
    evidence, failure = _compare_correctness_case(
        specification,
        tmp_path,
        tmp_path,
        "fp32",
        PrecisionMode.FP32,
        "b1_s8",
        baseline,
        candidate,
    )
    assert failure is FailureCode.CORPUS_INVALID
    assert evidence["input_integrity"]["baseline"] is False


def test_correctness_comparison_rejects_reference_schema_and_identity_drift(
    tmp_path: Path,
) -> None:
    specification = _specification()
    reference = tmp_path / "reference"
    reference.mkdir()
    stable = {
        "worker": {"input_integrity_stable": True},
        "run_result": {},
    }
    metadata = reference / "tiny-transformer-fp32-b1_s8.json"
    metadata.write_text("[]", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="schema is invalid"):
        _compare_correctness_case(
            specification,
            tmp_path,
            tmp_path,
            "fp32",
            PrecisionMode.FP32,
            "b1_s8",
            stable,
            stable,
        )

    value = np.asarray([1.0], dtype=np.float32)
    np.save(reference / "tiny-transformer-fp32-b1_s8-output.npy", value, allow_pickle=False)
    metadata.write_text(
        json.dumps(
            [
                {
                    "name": "output",
                    "dtype": "float32",
                    "shape": [1],
                    "sha256": digest("f"),
                    "repetitions": 2,
                    "bitwise_deterministic": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(InvalidInputError, match="locked evidence"):
        _compare_correctness_case(
            specification,
            tmp_path,
            tmp_path,
            "fp32",
            PrecisionMode.FP32,
            "b1_s8",
            stable,
            stable,
        )


def test_dtype_validity_and_observation_failure_branches() -> None:
    assert _tensor_dtype("float32") == "float32"
    with pytest.raises(InvalidInputError, match="unsupported"):
        _tensor_dtype("uint16")

    specification = _specification()
    gpu = CommandResult(
        ("nvidia-smi",),
        0,
        (f"{specification.hardware_validity.selected_gpu_uuid}, 40, 2000, 9000, 100, 300, 0\n"),
        "",
        0.01,
    )
    runner = _ObservationRunner(gpu, process_returncode=1)
    _, reasons = _observe_validity(
        runner,
        specification.hardware_validity.selected_gpu_uuid,
        specification,
    )
    assert reasons == ("process_observation_failed",)

    missing = _block_variation_reasons({}, {}, specification)
    assert "graphics_clock_mhz_observation_missing" in missing
    assert "power_watts_observation_missing" in missing
    assert "power_limit_observation_missing" in missing


class _ObservationRunner:
    def __init__(self, gpu: CommandResult, *, process_returncode: int) -> None:
        self.gpu = gpu
        self.process_returncode = process_returncode

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        if "--query-compute-apps=gpu_uuid,pid,process_name" in command:
            return CommandResult(command, self.process_returncode, "", "failed", 0.01)
        return self.gpu


def _sha256_file(path: Path) -> str:
    from upgrade_guard.contracts.base import sha256_file

    return sha256_file(path)
