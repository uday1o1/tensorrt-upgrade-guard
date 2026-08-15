from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.worker.build_engine import _capturing_logger, _strongly_typed_network_flags
from upgrade_guard.worker.evidence import validate_repetitions
from upgrade_guard.worker.run_correctness import _tactic_evidence


def _worker_result(
    root: Path,
    values: list[np.ndarray[tuple[int, ...], np.dtype[np.float32]]],
) -> tuple[Path, str, dict[str, str]]:
    engine = root / "engine.plan"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(b"trusted test engine")
    outputs = root / "case" / "outputs"
    outputs.mkdir(parents=True)
    repetitions = []
    for index, value in enumerate(values):
        path = outputs / f"output.repetition-{index:02d}.npy"
        np.save(path, value, allow_pickle=False)
        repetitions.append(
            {
                "index": index,
                "outputs": [
                    {
                        "name": "output",
                        "path": f"/output/case/outputs/{path.name}",
                        "sha256": sha256_file(path),
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                    }
                ],
            }
        )
    input_hashes = {"x": "sha256:" + "1" * 64}
    result = root / "case" / "correctness.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-correctness/v1",
                "status": "passed",
                "engine_sha256": sha256_file(engine),
                "input_sha256": input_hashes,
                "repetitions": repetitions,
            }
        ),
        encoding="utf-8",
    )
    return result, sha256_file(engine), input_hashes


def _validate(
    root: Path,
    result: Path,
    engine_hash: str,
    input_hashes: dict[str, str],
    expected: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    *,
    atol: float = 0.0,
) -> dict[str, object]:
    return validate_repetitions(
        result_path=result,
        runs_root=root,
        expected_output_name="output",
        expected=expected,
        atol=atol,
        rtol=0.0,
        expected_engine_sha256=engine_hash,
        expected_input_hashes=input_hashes,
    )


def test_validates_every_repetition_and_recorded_hash(tmp_path: Path) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)

    evidence = _validate(tmp_path, result, engine_hash, input_hashes, expected)

    assert evidence["repetitions"] == 20
    assert evidence["bitwise_stable"] is True
    assert evidence["tolerance_stable"] is True
    output_hashes = evidence["output_sha256"]
    assert isinstance(output_hashes, list)
    assert len(output_hashes) == 20


def test_three_way_mode_retains_numerical_failure_without_preempting_classification(
    tmp_path: Path,
) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    observed = np.asarray([2.0, 3.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [observed.copy()] * 20)

    evidence = validate_repetitions(
        result_path=result,
        runs_root=tmp_path,
        expected_output_name="output",
        expected=expected,
        atol=0.0,
        rtol=0.0,
        expected_engine_sha256=engine_hash,
        expected_input_hashes=input_hashes,
        enforce_numerical_gates=False,
    )

    assert evidence["maximum_absolute_error"] == 1.0
    assert evidence["tolerance_stable"] is True


def test_validates_named_input_integrity_for_every_repetition(tmp_path: Path) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)
    payload = json.loads(result.read_text(encoding="utf-8"))
    value_hash = "sha256:" + "2" * 64
    payload["input_integrity_stable"] = True
    for repetition in payload["repetitions"]:
        repetition["inputs"] = [
            {
                "name": "x",
                "source_sha256": input_hashes["x"],
                "host_value_sha256": value_hash,
                "device_value_sha256": value_hash,
                "stable": True,
            }
        ]
    result.write_text(json.dumps(payload), encoding="utf-8")

    evidence = _validate(tmp_path, result, engine_hash, input_hashes, expected)
    assert evidence["input_integrity_stable"] is True

    payload["repetitions"][7]["inputs"][0]["device_value_sha256"] = "sha256:" + "3" * 64
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity hashes"):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="unexpected"), "schema changed"),
        (lambda payload: payload.update(status="failed"), "did not pass"),
        (lambda payload: payload.update(engine_sha256="sha256:" + "0" * 64), "engine hash"),
        (lambda payload: payload.update(input_sha256={}), "input hashes"),
        (lambda payload: payload.update(repetitions=[]), "repetition count"),
        (
            lambda payload: payload["repetitions"][0].update(index=4),
            "repetition indexes",
        ),
        (
            lambda payload: payload["repetitions"][0].update(outputs=[]),
            "output inventory",
        ),
        (
            lambda payload: payload["repetitions"][0]["outputs"][0].update(name="wrong"),
            "output name",
        ),
    ],
)
def test_worker_result_contract_fails_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)
    payload = json.loads(result.read_text(encoding="utf-8"))
    mutation(payload)
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ([], "inventory changed"),
        ([{"name": 1}], "record is invalid"),
        (
            [
                {
                    "name": "x",
                    "source_sha256": "sha256:" + "1" * 64,
                    "host_value_sha256": "sha256:" + "2" * 64,
                    "device_value_sha256": "sha256:" + "2" * 64,
                    "stable": False,
                }
            ],
            "integrity failed",
        ),
        (
            [
                {
                    "name": "other",
                    "source_sha256": "sha256:" + "1" * 64,
                    "host_value_sha256": "sha256:" + "2" * 64,
                    "device_value_sha256": "sha256:" + "2" * 64,
                    "stable": True,
                }
            ],
            "repetition input hashes",
        ),
    ],
)
def test_named_input_integrity_contract_rejects_invalid_records(
    tmp_path: Path,
    inputs: list[object],
    message: str,
) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["input_integrity_stable"] = True
    for repetition in payload["repetitions"]:
        repetition["inputs"] = inputs
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)


def test_worker_result_rejects_unsafe_output_paths_and_non_objects(tmp_path: Path) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["repetitions"][0]["outputs"][0]["path"] = "/output/../secret.npy"
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="escaped /output"):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)

    result.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)

    result.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)


def test_tactic_diagnostic_binds_engine_and_activation_shape(tmp_path: Path) -> None:
    path = tmp_path / "tactic-diagnostics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": "upgradeguard.dev/plugin-tactic/v1",
                    "event": "enqueue",
                    "tactic": "kVECTORIZED_WARP",
                    "rows": 34,
                    "hidden": 259,
                }
            )
            for _ in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    digest = "sha256:" + "1" * 64
    evidence = _tactic_evidence(
        path,
        digest,
        {
            "x": np.zeros((2, 17, 259), dtype=np.float32),
            "residual": np.zeros((2, 17, 259), dtype=np.float32),
            "gamma": np.zeros((259,), dtype=np.float32),
        },
        expected_enqueue_count=20,
    )
    assert evidence["engine_sha256"] == digest
    assert evidence["selected_tactic"] == "kVECTORIZED_WARP"
    assert evidence["enqueue_count"] == 20


def test_retained_tactic_diagnostic_is_validated_with_correctness(tmp_path: Path) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)
    diagnostic = tmp_path / "case" / "tactic.jsonl"
    diagnostic.write_text("selected tactic\n", encoding="utf-8")
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["tactic_diagnostic"] = {
        "path": "/output/case/tactic.jsonl",
        "sha256": sha256_file(diagnostic),
        "bytes": diagnostic.stat().st_size,
        "engine_sha256": engine_hash,
        "selected_tactic": "kVECTORIZED_WARP",
        "rows": 1,
        "hidden": 2,
        "enqueue_count": 20,
    }
    result.write_text(json.dumps(payload), encoding="utf-8")

    evidence = validate_repetitions(
        result_path=result,
        runs_root=tmp_path,
        expected_output_name="output",
        expected=expected,
        atol=0.0,
        rtol=0.0,
        expected_engine_sha256=engine_hash,
        expected_input_hashes=input_hashes,
        require_tactic_diagnostic=True,
    )
    assert evidence["tactic_diagnostic"] == payload["tactic_diagnostic"]

    payload["tactic_diagnostic"]["enqueue_count"] = 19
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selected-tactic evidence differs"):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)


@pytest.mark.parametrize(
    "value",
    [
        "not-an-object",
        {},
        {"path": "relative.jsonl"},
    ],
)
def test_invalid_tactic_diagnostics_fail_closed(tmp_path: Path, value: object) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["tactic_diagnostic"] = value
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected-tactic"):
        _validate(tmp_path, result, engine_hash, input_hashes, expected)


def test_required_tactic_diagnostic_cannot_be_omitted(tmp_path: Path) -> None:
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, [expected.copy()] * 20)

    with pytest.raises(RuntimeError, match="omitted selected-tactic"):
        validate_repetitions(
            result_path=result,
            runs_root=tmp_path,
            expected_output_name="output",
            expected=expected,
            atol=0.0,
            rtol=0.0,
            expected_engine_sha256=engine_hash,
            expected_input_hashes=input_hashes,
            require_tactic_diagnostic=True,
        )


@pytest.mark.parametrize("failure", ["hash", "metadata", "nonfinite", "determinism"])
def test_fails_closed_on_invalid_repetition_evidence(tmp_path: Path, failure: str) -> None:
    expected = np.zeros(2, dtype=np.float32)
    values = [expected.copy() for _ in range(20)]
    if failure == "determinism":
        values[0] = np.full(2, -0.75, dtype=np.float32)
        values[1] = np.full(2, 0.75, dtype=np.float32)
    result, engine_hash, input_hashes = _worker_result(tmp_path, values)
    payload = json.loads(result.read_text(encoding="utf-8"))
    output = payload["repetitions"][1]["outputs"][0]
    if failure == "hash":
        output["sha256"] = "sha256:" + "0" * 64
    elif failure == "metadata":
        output["shape"] = [3]
    elif failure == "nonfinite":
        path = tmp_path / "case" / "outputs" / "output.repetition-01.npy"
        np.save(path, np.asarray([np.nan, 0.0], dtype=np.float32), allow_pickle=False)
        output["sha256"] = sha256_file(path)
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError):
        _validate(
            tmp_path,
            result,
            engine_hash,
            input_hashes,
            expected,
            atol=1.0 if failure == "determinism" else 0.0,
        )


def test_strong_typing_is_required() -> None:
    supported = SimpleNamespace(NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=3))
    unsupported = SimpleNamespace(NetworkDefinitionCreationFlag=SimpleNamespace())

    assert _strongly_typed_network_flags(supported) == 8
    with pytest.raises(RuntimeError, match="STRONGLY_TYPED"):
        _strongly_typed_network_flags(unsupported)


def test_builder_logger_retains_structured_warnings() -> None:
    class ILogger:
        INTERNAL_ERROR = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
        VERBOSE = 4

    logger = _capturing_logger(SimpleNamespace(ILogger=ILogger), verbose=False)
    logger.log(ILogger.INFO, "ignored detail")
    logger.log(ILogger.WARNING, "bounded warning")
    assert logger.messages == [{"severity": "WARNING", "message": "bounded warning"}]


def test_worker_build_context_excludes_repository_state() -> None:
    assert Path(".dockerignore").read_text(encoding="utf-8").splitlines() == [
        "**",
        "!.dockerignore",
        "!containers/",
        "!containers/Dockerfile.worker",
        "!containers/requirements-worker.txt",
        "!containers/Dockerfile.reference",
        "!containers/requirements-reference.txt",
    ]
    dockerfile = Path("containers/Dockerfile.worker").read_text(encoding="utf-8")
    requirements = Path("containers/requirements-worker.txt").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--hash=sha256:" in requirements


def test_worker_numpy_lock_satisfies_pinned_ngc_numba_metadata() -> None:
    requirements_in = Path("containers/requirements-worker.in").read_text(encoding="utf-8")
    requirements_lock = Path("containers/requirements-worker.txt").read_text(encoding="utf-8")
    locked_line = next(
        line for line in requirements_lock.splitlines() if line.startswith("numpy==")
    )
    locked_version = locked_line.removeprefix("numpy==").split()[0]
    assert f"numpy=={locked_version}" in requirements_in
    assert Version(locked_version) in SpecifierSet(">=1.22,<2.5")
