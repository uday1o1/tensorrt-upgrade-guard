from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.worker.build_engine import _capturing_logger, _strongly_typed_network_flags
from upgrade_guard.worker.evidence import validate_repetitions


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


def test_gpu_workflows_do_not_upload_generated_state() -> None:
    workflow_root = Path(".github/workflows")
    for name in ("gpu-smoke.yml", "gpu-qualification.yml", "gpu-sanitizer.yml"):
        text = (workflow_root / name).read_text(encoding="utf-8")
        assert "upload-artifact" not in text
        assert ".upgrade-guard/cuda-pm" not in text
    smoke = (workflow_root / "gpu-smoke.yml").read_text(encoding="utf-8")
    for trigger in ("corpus/**", "matrices/**", "models/**", "qualification/**", "scripts/**"):
        assert f'      - "{trigger}"' in smoke


def test_worker_build_context_excludes_repository_state() -> None:
    assert Path(".dockerignore").read_text(encoding="utf-8").splitlines() == [
        "**",
        "!.dockerignore",
        "!containers/",
        "!containers/Dockerfile.worker",
        "!containers/requirements-worker.txt",
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
