"""Both-worker target-readiness validator tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import validate_target_readiness as readiness
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_bytes, sha256_file
from upgrade_guard.corpus.reference import ReferenceOutput


def _save(path: Path, value: np.ndarray[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def _build_manifest(root: Path, model: Path, identity: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    engine = root / "engine.plan"
    inspector = root / "inspector.json"
    timing_cache = root / "timing.cache"
    engine.write_bytes(f"engine-{identity}".encode())
    inspector.write_text("[]\n", encoding="utf-8")
    timing_cache.write_bytes(f"cache-{identity}".encode())
    command = ["python3", "-m", "upgrade_guard.worker.build_engine", "--engine", str(engine)]
    (root / "build.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-build/v1",
                "status": "passed",
                "model": {"path": f"/corpus/{model.name}", "sha256": sha256_file(model)},
                "engine": {
                    "path": f"/output/{identity}/engine.plan",
                    "sha256": sha256_file(engine),
                    "bytes": engine.stat().st_size,
                    "device_memory_bytes": 1024,
                },
                "inspector": {
                    "path": f"/output/{identity}/inspector.json",
                    "sha256": sha256_file(inspector),
                },
                "timing_cache": {
                    "path": f"/output/{identity}/timing.cache",
                    "input_sha256": None,
                    "output_sha256": sha256_file(timing_cache),
                },
                "parser_errors": [],
                "strongly_typed": True,
                "tensorrt_version": "test-version",
                "duration_seconds": 1.0,
                "command": command,
                "command_sha256": command_sha256(command),
            }
        ),
        encoding="utf-8",
    )


def _correctness(
    *,
    runs: Path,
    case_root: Path,
    engine: Path,
    inputs: dict[str, Path],
    output_name: str,
    expected: np.ndarray[Any, Any],
    repetitions: int = 2,
) -> None:
    outputs = case_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    retained = []
    for index in range(repetitions):
        output = outputs / f"{output_name}.repetition-{index:02d}.npy"
        np.save(output, expected.copy(), allow_pickle=False)
        retained.append(
            {
                "index": index,
                "outputs": [
                    {
                        "name": output_name,
                        "path": "/output/" + output.relative_to(runs).as_posix(),
                        "sha256": sha256_file(output),
                        "dtype": str(expected.dtype),
                        "shape": list(expected.shape),
                    }
                ],
            }
        )
    (case_root / "correctness.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-correctness/v1",
                "status": "passed",
                "engine_sha256": sha256_file(engine),
                "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
                "repetitions": retained,
            }
        ),
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    core = root / "core"
    plugin = root / "plugin-corpus"
    mobilenet = root / "mobilenet-corpus"
    runs = root / "runs"
    output = root / "readiness.json"
    standard_model = core / "models" / "tiny-transformer-fp32.onnx"
    plugin_model = plugin / "residual-rmsnorm-fp32.onnx"
    mobilenet_model = mobilenet / "mobilenetv3-small-075-dynamic.onnx"
    for path, content in (
        (standard_model, b"standard-model"),
        (plugin_model, b"plugin-model"),
        (mobilenet_model, b"mobilenet-model"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    standard_expected: dict[str, np.ndarray[Any, Any]] = {}
    for index, case in enumerate(readiness.STANDARD_CASES, start=1):
        case_root = core / "inputs" / "tiny-transformer-fp32" / case
        tokens = np.full((1, index, 2), float(index), dtype=np.float32)
        mask = np.zeros((1, 1, 1, index), dtype=np.float32)
        _save(case_root / "tokens.npy", tokens)
        _save(case_root / "mask.npy", mask)
        standard_expected[case] = tokens

    plugin_case = plugin / "fp32" / readiness.PLUGIN_CASE
    plugin_expected = np.ones((1, 2, 3), dtype=np.float32)
    for name, value in (
        ("x", np.ones_like(plugin_expected)),
        ("residual", np.zeros_like(plugin_expected)),
        ("gamma", np.ones(3, dtype=np.float32)),
        ("expected", plugin_expected),
    ):
        _save(plugin_case / f"{name}.npy", value)

    mobilenet_case = mobilenet / "inputs" / readiness.MOBILENET_CASE
    mobilenet_expected = np.ones((1, 4), dtype=np.float32)
    _save(mobilenet_case / "x.npy", np.ones((1, 3, 2, 2), dtype=np.float32))
    _save(mobilenet_case / "expected.npy", mobilenet_expected)
    (mobilenet / "mobilenet-corpus.lock.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/mobilenet-corpus/v1",
                "cases": [{"id": readiness.MOBILENET_CASE, "output_name": "400"}],
            }
        ),
        encoding="utf-8",
    )

    for environment in readiness.ENVIRONMENTS:
        standard_root = runs / environment / "standard"
        plugin_root = runs / environment / "plugin" / "fp32"
        mobilenet_root = runs / environment / "mobilenet"
        _build_manifest(standard_root, standard_model, f"{environment}/standard")
        _build_manifest(plugin_root, plugin_model, f"{environment}/plugin/fp32")
        _build_manifest(mobilenet_root, mobilenet_model, f"{environment}/mobilenet")
        for case in readiness.STANDARD_CASES:
            corpus_case = core / "inputs" / "tiny-transformer-fp32" / case
            _correctness(
                runs=runs,
                case_root=standard_root / case,
                engine=standard_root / "engine.plan",
                inputs={name: corpus_case / f"{name}.npy" for name in ("tokens", "mask")},
                output_name="output",
                expected=standard_expected[case],
            )
        _correctness(
            runs=runs,
            case_root=plugin_root / readiness.PLUGIN_CASE,
            engine=plugin_root / "engine.plan",
            inputs={name: plugin_case / f"{name}.npy" for name in ("x", "residual", "gamma")},
            output_name="output",
            expected=plugin_expected,
        )
        _correctness(
            runs=runs,
            case_root=mobilenet_root / readiness.MOBILENET_CASE,
            engine=mobilenet_root / "engine.plan",
            inputs={"x": mobilenet_case / "x.npy"},
            output_name="400",
            expected=mobilenet_expected,
        )
    return core, plugin, mobilenet, runs, output


def _run(
    monkeypatch: pytest.MonkeyPatch,
    core: Path,
    plugin: Path,
    mobilenet: Path,
    runs: Path,
    output: Path,
) -> None:
    def reference(_: Path, inputs: dict[str, np.ndarray[Any, Any]]) -> tuple[ReferenceOutput, ...]:
        value = inputs["tokens"]
        return (
            ReferenceOutput(
                name="output",
                dtype=str(value.dtype),
                shape=value.shape,
                sha256=sha256_bytes(value.tobytes(order="C")),
                values=value,
            ),
        )

    monkeypatch.setattr(readiness, "run_onnx_reference", reference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(readiness.__file__),
            "--core-corpus",
            str(core),
            "--plugin-corpus",
            str(plugin),
            "--mobilenet-corpus",
            str(mobilenet),
            "--runs",
            str(runs),
            "--output",
            str(output),
        ],
    )
    readiness.main()


def test_validates_exact_both_worker_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    core, plugin, mobilenet, runs, output = _fixture(tmp_path)
    _run(monkeypatch, core, plugin, mobilenet, runs, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["repetitions"] == 2
    assert payload["environment_inventory"] == ["baseline", "candidate"]
    assert payload["workload_inventory"] == [
        "standard/b1_s8",
        "standard/b1_s128",
        "plugin/fp32/tail-random-h259",
        "mobilenet/minimum",
    ]
    assert [item["environment"] for item in payload["environments"]] == [
        "baseline",
        "candidate",
    ]
    for environment in payload["environments"]:
        assert set(environment["builds"]) == {"standard", "plugin_fp32", "mobilenet"}
        assert all(build["strongly_typed"] for build in environment["builds"].values())
        assert all(case["repetitions"] == 2 for case in environment["workloads"])


@pytest.mark.parametrize(
    "failure",
    ["inventory", "build", "engine_hash", "input_hash", "schema", "nonfinite", "determinism"],
)
def test_fails_closed_on_incomplete_or_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    core, plugin, mobilenet, runs, output = _fixture(tmp_path)
    result = runs / "candidate" / "plugin" / "fp32" / readiness.PLUGIN_CASE / "correctness.json"
    if failure == "inventory":
        (runs / "unexpected").mkdir()
    elif failure == "build":
        path = runs / "candidate" / "mobilenet" / "build.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "failed"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif failure in {"engine_hash", "input_hash", "schema"}:
        payload = json.loads(result.read_text(encoding="utf-8"))
        if failure == "engine_hash":
            payload["engine_sha256"] = "sha256:" + "0" * 64
        elif failure == "input_hash":
            payload["input_sha256"]["x"] = "sha256:" + "0" * 64
        else:
            payload["schema_version"] = "unexpected"
        result.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = json.loads(result.read_text(encoding="utf-8"))
        expected = np.ones((1, 2, 3), dtype=np.float32)
        values = (
            [np.full_like(expected, np.nan), expected]
            if failure == "nonfinite"
            else [expected - 0.0001, expected + 0.0001]
        )
        for index, value in enumerate(values):
            output_path = (
                runs
                / "candidate"
                / "plugin"
                / "fp32"
                / readiness.PLUGIN_CASE
                / "outputs"
                / f"output.repetition-{index:02d}.npy"
            )
            np.save(output_path, value, allow_pickle=False)
            payload["repetitions"][index]["outputs"][0]["sha256"] = sha256_file(output_path)
        result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError):
        _run(monkeypatch, core, plugin, mobilenet, runs, output)
    assert not output.exists()


def test_repetition_count_is_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    core, plugin, mobilenet, runs, output = _fixture(tmp_path)
    result = runs / "baseline" / "mobilenet" / readiness.MOBILENET_CASE / "correctness.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["repetitions"].pop()
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="repetition count"):
        _run(monkeypatch, core, plugin, mobilenet, runs, output)
