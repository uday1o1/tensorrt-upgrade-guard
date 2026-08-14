"""Authored-policy plugin and MobileNet validator tests."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from scripts import validate_mobilenet_outputs, validate_plugin_outputs
from scripts.generate_remote_evidence import _validate_extended_typed_chains
from tests.factories import digest, environment_lock
from upgrade_guard.containers.commands import command_sha256
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.build import BuildManifest
from upgrade_guard.contracts.case import CaseManifest, SourceAttribution
from upgrade_guard.contracts.common import ArtifactReference, PrecisionMode, TensorContract
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.contracts.extended import (
    ExtendedCorpusCase,
    ExtendedCorpusManifest,
    ExtendedCorpusModel,
)
from upgrade_guard.contracts.results import RunResult

PROJECT = Path(__file__).resolve().parents[2]
REPETITIONS = 20
ZERO_SHA = "sha256:" + "0" * 64


def _save(path: Path, value: np.ndarray[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def _specification(path: Path, *, require_top1: bool = False) -> None:
    value = yaml.safe_load((PROJECT / "qualification/full.yaml").read_text(encoding="utf-8"))
    if require_top1:
        value["numerical"].update(
            {
                "baseline_to_reference": {"atol": 0.0001, "rtol": 0},
                "candidate_to_reference": {"atol": 0.0001, "rtol": 0},
                "candidate_to_baseline": {"atol": 0.0001, "rtol": 0},
                "require_top1_agreement": True,
                "require_top5_agreement": True,
            }
        )
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _worker_result(
    *,
    runs: Path,
    case_root: Path,
    engine: Path,
    input_hashes: dict[str, str],
    output_name: str,
    value: np.ndarray[Any, Any],
) -> None:
    outputs = case_root / "outputs"
    outputs.mkdir(parents=True)
    tactic_diagnostic = case_root / "tactic-diagnostics.jsonl"
    tactic_diagnostic.write_text(
        '{"hidden":6,"rows":2,"selected_tactic":"kVECTORIZED_WARP"}\n',
        encoding="utf-8",
    )
    repetitions = []
    for index in range(REPETITIONS):
        output = outputs / f"{output_name}.repetition-{index:02d}.npy"
        np.save(output, value, allow_pickle=False)
        repetitions.append(
            {
                "index": index,
                "outputs": [
                    {
                        "name": output_name,
                        "path": "/output/" + output.relative_to(runs).as_posix(),
                        "sha256": sha256_file(output),
                        "bytes": output.stat().st_size,
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                    }
                ],
                "inputs": [
                    {
                        "name": name,
                        "source_sha256": value,
                        "host_value_sha256": value,
                        "device_value_sha256": value,
                        "stable": True,
                    }
                    for name, value in sorted(input_hashes.items())
                ],
            }
        )
    command = ("python3", "-m", "upgrade_guard.worker.run_correctness")
    (case_root / "correctness.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-correctness/v1",
                "status": "passed",
                "command": command,
                "command_sha256": command_sha256(command),
                "engine_sha256": sha256_file(engine),
                "input_sha256": input_hashes,
                "repetitions": repetitions,
                "input_integrity_stable": True,
                "tactic_diagnostic": {
                    "path": "/output/" + tactic_diagnostic.relative_to(runs).as_posix(),
                    "sha256": sha256_file(tactic_diagnostic),
                    "bytes": tactic_diagnostic.stat().st_size,
                    "engine_sha256": sha256_file(engine),
                    "selected_tactic": "kVECTORIZED_WARP",
                    "rows": 2,
                    "hidden": 6,
                    "enqueue_count": REPETITIONS,
                },
                "memory_diagnostics": {"execution_context_device_memory_bytes": 64},
                "tensorrt_version": "11.2.1",
                "started_unix_seconds": 1.0,
                "ended_unix_seconds": 2.0,
                "duration_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    main: Callable[[], None],
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [main.__module__, *arguments])
    main()


def _artifact(root: Path, path: Path, media_type: str) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        media_type=media_type,
    )


def _matrix_inputs(root: Path) -> list[str]:
    baseline = environment_lock(environment_id="baseline", worker_manifest_character="1")
    candidate = environment_lock(environment_id="candidate", worker_manifest_character="2")
    matrix = MatrixLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="EnvironmentLock",
        source_matrix_sha256=digest("3"),
        gpu_uuid=baseline.probe.gpu.uuid,
        created_at=baseline.probed_at,
        environments=(baseline, candidate),
        lock_sha256=ZERO_SHA,
    )
    matrix = matrix.model_copy(update={"lock_sha256": matrix.computed_sha256()})
    matrix_path = root / "matrix.lock.json"
    matrix_path.write_text(matrix.model_dump_json(indent=2) + "\n", encoding="utf-8")
    commit_path = root / "source.commit"
    commit_path.write_text("1" * 40 + "\n", encoding="utf-8")
    return [
        "--project-root",
        str(PROJECT),
        "--matrix-lock",
        str(matrix_path),
        "--source-commit-file",
        str(commit_path),
    ]


def _promotion_arguments(root: Path, *, plugin: bool = False) -> list[str]:
    arguments = _matrix_inputs(root)
    if plugin:
        arguments.extend(
            [
                "--plugin-build-root",
                str(root / "plugin-build"),
                "--plugin-build-log",
                str(root / "plugin-build.log"),
            ]
        )
    return arguments


def _write_build(
    *,
    runs: Path,
    build_root: Path,
    model: Path,
    engine: Path,
) -> None:
    inspector = build_root / "inspector.json"
    cache = build_root / "timing.cache"
    inspector.write_text("{}\n", encoding="utf-8")
    cache.write_bytes(b"cache")
    command = ("python3", "-m", "upgrade_guard.worker.build_engine")
    (build_root / "build.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/worker-build/v1",
                "status": "passed",
                "command": command,
                "command_sha256": command_sha256(command),
                "model": {
                    "path": f"/corpus/{model.name}",
                    "sha256": sha256_file(model),
                    "bytes": model.stat().st_size,
                },
                "engine": {
                    "path": "/output/" + engine.relative_to(runs).as_posix(),
                    "sha256": sha256_file(engine),
                    "bytes": engine.stat().st_size,
                    "device_memory_bytes": 64,
                },
                "memory_diagnostics": {},
                "inspector": {
                    "path": "/output/" + inspector.relative_to(runs).as_posix(),
                    "sha256": sha256_file(inspector),
                    "bytes": inspector.stat().st_size,
                },
                "timing_cache": {
                    "path": "/output/" + cache.relative_to(runs).as_posix(),
                    "input_sha256": None,
                    "output_sha256": sha256_file(cache),
                    "bytes": cache.stat().st_size,
                },
                "builder_configuration": {"strongly_typed": "true"},
                "timing_cache_state": "cold",
                "tensorrt_version": "11.2.1",
                "started_unix_seconds": 1.0,
                "ended_unix_seconds": 2.0,
                "duration_seconds": 1.0,
                "strongly_typed": True,
            }
        ),
        encoding="utf-8",
    )


def _write_extended_manifest(
    *,
    suite: str,
    corpus: Path,
    cases: tuple[ExtendedCorpusCase, ...],
    models: tuple[ExtendedCorpusModel, ...],
) -> None:
    manifest = ExtendedCorpusManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ExtendedCorpusManifest",
        suite=suite,
        reference_environment_sha256=digest("e"),
        models=models,
        cases=cases,
        manifest_sha256=ZERO_SHA,
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest.computed_sha256()})
    path = corpus / "extended-corpus-manifest.json"
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    lock_path = corpus / f"{suite}-corpus.lock.json"
    current = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    current.update(
        {
            "schema_version": f"upgradeguard.dev/{suite}-corpus/v1",
            "extended_manifest": {
                "path": path.relative_to(corpus).as_posix(),
                "sha256": sha256_file(path),
                "manifest_sha256": manifest.manifest_sha256,
            },
        }
    )
    lock_path.write_text(json.dumps(current), encoding="utf-8")


def _source() -> SourceAttribution:
    return SourceAttribution(
        name="fixture",
        source_url="https://example.invalid/model",
        source_revision="fixture",
        license_name="Apache-2.0",
        license_url="https://example.invalid/license",
        redistribution_allowed=True,
    )


def _plugin_fixture(
    root: Path,
    *,
    candidate_fp32: np.ndarray[Any, Any] | None = None,
) -> tuple[Path, Path, Path, Path]:
    corpus = root / "corpus"
    runs = root / "plugin-runs"
    specification = root / "qualification.yaml"
    output = root / "validation.json"
    _specification(specification)
    extended_models = []
    extended_cases = []
    for precision, dtype in (("fp32", np.float32), ("fp16", np.float16)):
        precision_mode = PrecisionMode.FP32 if precision == "fp32" else PrecisionMode.EXPLICIT_FP16
        model = corpus / f"residual-rmsnorm-{precision}.onnx"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(f"model-{precision}".encode())
        extended_models.append(
            ExtendedCorpusModel(
                model_id=f"residual-rmsnorm-plugin-{precision}",
                precision=precision_mode,
                artifact=_artifact(corpus, model, "application/onnx"),
                source=_source(),
                opset=17,
                ir_version=9,
                profile_id="residual-rmsnorm-dynamic",
                reference_runner="project_formula",
                semantic_policy={"comparison": "elementwise"},
            )
        )
        case = corpus / precision / "tail"
        expected = np.ones((1, 2, 6), dtype=dtype)
        for name, value in (
            ("x", expected),
            ("residual", np.zeros_like(expected)),
            ("gamma", np.ones(6, dtype=np.float32)),
            ("expected", expected),
        ):
            _save(case / f"{name}.npy", value)
        inputs = {name: sha256_file(case / f"{name}.npy") for name in ("x", "residual", "gamma")}
        tensor_dtype = "float32" if precision == "fp32" else "float16"
        extended_cases.append(
            ExtendedCorpusCase(
                id=f"{precision}-tail",
                model_id=f"residual-rmsnorm-plugin-{precision}",
                precision=precision_mode,
                shape_id="tail",
                profile_id="residual-rmsnorm-dynamic",
                inputs=(
                    TensorContract(name="x", dtype=tensor_dtype, shape=tuple(expected.shape)),
                    TensorContract(
                        name="residual", dtype=tensor_dtype, shape=tuple(expected.shape)
                    ),
                    TensorContract(name="gamma", dtype="float32", shape=(6,)),
                ),
                input_fixtures=tuple(
                    _artifact(corpus, case / f"{name}.npy", "application/x-npy")
                    for name in ("x", "residual", "gamma")
                ),
                outputs=(
                    TensorContract(name="output", dtype=tensor_dtype, shape=tuple(expected.shape)),
                ),
                reference_output=_artifact(corpus, case / "expected.npy", "application/x-npy"),
                workload_weight=1.0,
            )
        )
        for environment in ("baseline", "candidate"):
            engine = runs / environment / precision / "engine.plan"
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.write_bytes(f"{environment}-{precision}".encode())
            _write_build(
                runs=runs,
                build_root=engine.parent,
                model=model,
                engine=engine,
            )
            _worker_result(
                runs=runs,
                case_root=runs / environment / precision / "tail",
                engine=engine,
                input_hashes=inputs,
                output_name="output",
                value=(
                    candidate_fp32
                    if environment == "candidate"
                    and precision == "fp32"
                    and candidate_fp32 is not None
                    else expected
                ),
            )
            binary = runs / environment / "libupgrade_guard_residual_rmsnorm.so"
            binary.write_bytes(f"plugin-{environment}".encode())
    _write_extended_manifest(
        suite="plugin",
        corpus=corpus,
        cases=tuple(extended_cases),
        models=tuple(extended_models),
    )
    plugin_build = root / "plugin-build"
    for environment in ("baseline", "candidate"):
        compile_commands = plugin_build / environment / "build" / "compile_commands.json"
        compile_commands.parent.mkdir(parents=True)
        compile_commands.write_text("[]\n", encoding="utf-8")
    (root / "plugin-build.log").write_text("compile and tests passed\n", encoding="utf-8")
    return corpus, runs, specification, output


def test_plugin_validator_emits_authored_three_way_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus, runs, specification, output = _plugin_fixture(tmp_path)
    _invoke(
        monkeypatch,
        validate_plugin_outputs.main,
        [
            "--corpus",
            str(corpus),
            "--runs",
            str(runs),
            "--specification",
            str(specification),
            "--output",
            str(output),
            *_promotion_arguments(tmp_path, plugin=True),
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "upgradeguard.dev/plugin-validation/v2"
    assert payload["repetitions"] == REPETITIONS
    for case in payload["cases"]:
        assert case["numerical"]["effective_policy"]["relative_error_guard"] == 1e-12
        assert len(case["numerical"]["repetitions"]) == REPETITIONS
        assert set(case["numerical"]["repetitions"][0]) >= {
            "baseline_to_reference",
            "candidate_to_reference",
            "candidate_to_baseline",
        }
        for chain in case["stable_artifacts"].values():
            case_manifest = CaseManifest.model_validate_json(
                (runs / chain["case_manifest"]["path"]).read_text(encoding="utf-8")
            )
            build = BuildManifest.model_validate_json(
                (runs / chain["build_manifest"]["path"]).read_text(encoding="utf-8")
            )
            run = RunResult.model_validate_json(
                (runs / chain["run_result"]["path"]).read_text(encoding="utf-8")
            )
            assert case_manifest.computed_sha256() == case_manifest.manifest_sha256
            assert build.case_manifest_sha256 == case_manifest.manifest_sha256
            assert run.build_manifest_sha256 == chain["build_manifest"]["sha256"]
    matrix = MatrixLock.model_validate_json(
        (tmp_path / "matrix.lock.json").read_text(encoding="utf-8")
    )
    indexed = _validate_extended_typed_chains(
        tmp_path,
        payload,
        suite="plugin",
        matrix=matrix,
        specification_sha256=sha256_file(specification),
        source_commit="1" * 40,
        corpus_lock_sha256=sha256_file(corpus / "plugin-corpus.lock.json"),
    )
    assert indexed["chain_count"] == 4


@pytest.mark.parametrize(
    ("candidate", "expected_code"),
    [
        (np.full((1, 2, 6), 1.5, dtype=np.float32), "NUMERICAL_REGRESSION"),
        (
            np.asarray([[[np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]] * 2], dtype=np.float32),
            "NONFINITE_OUTPUT",
        ),
    ],
    ids=("numerical", "nonfinite"),
)
def test_plugin_validator_atomically_retains_typed_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate: np.ndarray[Any, Any],
    expected_code: str,
) -> None:
    corpus, runs, specification, output = _plugin_fixture(
        tmp_path,
        candidate_fp32=candidate,
    )

    with pytest.raises(RuntimeError, match=expected_code):
        _invoke(
            monkeypatch,
            validate_plugin_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path, plugin=True),
            ],
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_code"] == expected_code
    assert payload["failure"]["code"] == expected_code
    assert payload["failure"]["environment_id"] == "candidate"
    assert payload["cases"][0]["failure_code"] == expected_code
    assert len(payload["cases"][0]["numerical"]["repetitions"]) == REPETITIONS
    candidate_chain = payload["cases"][0]["stable_artifacts"]["candidate"]
    candidate_run = RunResult.model_validate_json(
        (runs / candidate_chain["run_result"]["path"]).read_text(encoding="utf-8")
    )
    assert candidate_run.failure is not None
    assert candidate_run.failure.code.value == expected_code


def test_mobilenet_candidate_schema_drift_retains_typed_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = np.asarray([[1.0, 0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    corpus, runs, specification, output = _mobilenet_fixture(
        tmp_path,
        candidate=candidate,
        require_top1=True,
    )

    with pytest.raises(RuntimeError, match="OUTPUT_SCHEMA_CHANGED"):
        _invoke(
            monkeypatch,
            validate_mobilenet_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path),
            ],
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failure_code"] == "OUTPUT_SCHEMA_CHANGED"
    assert payload["failure"]["code"] == "OUTPUT_SCHEMA_CHANGED"
    assert payload["failure"]["environment_id"] == "candidate"
    candidate_chain = payload["cases"][0]["stable_artifacts"]["candidate"]
    candidate_run = RunResult.model_validate_json(
        (runs / candidate_chain["run_result"]["path"]).read_text(encoding="utf-8")
    )
    assert candidate_run.failure is not None
    assert candidate_run.failure.code.value == "OUTPUT_SCHEMA_CHANGED"


def _mobilenet_fixture(
    root: Path,
    *,
    candidate: np.ndarray[Any, Any],
    require_top1: bool,
    baseline: np.ndarray[Any, Any] | None = None,
) -> tuple[Path, Path, Path, Path]:
    corpus = root / "corpus"
    runs = root / "mobilenet-runs"
    specification = root / "qualification.yaml"
    output = root / "validation.json"
    _specification(specification, require_top1=require_top1)
    case = corpus / "inputs" / "minimum"
    expected = np.asarray([[1.0, 0.99995, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    _save(case / "x.npy", np.ones((1, 3, 2, 2), dtype=np.float32))
    _save(case / "expected.npy", expected)
    model = corpus / "mobilenetv3-small-075-dynamic.onnx"
    model.write_bytes(b"mobilenet-model")
    (corpus / "mobilenet-corpus.lock.json").write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/mobilenet-corpus/v1",
                "cases": [{"id": "minimum", "output_name": "400"}],
            }
        ),
        encoding="utf-8",
    )
    inputs = {"x": sha256_file(case / "x.npy")}
    for environment, value in (
        ("baseline", expected if baseline is None else baseline),
        ("candidate", candidate),
    ):
        engine = runs / environment / "engine.plan"
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(environment.encode())
        _write_build(
            runs=runs,
            build_root=engine.parent,
            model=model,
            engine=engine,
        )
        _worker_result(
            runs=runs,
            case_root=runs / environment / "minimum",
            engine=engine,
            input_hashes=inputs,
            output_name="400",
            value=value,
        )
    _write_extended_manifest(
        suite="mobilenet",
        corpus=corpus,
        cases=(
            ExtendedCorpusCase(
                id="minimum",
                model_id="mobilenetv3-small-075-dynamic",
                precision=PrecisionMode.FP32,
                shape_id="minimum",
                profile_id="mobilenet-dynamic",
                inputs=(TensorContract(name="x", dtype="float32", shape=(1, 3, 2, 2)),),
                input_fixtures=(_artifact(corpus, case / "x.npy", "application/x-npy"),),
                outputs=(TensorContract(name="400", dtype="float32", shape=(1, 6)),),
                reference_output=_artifact(corpus, case / "expected.npy", "application/x-npy"),
                workload_weight=1.0,
            ),
        ),
        models=(
            ExtendedCorpusModel(
                model_id="mobilenetv3-small-075-dynamic",
                precision=PrecisionMode.FP32,
                artifact=_artifact(corpus, model, "application/onnx"),
                source=_source(),
                opset=17,
                ir_version=9,
                profile_id="mobilenet-dynamic",
                reference_runner="onnxruntime_cpu",
                semantic_policy={
                    "comparison": "classification",
                    "top1": "required",
                    "top5": "required",
                },
            ),
        ),
    )
    return corpus, runs, specification, output


def test_mobilenet_validator_retains_three_way_classification_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = np.asarray([[1.0, 0.99995, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    corpus, runs, specification, output = _mobilenet_fixture(
        tmp_path,
        candidate=expected,
        require_top1=False,
    )
    _invoke(
        monkeypatch,
        validate_mobilenet_outputs.main,
        [
            "--corpus",
            str(corpus),
            "--runs",
            str(runs),
            "--specification",
            str(specification),
            "--output",
            str(output),
            *_promotion_arguments(tmp_path),
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    numerical = payload["cases"][0]["numerical"]
    assert payload["schema_version"] == "upgradeguard.dev/mobilenet-validation/v2"
    assert numerical["semantic_kind"] == "classification"
    assert numerical["repetitions"][0]["candidate_to_reference"]["top1_agreement"] is True
    assert numerical["repetitions"][0]["candidate_to_reference"]["top5_agreement"] is True
    matrix = MatrixLock.model_validate_json(
        (tmp_path / "matrix.lock.json").read_text(encoding="utf-8")
    )
    indexed = _validate_extended_typed_chains(
        tmp_path,
        payload,
        suite="mobilenet",
        matrix=matrix,
        specification_sha256=sha256_file(specification),
        source_commit="1" * 40,
        corpus_lock_sha256=sha256_file(corpus / "mobilenet-corpus.lock.json"),
    )
    assert indexed["chain_count"] == 2


def test_mobilenet_authored_top1_gate_fails_even_within_elementwise_tolerance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = np.asarray([[0.99995, 1.0, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    corpus, runs, specification, output = _mobilenet_fixture(
        tmp_path,
        candidate=candidate,
        require_top1=True,
    )
    with pytest.raises(RuntimeError, match="three-way gate failed"):
        _invoke(
            monkeypatch,
            validate_mobilenet_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path),
            ],
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_code"] == "NUMERICAL_REGRESSION"
    assert payload["failure"]["code"] == "NUMERICAL_REGRESSION"
    assert payload["failure"]["environment_id"] == "candidate"
    assert payload["cases"][0]["failure_code"] == "NUMERICAL_REGRESSION"


def test_mobilenet_validator_classifies_baseline_reference_failure_as_corpus_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = np.asarray([[1.0, 0.99995, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    baseline = np.asarray([[0.2, 0.1, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    corpus, runs, specification, output = _mobilenet_fixture(
        tmp_path,
        candidate=expected,
        baseline=baseline,
        require_top1=True,
    )

    with pytest.raises(SystemExit) as exit_info:
        _invoke(
            monkeypatch,
            validate_mobilenet_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path),
            ],
        )
    assert exit_info.value.code == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_code"] == "CORPUS_INVALID"
    assert payload["failure"]["code"] == "CORPUS_INVALID"
    assert payload["failure"]["environment_id"] == "baseline"


def test_mobilenet_validator_preserves_typed_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = np.asarray([[1.0, 0.99995, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    corpus, runs, specification, output = _mobilenet_fixture(
        tmp_path,
        candidate=expected,
        require_top1=True,
    )
    result_path = runs / "candidate" / "minimum" / "correctness.json"
    worker = json.loads(result_path.read_text(encoding="utf-8"))
    worker["status"] = "failed"
    worker["failure_code"] = "EXECUTION_FAILED"
    worker["error_type"] = "RuntimeError"
    worker["message"] = "typed worker execution failure"
    result_path.write_text(json.dumps(worker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="EXECUTION_FAILED"):
        _invoke(
            monkeypatch,
            validate_mobilenet_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path),
            ],
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_code"] == "EXECUTION_FAILED"
    assert payload["failure"]["code"] == "EXECUTION_FAILED"
    stable = payload["cases"][0]["stable_artifacts"]["candidate"]
    run = RunResult.model_validate_json(
        (runs / stable["run_result"]["path"]).read_text(encoding="utf-8")
    )
    assert run.status.value == "failed"
    assert run.failure is not None and run.failure.code.value == "EXECUTION_FAILED"


def test_plugin_validator_promotes_strict_failed_worker_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus, runs, specification, output = _plugin_fixture(tmp_path)
    result_path = runs / "candidate" / "fp32" / "tail" / "correctness.json"
    worker = json.loads(result_path.read_text(encoding="utf-8"))
    worker.update(
        {
            "status": "failed",
            "failure_code": "EXECUTION_FAILED",
            "error_type": "RuntimeError",
            "message": "typed worker execution failure",
        }
    )
    result_path.write_text(json.dumps(worker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="EXECUTION_FAILED"):
        _invoke(
            monkeypatch,
            validate_plugin_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path, plugin=True),
            ],
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    stable = payload["cases"][0]["stable_artifacts"]["candidate"]
    run = RunResult.model_validate_json(
        (runs / stable["run_result"]["path"]).read_text(encoding="utf-8")
    )
    assert run.status.value == "failed"
    assert run.failure is not None and run.failure.code.value == "EXECUTION_FAILED"


def test_plugin_validator_promotes_strict_failed_build_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus, runs, specification, output = _plugin_fixture(tmp_path)
    result_path = runs / "candidate" / "fp32" / "build.json"
    worker = json.loads(result_path.read_text(encoding="utf-8"))
    worker.update(
        {
            "status": "failed",
            "failure_code": "ENGINE_BUILD_FAILED",
            "error_type": "RuntimeError",
            "message": "typed worker build failure",
        }
    )
    result_path.write_text(json.dumps(worker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ENGINE_BUILD_FAILED"):
        _invoke(
            monkeypatch,
            validate_plugin_outputs.main,
            [
                "--corpus",
                str(corpus),
                "--runs",
                str(runs),
                "--specification",
                str(specification),
                "--output",
                str(output),
                *_promotion_arguments(tmp_path, plugin=True),
            ],
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    stable = payload["cases"][0]["stable_artifacts"]["candidate"]
    assert stable["run_result"] is None
    build = BuildManifest.model_validate_json(
        (runs / stable["build_manifest"]["path"]).read_text(encoding="utf-8")
    )
    assert build.status.value == "failed"
    assert build.failure is not None and build.failure.code.value == "ENGINE_BUILD_FAILED"
