"""Frozen model generation, CPU reference, and materialization tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.corpus.generators import generate_plugin_micrograph, generate_tiny_transformer
from upgrade_guard.corpus.materialize import load_recipe, materialize_corpus
from upgrade_guard.corpus.mobilenet import (
    derive_dynamic_mobilenet,
    deterministic_image_input,
    download_source,
)
from upgrade_guard.corpus.plugin import CASES, materialize_plugin_corpus
from upgrade_guard.corpus.reference import (
    deterministic_transformer_inputs,
    residual_rmsnorm_reference,
    run_onnx_reference,
)
from upgrade_guard.errors import InvalidInputError

FP32_SHA = "sha256:16dd39f7df92632a0d9268b0b669ee8e110d0bce6b2da189fd046e3b4d2e71b4"
FP16_SHA = "sha256:46687e2f106b1439655e641944a2bb251f40cc7ae673814fa81a383b8f5ec2d5"


def test_transformer_models_are_locked_valid_and_cpu_executable(tmp_path: Path) -> None:
    for precision, expected in (("fp32", FP32_SHA), ("fp16", FP16_SHA)):
        model_path = tmp_path / f"transformer-{precision}.onnx"
        generated = generate_tiny_transformer(model_path, precision=precision)  # type: ignore[arg-type]
        assert generated.sha256 == expected
        onnx.checker.check_model(model_path, full_check=True)
        inputs = deterministic_transformer_inputs(1, 8, precision=precision)
        outputs = run_onnx_reference(model_path, inputs)
        assert [(item.name, item.dtype, item.shape) for item in outputs] == [
            ("output", "float32" if precision == "fp32" else "float16", (1, 8, 256))
        ]


def test_generation_is_byte_reproducible(tmp_path: Path) -> None:
    first = generate_tiny_transformer(tmp_path / "first.onnx")
    second = generate_tiny_transformer(tmp_path / "second.onnx")
    assert first.sha256 == second.sha256 == FP32_SHA
    assert first.path.read_bytes() == second.path.read_bytes()


def test_plugin_graph_and_reference_cover_tails_and_validation(tmp_path: Path) -> None:
    graph = generate_plugin_micrograph(tmp_path / "plugin.onnx")
    assert graph.bytes > 0
    x = np.arange(2 * 7, dtype=np.float16).reshape(2, 7) / 10
    residual = np.flip(x, axis=-1).copy()
    gamma = np.linspace(0.5, 1.5, 7, dtype=np.float32)
    output = residual_rmsnorm_reference(x, residual, gamma, epsilon=1e-5)
    expected = (x.astype(np.float32) + residual.astype(np.float32)) * gamma
    expected /= np.sqrt(
        np.mean(
            (x.astype(np.float32) + residual.astype(np.float32)) ** 2,
            axis=-1,
            keepdims=True,
        )
        + 1e-5
    )
    np.testing.assert_allclose(output, expected.astype(np.float16), atol=1e-3, rtol=1e-3)
    with pytest.raises(InvalidInputError, match="epsilon"):
        residual_rmsnorm_reference(x, residual, gamma, epsilon=0)
    with pytest.raises(InvalidInputError, match="gamma"):
        residual_rmsnorm_reference(x, residual, gamma.astype(np.float16), epsilon=1e-5)


def test_smoke_recipe_materializes_atomically_and_rejects_mutation(tmp_path: Path) -> None:
    recipe = tmp_path / "corpus.yaml"
    recipe.write_text(
        f"""api_version: upgradeguard.dev/v1alpha1
kind: CorpusRecipe
id: smoke
generator_version: tiny-transformer-v1
precisions: [fp32]
expected_model_sha256:
  fp32: {FP32_SHA}
transformer_shapes:
  - id: b1_s8
    batch: 1
    sequence: 8
    weight: 1.0
""",
        encoding="utf-8",
    )
    destination = tmp_path / "materialized"
    lock = materialize_corpus(recipe, destination)
    assert lock.id == "smoke"
    assert (destination / "corpus.lock.json").is_file()
    assert len(lock.artifacts) == 4
    with pytest.raises(InvalidInputError, match="already exists"):
        materialize_corpus(recipe, destination)

    recipe.write_text(recipe.read_text().replace("id: smoke", "id: smoke\nunknown: true"))
    with pytest.raises(InvalidInputError, match="corpus recipe is invalid"):
        load_recipe(recipe)


def test_plugin_corpus_is_complete_and_locked(tmp_path: Path) -> None:
    destination = tmp_path / "plugin"
    lock = materialize_plugin_corpus(destination)
    assert lock["cases"] == [case.id for case in CASES]
    assert len(lock["artifacts"]) == 62
    for artifact in lock["artifacts"]:
        path = destination / str(artifact["path"])
        assert path.stat().st_size == artifact["bytes"]
        assert sha256_file(path) == artifact["sha256"]
    assert (
        np.load(
            destination / "fp32" / "minimum-zero-h7" / "expected.npy",
            allow_pickle=False,
        ).sum()
        == 0
    )
    with pytest.raises(InvalidInputError, match="already exists"):
        materialize_plugin_corpus(destination)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        result = self.payload[self.offset : self.offset + size]
        self.offset += len(result)
        return result


class _Opener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def open(self, request: object, timeout: int) -> _Response:
        del request, timeout
        return _Response(self.payload)


def test_mobilenet_download_derivation_and_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["output"])],
        "mobilenet-fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])],
    )
    payload = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    ).SerializeToString()
    expected_sha = "sha256:" + hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr("upgrade_guard.corpus.mobilenet.SOURCE_BYTES", len(payload))
    monkeypatch.setattr("upgrade_guard.corpus.mobilenet.SOURCE_SHA256", expected_sha)
    monkeypatch.setattr(
        "upgrade_guard.corpus.mobilenet.urllib.request.build_opener",
        lambda *args: _Opener(payload),
    )
    source = tmp_path / "source.onnx"
    download_source(source)
    derived = tmp_path / "dynamic.onnx"
    identity = derive_dynamic_mobilenet(source, derived)
    assert identity.source_sha256 == expected_sha
    model = onnx.load(derived)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    assert [dimensions[index].dim_param for index in (0, 2, 3)] == [
        "batch",
        "height",
        "width",
    ]
    first = deterministic_image_input(1, 225, 223)
    second = deterministic_image_input(1, 225, 223)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(InvalidInputError, match="outside the locked profile"):
        deterministic_image_input(1, 159, 224)
