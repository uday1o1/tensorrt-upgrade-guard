"""MobileNet corpus materializer image provenance tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import materialize_mobilenet_corpus
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.corpus.mobilenet import IMAGE_FIXTURES, DerivedMobileNet
from upgrade_guard.corpus.reference import ReferenceOutput


def test_materializer_retains_numeric_boundaries_and_ppm_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "corpus"

    def download(destination: Path) -> None:
        destination.write_bytes(b"source")

    def derive(source: Path, destination: Path) -> DerivedMobileNet:
        assert source.read_bytes() == b"source"
        destination.write_bytes(b"derived")
        return DerivedMobileNet(
            source_sha256="sha256:" + "1" * 64,
            derived_sha256="sha256:" + "2" * 64,
            derived_bytes=7,
            input_name="x",
            output_names=("400",),
            opset=17,
            ir_version=9,
        )

    def reference(_: Path, inputs: dict[str, np.ndarray]) -> tuple[ReferenceOutput, ...]:
        values = np.mean(inputs["x"], axis=tuple(range(1, inputs["x"].ndim)), keepdims=True)
        return (
            ReferenceOutput(
                name="400",
                dtype=str(values.dtype),
                shape=values.shape,
                sha256=sha256_bytes(values.tobytes(order="C")),
                values=values,
            ),
        )

    monkeypatch.setattr(materialize_mobilenet_corpus, "download_source", download)
    monkeypatch.setattr(materialize_mobilenet_corpus, "derive_dynamic_mobilenet", derive)
    monkeypatch.setattr(materialize_mobilenet_corpus, "run_onnx_reference", reference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(materialize_mobilenet_corpus.__file__),
            str(output),
            "--reference-lock-sha256",
            "sha256:" + ("e" * 64),
        ],
    )
    materialize_mobilenet_corpus.main()

    lock = json.loads((output / "mobilenet-corpus.lock.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in lock["cases"]}
    assert set(materialize_mobilenet_corpus.CASES).issubset(cases)
    assert set(IMAGE_FIXTURES).issubset(cases)
    for name, (_, source_sha256) in IMAGE_FIXTURES.items():
        case = cases[name]
        assert case["kind"] == "redistributable-image"
        assert case["source"]["sha256"] == source_sha256
        assert case["source"]["license"] == "Apache-2.0"
        assert case["preprocessing"] == "imagenet-rgb-nearest-v1"
        assert case["tensor_sha256"].startswith("sha256:")
        assert (output / case["source"]["corpus_path"]).is_file()
    artifact_paths = {artifact["path"] for artifact in lock["artifacts"]}
    assert "inputs/image-gradient/source.ppm" in artifact_paths
    assert "inputs/image-checkerboard/source.ppm" in artifact_paths
