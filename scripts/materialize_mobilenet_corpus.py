"""Materialize and CPU-reference the pinned dynamic MobileNet corpus."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.case import SourceAttribution
from upgrade_guard.contracts.common import ArtifactReference, PrecisionMode, TensorContract
from upgrade_guard.contracts.extended import (
    ExtendedCorpusCase,
    ExtendedCorpusManifest,
    ExtendedCorpusModel,
)
from upgrade_guard.corpus.mobilenet import (
    IMAGE_FIXTURES,
    SOURCE_REVISION,
    SOURCE_URL,
    derive_dynamic_mobilenet,
    deterministic_image_input,
    download_source,
    preprocess_ppm_fixture,
)
from upgrade_guard.corpus.reference import run_onnx_reference

CASES = {
    "minimum": (1, 160, 160),
    "optimum": (8, 224, 224),
    "maximum": (16, 320, 320),
    "odd-spatial": (1, 225, 223),
    "batch-boundary": (16, 160, 160),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference-lock-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise RuntimeError("refusing to overwrite MobileNet corpus")
    arguments.output.mkdir(parents=True)
    source = arguments.output / "source.onnx"
    model = arguments.output / "mobilenetv3-small-075-dynamic.onnx"
    if arguments.source is None:
        download_source(source)
    else:
        shutil.copyfile(arguments.source, source)
    identity = derive_dynamic_mobilenet(source, model)
    artifacts = [artifact(arguments.output, source), artifact(arguments.output, model)]
    cases: list[dict[str, object]] = []
    extended_cases: list[ExtendedCorpusCase] = []
    case_count = len(CASES) + len(IMAGE_FIXTURES)
    for name, (batch, height, width) in CASES.items():
        case_root = arguments.output / "inputs" / name
        case_root.mkdir(parents=True)
        input_value = deterministic_image_input(batch, height, width)
        input_path = case_root / "x.npy"
        np.save(input_path, input_value, allow_pickle=False)
        outputs = run_onnx_reference(model, {"x": input_value})
        repeated = run_onnx_reference(model, {"x": input_value})
        if outputs[0].sha256 != repeated[0].sha256 or not np.array_equal(
            outputs[0].values, repeated[0].values
        ):
            raise RuntimeError("MobileNet reference is not bitwise deterministic")
        expected_path = case_root / "expected.npy"
        np.save(expected_path, outputs[0].values, allow_pickle=False)
        artifacts.extend(
            [artifact(arguments.output, input_path), artifact(arguments.output, expected_path)]
        )
        extended_cases.append(
            _extended_case(
                arguments.output,
                name,
                input_path,
                expected_path,
                output_name=outputs[0].name,
                output_dtype=outputs[0].dtype,
                output_shape=outputs[0].shape,
                workload_weight=1.0 / case_count,
            )
        )
        cases.append(
            {
                "id": name,
                "kind": "deterministic-numeric",
                "shape": [batch, 3, height, width],
                "output_name": outputs[0].name,
                "output_shape": outputs[0].shape,
                "reference_repetitions": 2,
                "reference_bitwise_deterministic": True,
            }
        )
    project_root = Path(__file__).resolve().parents[1]
    for name, (relative_source, source_sha256) in IMAGE_FIXTURES.items():
        case_root = arguments.output / "inputs" / name
        case_root.mkdir(parents=True)
        authored_source = project_root / relative_source
        source_path = case_root / "source.ppm"
        shutil.copyfile(authored_source, source_path)
        preprocessed = preprocess_ppm_fixture(source_path, source_sha256)
        input_path = case_root / "x.npy"
        np.save(input_path, preprocessed.values, allow_pickle=False)
        outputs = run_onnx_reference(model, {"x": preprocessed.values})
        repeated = run_onnx_reference(model, {"x": preprocessed.values})
        if outputs[0].sha256 != repeated[0].sha256 or not np.array_equal(
            outputs[0].values, repeated[0].values
        ):
            raise RuntimeError("MobileNet reference is not bitwise deterministic")
        expected_path = case_root / "expected.npy"
        np.save(expected_path, outputs[0].values, allow_pickle=False)
        artifacts.extend(
            [
                artifact(arguments.output, source_path),
                artifact(arguments.output, input_path),
                artifact(arguments.output, expected_path),
            ]
        )
        extended_cases.append(
            _extended_case(
                arguments.output,
                name,
                input_path,
                expected_path,
                output_name=outputs[0].name,
                output_dtype=outputs[0].dtype,
                output_shape=outputs[0].shape,
                workload_weight=1.0 / case_count,
            )
        )
        cases.append(
            {
                "id": name,
                "kind": "redistributable-image",
                "shape": list(preprocessed.values.shape),
                "output_name": outputs[0].name,
                "output_shape": outputs[0].shape,
                "source": {
                    "authored_path": relative_source.as_posix(),
                    "corpus_path": source_path.relative_to(arguments.output).as_posix(),
                    "sha256": preprocessed.source_sha256,
                    "license": "Apache-2.0",
                },
                "preprocessing": preprocessed.preprocessing,
                "tensor_sha256": preprocessed.tensor_sha256,
                "reference_repetitions": 2,
                "reference_bitwise_deterministic": True,
            }
        )
    zero = "sha256:" + ("0" * 64)
    extended_manifest = ExtendedCorpusManifest(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ExtendedCorpusManifest",
        suite="mobilenet",
        reference_environment_sha256=arguments.reference_lock_sha256,
        models=(
            ExtendedCorpusModel(
                model_id="mobilenetv3-small-075-dynamic",
                precision=PrecisionMode.FP32,
                artifact=artifact_reference(arguments.output, model),
                source=SourceAttribution(
                    name="ONNX Models MobileNetV3 Small 0.75",
                    source_url=SOURCE_URL,
                    source_revision=SOURCE_REVISION,
                    license_name="Apache-2.0",
                    license_url="https://github.com/onnx/models/blob/main/LICENSE",
                    redistribution_allowed=False,
                ),
                opset=identity.opset,
                ir_version=identity.ir_version,
                profile_id="mobilenet-dynamic",
                reference_runner="onnxruntime_cpu",
                semantic_policy={
                    "comparison": "classification",
                    "top1": "required",
                    "top5": "required",
                },
            ),
        ),
        cases=tuple(extended_cases),
        manifest_sha256=zero,
    )
    extended_manifest = extended_manifest.model_copy(
        update={"manifest_sha256": extended_manifest.computed_sha256()}
    )
    extended_manifest_path = arguments.output / "extended-corpus-manifest.json"
    extended_manifest_path.write_text(
        json.dumps(
            extended_manifest.model_dump(mode="json"),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(artifact(arguments.output, extended_manifest_path))
    lock = {
        "schema_version": "upgradeguard.dev/mobilenet-corpus/v1",
        "source_sha256": identity.source_sha256,
        "derived_sha256": identity.derived_sha256,
        "reference_environment_sha256": arguments.reference_lock_sha256,
        "extended_manifest": {
            "path": extended_manifest_path.relative_to(arguments.output).as_posix(),
            "sha256": sha256_file(extended_manifest_path),
            "manifest_sha256": extended_manifest.manifest_sha256,
        },
        "cases": cases,
        "artifacts": artifacts,
    }
    (arguments.output / "mobilenet-corpus.lock.json").write_text(
        json.dumps(lock, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": media_type(path),
    }


def artifact_reference(root: Path, path: Path) -> ArtifactReference:
    return ArtifactReference.model_validate(artifact(root, path))


def media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".npy": "application/x-npy",
        ".onnx": "application/onnx",
        ".ppm": "image/x-portable-pixmap",
    }.get(path.suffix, "application/octet-stream")


def _extended_case(
    root: Path,
    name: str,
    input_path: Path,
    expected_path: Path,
    *,
    output_name: str,
    output_dtype: str,
    output_shape: tuple[int, ...],
    workload_weight: float,
) -> ExtendedCorpusCase:
    input_value = np.load(input_path, allow_pickle=False)
    return ExtendedCorpusCase(
        id=name,
        model_id="mobilenetv3-small-075-dynamic",
        precision=PrecisionMode.FP32,
        shape_id=name,
        profile_id="mobilenet-dynamic",
        inputs=(
            TensorContract(
                name="x",
                dtype=cast(Any, str(input_value.dtype)),
                shape=tuple(input_value.shape),
            ),
        ),
        input_fixtures=(artifact_reference(root, input_path),),
        outputs=(
            TensorContract(
                name=output_name,
                dtype=cast(Any, output_dtype),
                shape=output_shape,
            ),
        ),
        reference_output=artifact_reference(root, expected_path),
        workload_weight=workload_weight,
    )


if __name__ == "__main__":
    main()
