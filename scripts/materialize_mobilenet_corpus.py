"""Materialize and CPU-reference the pinned dynamic MobileNet corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.corpus.mobilenet import (
    derive_dynamic_mobilenet,
    deterministic_image_input,
    download_source,
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
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise RuntimeError("refusing to overwrite MobileNet corpus")
    arguments.output.mkdir(parents=True)
    source = arguments.output / "source.onnx"
    model = arguments.output / "mobilenetv3-small-075-dynamic.onnx"
    download_source(source)
    identity = derive_dynamic_mobilenet(source, model)
    artifacts = [artifact(arguments.output, source), artifact(arguments.output, model)]
    cases = []
    for name, (batch, height, width) in CASES.items():
        case_root = arguments.output / "inputs" / name
        case_root.mkdir(parents=True)
        input_value = deterministic_image_input(batch, height, width)
        input_path = case_root / "x.npy"
        np.save(input_path, input_value, allow_pickle=False)
        outputs = run_onnx_reference(model, {"x": input_value})
        expected_path = case_root / "expected.npy"
        np.save(expected_path, outputs[0].values, allow_pickle=False)
        artifacts.extend(
            [artifact(arguments.output, input_path), artifact(arguments.output, expected_path)]
        )
        cases.append(
            {
                "id": name,
                "shape": [batch, 3, height, width],
                "output_name": outputs[0].name,
                "output_shape": outputs[0].shape,
            }
        )
    lock = {
        "schema_version": "upgradeguard.dev/mobilenet-corpus/v1",
        "source_sha256": identity.source_sha256,
        "derived_sha256": identity.derived_sha256,
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
    }


if __name__ == "__main__":
    main()
