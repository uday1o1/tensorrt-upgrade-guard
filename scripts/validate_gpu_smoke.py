"""Validate the bounded standard and plugin GPU smoke paths against CPU references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.corpus.reference import run_onnx_reference
from upgrade_guard.worker.evidence import validate_repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-corpus", type=Path, required=True)
    parser.add_argument("--plugin-corpus", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    standard_engine = arguments.runs / "standard" / "engine.plan"
    standard_model = arguments.core_corpus / "models" / "tiny-transformer-fp32.onnx"
    standard_cases = []
    for case_name in ("b1_s8", "b1_s128"):
        case_root = arguments.core_corpus / "inputs" / "tiny-transformer-fp32" / case_name
        inputs = {
            name: np.load(case_root / f"{name}.npy", allow_pickle=False)
            for name in ("tokens", "mask")
        }
        reference = run_onnx_reference(standard_model, inputs)
        if len(reference) != 1:
            raise RuntimeError("standard smoke model output inventory changed")
        evidence = validate_repetitions(
            result_path=arguments.runs / "standard" / case_name / "correctness.json",
            runs_root=arguments.runs,
            expected_output_name=reference[0].name,
            expected=reference[0].values,
            atol=1e-5,
            rtol=1e-4,
            expected_engine_sha256=sha256_file(standard_engine),
            expected_input_hashes={name: sha256_file(case_root / f"{name}.npy") for name in inputs},
        )
        standard_cases.append({"case": case_name, **evidence})

    plugin_case = arguments.plugin_corpus / "fp32" / "tail-random-h259"
    plugin_evidence = validate_repetitions(
        result_path=arguments.runs / "plugin" / "correctness.json",
        runs_root=arguments.runs,
        expected_output_name="output",
        expected=np.load(plugin_case / "expected.npy", allow_pickle=False),
        atol=1e-5,
        rtol=1e-4,
        expected_engine_sha256=sha256_file(arguments.runs / "plugin" / "engine.plan"),
        expected_input_hashes={
            name: sha256_file(plugin_case / f"{name}.npy") for name in ("x", "residual", "gamma")
        },
    )
    payload = {
        "schema_version": "upgradeguard.dev/gpu-smoke/v1",
        "status": "passed",
        "standard_cases": standard_cases,
        "plugin_tail_case": plugin_evidence,
        "same_environment_reload": True,
    }
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
