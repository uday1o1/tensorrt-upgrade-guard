"""Validate MobileNet TensorRT outputs and classification semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.worker.evidence import validate_repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = []
    for case_root in sorted((arguments.corpus / "inputs").iterdir()):
        expected = np.load(case_root / "expected.npy", allow_pickle=False)
        expected_top5 = np.argsort(expected, axis=-1)[:, -5:]
        expected_input_hashes = {"x": sha256_file(case_root / "x.npy")}
        case = {"case": case_root.name, "workers": {}}
        for environment in ("baseline", "candidate"):
            result_root = arguments.runs / environment / case_root.name
            validation = validate_repetitions(
                result_path=result_root / "correctness.json",
                runs_root=arguments.runs,
                expected_output_name="400",
                expected=expected,
                atol=1e-5,
                rtol=1e-4,
                expected_engine_sha256=sha256_file(arguments.runs / environment / "engine.plan"),
                expected_input_hashes=expected_input_hashes,
            )
            top1 = True
            top5 = True
            for repetition in range(20):
                observed = np.load(
                    result_root / "outputs" / f"400.repetition-{repetition:02d}.npy",
                    allow_pickle=False,
                )
                top1 = top1 and bool(
                    np.array_equal(np.argmax(observed, axis=-1), np.argmax(expected, axis=-1))
                )
                top5 = top5 and bool(
                    np.all(
                        [
                            set(observed_row) == set(expected_row)
                            for observed_row, expected_row in zip(
                                np.argsort(observed, axis=-1)[:, -5:],
                                expected_top5,
                                strict=True,
                            )
                        ]
                    )
                )
            case["workers"][environment] = {
                "numerical_passed": True,
                "top1_agreement": top1,
                "top5_agreement": top5,
                **validation,
            }
            if not top1 or not top5:
                raise RuntimeError(f"MobileNet gate failed for {environment}/{case_root.name}")
        evidence.append(case)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/mobilenet-validation/v1",
                "status": "passed",
                "cases": evidence,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
