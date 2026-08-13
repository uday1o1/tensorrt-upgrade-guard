"""Validate MobileNet TensorRT outputs and classification semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
        case = {"case": case_root.name, "workers": {}}
        for environment in ("baseline", "candidate"):
            observed = np.load(
                arguments.runs / environment / case_root.name / "outputs" / "400.repetition-00.npy",
                allow_pickle=False,
            )
            if observed.shape != expected.shape or observed.dtype != expected.dtype:
                raise RuntimeError("MobileNet output schema changed")
            absolute = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
            numerical = bool(np.all(absolute <= 1e-5 + 1e-4 * np.abs(expected.astype(np.float64))))
            top1 = bool(np.array_equal(np.argmax(observed, axis=-1), np.argmax(expected, axis=-1)))
            top5 = bool(
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
                "numerical_passed": numerical,
                "top1_agreement": top1,
                "top5_agreement": top5,
                "maximum_absolute_error": float(np.max(absolute)),
            }
            if not numerical or not top1 or not top5:
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
