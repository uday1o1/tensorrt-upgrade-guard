"""Validate worker plugin outputs against frozen project references."""

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
    for precision in ("fp32", "fp16"):
        atol = 1e-5 if precision == "fp32" else 5e-3
        rtol = 1e-4 if precision == "fp32" else 5e-3
        for case_root in sorted((arguments.corpus / precision).iterdir()):
            if not case_root.is_dir():
                continue
            expected = np.load(case_root / "expected.npy", allow_pickle=False)
            case_evidence = {"precision": precision, "case": case_root.name, "workers": {}}
            for environment in ("baseline", "candidate"):
                result_path = (
                    arguments.runs
                    / environment
                    / precision
                    / case_root.name
                    / "outputs"
                    / "output.repetition-00.npy"
                )
                observed = np.load(result_path, allow_pickle=False)
                identity = f"{environment}/{precision}/{case_root.name}"
                if observed.shape != expected.shape or observed.dtype != expected.dtype:
                    raise RuntimeError(f"plugin output schema changed for {identity}")
                absolute = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
                passed = bool(np.all(absolute <= atol + rtol * np.abs(expected.astype(np.float64))))
                case_evidence["workers"][environment] = {
                    "passed": passed,
                    "maximum_absolute_error": float(np.max(absolute)),
                }
                if not passed:
                    raise RuntimeError(f"plugin numerical gate failed for {identity}")
            evidence.append(case_evidence)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/plugin-validation/v1",
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
