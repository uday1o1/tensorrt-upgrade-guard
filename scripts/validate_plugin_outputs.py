"""Validate worker plugin outputs against frozen project references."""

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
    for precision in ("fp32", "fp16"):
        atol = 1e-5 if precision == "fp32" else 5e-3
        rtol = 1e-4 if precision == "fp32" else 5e-3
        for case_root in sorted((arguments.corpus / precision).iterdir()):
            if not case_root.is_dir():
                continue
            expected = np.load(case_root / "expected.npy", allow_pickle=False)
            expected_input_hashes = {
                name: sha256_file(case_root / f"{name}.npy") for name in ("x", "residual", "gamma")
            }
            case_evidence = {"precision": precision, "case": case_root.name, "workers": {}}
            for environment in ("baseline", "candidate"):
                result_root = arguments.runs / environment / precision / case_root.name
                identity = f"{environment}/{precision}/{case_root.name}"
                validation = validate_repetitions(
                    result_path=result_root / "correctness.json",
                    runs_root=arguments.runs,
                    expected_output_name="output",
                    expected=expected,
                    atol=atol,
                    rtol=rtol,
                    expected_engine_sha256=sha256_file(
                        arguments.runs / environment / precision / "engine.plan"
                    ),
                    expected_input_hashes=expected_input_hashes,
                )
                case_evidence["workers"][environment] = {"passed": True, **validation}
                if not validation["tolerance_stable"]:
                    raise RuntimeError(f"plugin determinism gate failed for {identity}")
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
