"""Validate real GPU seed controls and the paired slowdown decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upgrade_guard.compare.performance import AcceptedPair, GateOutcome, paired_ratio_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [json.loads(line) for line in arguments.samples.read_text().splitlines() if line]
    if len(records) != 20:
        raise RuntimeError("seeded slowdown requires exactly 20 GPU samples")
    for record in records:
        for seed in ("G2", "G3", "G5"):
            if not record[seed]["detected"] or record[seed]["control"] != "passed":
                raise RuntimeError(f"{seed} or its nearby control did not behave as expected")
    pairs = tuple(AcceptedPair(1.0, float(record["G5"]["ratio"])) for record in records)
    gate = paired_ratio_gate(
        pairs,
        allowance=0.10,
        seed=20260813,
        replicates=5000,
        minimum_pairs=20,
    )
    payload = {
        "schema_version": "upgradeguard.dev/seeded-gpu-validation/v1",
        "status": "passed" if gate.outcome is GateOutcome.REGRESSION else "failed",
        "G2": {"expected": "NUMERICAL_REGRESSION", "confirmed": True},
        "G3": {"expected": "NONFINITE_OUTPUT", "confirmed": True},
        "G5": {
            "expected": "PERFORMANCE_REGRESSION",
            "confirmed": gate.outcome is GateOutcome.REGRESSION,
            "accepted_pairs": gate.accepted_pairs,
            "point": gate.point,
            "one_sided_lower": gate.one_sided_lower,
            "one_sided_upper": gate.one_sided_upper,
        },
    }
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if gate.outcome is not GateOutcome.REGRESSION:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
