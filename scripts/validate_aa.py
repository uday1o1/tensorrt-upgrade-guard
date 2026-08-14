"""Validate a real same-environment A/A timing pilot without updating primary data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upgrade_guard.compare.performance import (
    AcceptedPair,
    GateOutcome,
    coefficient_of_variation,
    paired_ratio_gate,
)
from upgrade_guard.worker.trtexec import load_exported_times


def validate_aa(pairs_directory: Path) -> dict[str, object]:
    """Return fail-closed A/A evidence for one complete pilot directory."""

    pairs = []
    raw = []
    for directory in sorted(pairs_directory.glob("pair-*")):
        first = load_exported_times(directory / "a.json")
        second = load_exported_times(directory / "b.json")
        validity = json.loads((directory / "validity.json").read_text(encoding="utf-8"))
        if validity.get("status") != "passed":
            raise RuntimeError(f"A/A pair retained invalid hardware evidence: {directory}")
        pairs.append(AcceptedPair(first.median_milliseconds, second.median_milliseconds))
        raw.append(
            {
                "pair": directory.name,
                "a_median_milliseconds": first.median_milliseconds,
                "b_median_milliseconds": second.median_milliseconds,
                "hardware_validity": validity,
            }
        )
    accepted = tuple(pairs)
    gate = paired_ratio_gate(
        accepted,
        allowance=0.05,
        seed=20260813,
        replicates=5000,
        minimum_pairs=20,
    )
    medians = tuple(
        value
        for pair in accepted
        for value in (pair.baseline_milliseconds, pair.candidate_milliseconds)
    )
    passed = gate.outcome is GateOutcome.PASSED and len(accepted) >= 20
    return {
        "schema_version": "upgradeguard.dev/aa-pilot/v1",
        "status": "passed" if passed else "failed",
        "profiled": False,
        "accepted_pairs": len(accepted),
        "coefficient_of_variation": coefficient_of_variation(medians),
        "empirical_minimum_detectable_effect": max(0.0, gate.one_sided_upper - 1.0),
        "false_positive": gate.outcome is GateOutcome.REGRESSION,
        "gate": {
            "point": gate.point,
            "one_sided_lower": gate.one_sided_lower,
            "one_sided_upper": gate.one_sided_upper,
            "outcome": gate.outcome.value,
        },
        "pairs": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = validate_aa(arguments.pairs)
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
