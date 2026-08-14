"""Validate real GPU seed controls and the paired slowdown decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from upgrade_guard.compare.performance import AcceptedPair, GateOutcome, paired_ratio_gate

ORDER_SEED = 20260813
ORDER_SCHEDULE = (
    "baseline_then_candidate",
    "candidate_then_baseline",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "baseline_then_candidate",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "candidate_then_baseline",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "baseline_then_candidate",
    "candidate_then_baseline",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "baseline_then_candidate",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "candidate_then_baseline",
)


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate controls, the locked order schedule, and the paired seed gate."""

    if len(records) != 20:
        raise RuntimeError("seeded slowdown requires exactly 20 GPU samples")
    pairs: list[AcceptedPair] = []
    observed_orders: list[str] = []
    for pair_index, record in enumerate(records):
        for seed in ("G2", "G3"):
            if not record[seed]["detected"] or record[seed]["control"] != "passed":
                raise RuntimeError(f"{seed} or its nearby control did not behave as expected")
        performance = record["G5"]
        if (
            performance.get("control") != "passed"
            or performance.get("calibrated") is not True
            or performance.get("pair_index") != pair_index
            or performance.get("order_seed") != ORDER_SEED
            or float(performance.get("target_ratio", math.nan)) != 1.10
        ):
            raise RuntimeError("G5 calibration, control, pair index, or seed differs")
        order = performance.get("order")
        if order != ORDER_SCHEDULE[pair_index]:
            raise RuntimeError("G5 did not use the locked balanced randomized order schedule")
        baseline = float(performance["baseline_ms"])
        candidate = float(performance["candidate_ms"])
        ratio = float(performance["ratio"])
        if not all(
            math.isfinite(value) and value > 0.0 for value in (baseline, candidate, ratio)
        ) or not math.isclose(candidate / baseline, ratio, rel_tol=2e-5):
            raise RuntimeError("G5 retained invalid paired timing evidence")
        observed_orders.append(order)
        pairs.append(AcceptedPair(baseline, candidate))
    if observed_orders.count("baseline_then_candidate") != 10:
        raise RuntimeError("G5 order schedule is not balanced")
    gate = paired_ratio_gate(
        tuple(pairs),
        allowance=0.03,
        seed=ORDER_SEED,
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
            "target_ratio": 1.10,
            "allowance": 0.03,
            "order_seed": ORDER_SEED,
            "orders": observed_orders,
            "confirmed": gate.outcome is GateOutcome.REGRESSION,
            "accepted_pairs": gate.accepted_pairs,
            "point": gate.point,
            "one_sided_lower": gate.one_sided_lower,
            "one_sided_upper": gate.one_sided_upper,
        },
    }
    if gate.outcome is not GateOutcome.REGRESSION:
        raise RuntimeError("G5 paired interval did not confirm the seeded slowdown")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [json.loads(line) for line in arguments.samples.read_text().splitlines() if line]
    payload = validate_records(records)
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
