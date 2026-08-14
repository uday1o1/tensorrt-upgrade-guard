"""Validate real GPU seed controls and the paired slowdown decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from upgrade_guard.classify import classify_observed_facts
from upgrade_guard.compare.performance import AcceptedPair, GateOutcome, paired_ratio_gate
from upgrade_guard.errors import FailureCode

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
    "baseline_then_candidate",
    "candidate_then_baseline",
    "baseline_then_candidate",
    "candidate_then_baseline",
)
SEED_CONTRACTS = {
    "G1": ("unsupported_custom_domain_onnx_node", FailureCode.ONNX_PARSE_FAILED),
    "G2": ("plugin_omits_residual_at_hidden_259", FailureCode.NUMERICAL_REGRESSION),
    "G3": ("zero_epsilon_zero_input", FailureCode.NONFINITE_OUTPUT),
    "G4": ("vectorized_tail_out_of_bounds", FailureCode.SANITIZER_FAILURE),
    "G6": ("creator_omits_serialized_epsilon", FailureCode.NUMERICAL_REGRESSION),
    "G7": ("input_exceeds_optimization_profile", FailureCode.PROFILE_REJECTED),
}


def validate_records(
    records: list[dict[str, Any]], serialization_record: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate controls, the locked order schedule, and the paired seed gate."""

    if len(records) != len(ORDER_SCHEDULE):
        raise RuntimeError("seeded slowdown requires the full locked GPU sample schedule")
    pairs: list[AcceptedPair] = []
    observed_orders: list[str] = []
    classifications: dict[str, FailureCode] = {}
    for pair_index, record in enumerate(records):
        for seed, expected in (
            ("G2", FailureCode.NUMERICAL_REGRESSION),
            ("G3", FailureCode.NONFINITE_OUTPUT),
        ):
            seed_record = record.get(seed)
            if not isinstance(seed_record, dict):
                raise RuntimeError(f"{seed} evidence is not an object")
            classification = classify_seed_record(seed, seed_record)
            if classification is not expected:
                raise RuntimeError(f"{seed} received {classification} instead of {expected}")
            if classifications.setdefault(seed, classification) is not classification:
                raise RuntimeError(f"{seed} classification changed between samples")
        performance = record["G5"]
        if (
            performance.get("mechanism") != "controlled_device_delay"
            or performance.get("control") != "passed"
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
    if observed_orders.count("baseline_then_candidate") != len(ORDER_SCHEDULE) // 2:
        raise RuntimeError("G5 order schedule is not balanced")
    gate = paired_ratio_gate(
        tuple(pairs),
        allowance=0.03,
        seed=ORDER_SEED,
        replicates=5000,
        minimum_pairs=20,
    )
    g5_classification = classify_observed_facts(
        {"performance_valid": gate.outcome is not GateOutcome.REGRESSION}
    )
    if g5_classification is not FailureCode.PERFORMANCE_REGRESSION:
        raise RuntimeError("G5 did not receive PERFORMANCE_REGRESSION from the classifier")
    payload = {
        "schema_version": "upgradeguard.dev/seeded-gpu-validation/v1",
        "status": "passed" if gate.outcome is GateOutcome.REGRESSION else "failed",
        "G2": {"classification": classifications["G2"].value, "confirmed": True},
        "G3": {"classification": classifications["G3"].value, "confirmed": True},
        "G5": {
            "classification": g5_classification.value,
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
    if serialization_record is not None:
        payload["G6"] = validate_serialization_record(serialization_record)
    return payload


def classify_seed_record(seed: str, record: dict[str, Any]) -> FailureCode:
    """Validate one seed's mechanism/control and classify its observed facts."""

    contract = SEED_CONTRACTS.get(seed)
    if contract is None:
        raise RuntimeError(f"seed has no typed classification contract: {seed}")
    mechanism, expected = contract
    if (
        record.get("mechanism") != mechanism
        or record.get("detected") is not True
        or record.get("control") != "passed"
    ):
        raise RuntimeError(f"seed mechanism or nearby control differs: {mechanism}")
    facts = record.get("observed_facts")
    if not isinstance(facts, dict):
        raise RuntimeError(f"seed has no typed observed facts: {mechanism}")
    try:
        classification = classify_observed_facts(facts)
    except ValueError as error:
        raise RuntimeError(f"seed observed facts are invalid: {mechanism}") from error
    if classification is not expected:
        raise RuntimeError(f"{seed} received {classification} instead of {expected}")
    return classification


def validate_serialization_record(record: dict[str, Any]) -> dict[str, object]:
    """Classify the authentic creator-restoration fault from field evidence."""

    classification = classify_seed_record("G6", record)
    expected = record.get("expected_epsilon")
    control = record.get("control_epsilon")
    fault = record.get("fault_epsilon")
    if (
        not isinstance(expected, int | float)
        or not isinstance(control, int | float)
        or not isinstance(fault, int | float)
    ):
        raise RuntimeError("G6 epsilon field evidence is invalid")
    expected_value = float(expected)
    control_value = float(control)
    fault_value = float(fault)
    if abs(control_value - expected_value) >= 1e-8 or abs(fault_value - expected_value) <= 1e-3:
        raise RuntimeError("G6 did not isolate creator epsilon restoration")
    if (
        record.get("serialized_epsilon_present") is not True
        or record.get("control_restore_epsilon_present") is not True
        or record.get("fault_restore_epsilon_present") is not False
    ):
        raise RuntimeError("G6 did not retain serialized/restored field inventory")
    if classification is not FailureCode.NUMERICAL_REGRESSION:
        raise RuntimeError("G6 did not receive NUMERICAL_REGRESSION from the classifier")
    return {
        "classification": classification.value,
        "confirmed": True,
        "expected_epsilon": expected_value,
        "control_epsilon": control_value,
        "fault_epsilon": fault_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--serialization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [json.loads(line) for line in arguments.samples.read_text().splitlines() if line]
    serialization = (
        json.loads(arguments.serialization.read_text(encoding="utf-8"))
        if arguments.serialization is not None
        else None
    )
    payload = validate_records(records, serialization)
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
