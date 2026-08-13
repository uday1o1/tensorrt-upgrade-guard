"""Validate the engine-reported 64 MiB plugin workspace seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upgrade_guard.compare.memory import device_memory_gate
from upgrade_guard.compare.performance import GateOutcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--seeded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    def observations(root: Path) -> tuple[int, ...]:
        return tuple(
            int(json.loads(path.read_text())["engine"]["device_memory_bytes"])
            for path in sorted(root.glob("build-*.json"))
        )

    control = observations(arguments.control)
    seeded = observations(arguments.seeded)
    gate = device_memory_gate(control, seeded)
    passed = gate.outcome is GateOutcome.REGRESSION
    arguments.output.write_text(
        json.dumps(
            {
                "schema_version": "upgradeguard.dev/memory-seed/v1",
                "status": "passed" if passed else "failed",
                "expected": "MEMORY_REGRESSION",
                "control_device_memory_bytes": control,
                "seeded_device_memory_bytes": seeded,
                "allowance_bytes": gate.allowance_bytes,
                "outcome": gate.outcome.value,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
