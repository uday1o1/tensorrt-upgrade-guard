"""Capture and compare GPU benchmark validity observations for shell-driven pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upgrade_guard.containers.commands import CommandRunner
from upgrade_guard.contracts.qualification import QualificationSpec
from upgrade_guard.qualification import _block_variation_reasons, _observe_validity


def _specification(path: Path) -> QualificationSpec:
    import yaml

    return QualificationSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def capture(specification_path: Path, gpu_uuid: str, output: Path, *, loaded: bool) -> bool:
    specification = _specification(specification_path)
    observed, reasons = _observe_validity(
        CommandRunner(),
        gpu_uuid,
        specification,
        require_idle=not loaded,
    )
    payload = {
        "schema_version": "upgradeguard.dev/hardware-observation/v1",
        "status": "passed" if not reasons else "rejected",
        "phase": "loaded" if loaded else "idle",
        "observed": observed,
        "rejection_reasons": list(reasons),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return not reasons


def transition(
    specification_path: Path,
    before_path: Path,
    after_path: Path,
    output: Path,
) -> bool:
    specification = _specification(specification_path)
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    reasons = [
        *before.get("rejection_reasons", []),
        *after.get("rejection_reasons", []),
        *_block_variation_reasons(before["observed"], after["observed"], specification),
    ]
    payload = {
        "schema_version": "upgradeguard.dev/hardware-transition/v1",
        "status": "passed" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "before": before["observed"],
        "after": after["observed"],
    }
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return not reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--specification", type=Path, required=True)
    capture_parser.add_argument("--gpu", required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--loaded", action="store_true")
    transition_parser = subparsers.add_parser("transition")
    transition_parser.add_argument("--specification", type=Path, required=True)
    transition_parser.add_argument("--before", type=Path, required=True)
    transition_parser.add_argument("--after", type=Path, required=True)
    transition_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "capture":
        return (
            0
            if capture(
                arguments.specification,
                arguments.gpu,
                arguments.output,
                loaded=arguments.loaded,
            )
            else 1
        )
    return (
        0
        if transition(
            arguments.specification,
            arguments.before,
            arguments.after,
            arguments.output,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
