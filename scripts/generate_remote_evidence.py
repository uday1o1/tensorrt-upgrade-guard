"""Validate every remote gate and publish one hash-addressed evidence index."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.environment import MatrixLock


def _json(path: Path, expected_status: str = "passed") -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != expected_status:
        raise RuntimeError(f"required evidence did not pass: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    state = arguments.state.resolve()
    required_json = {
        "core_qualification": state / "core-run" / "qualification-summary.json",
        "plugin_validation": state / "plugin-runs" / "validation.json",
        "mobilenet_validation": state / "mobilenet-runs" / "validation.json",
        "aa_pilot": state / "aa" / "validation.json",
        "cuda_benchmark": state / "plugin" / "candidate" / "plugin-benchmark.json",
        "gpu_faults": state / "faults" / "validation.json",
        "memory_seed": state / "memory-seed" / "validation.json",
    }
    values = {name: _json(path) for name, path in required_json.items()}
    sanitizer = json.loads((state / "plugin" / "candidate" / "sanitizer-seed.json").read_text())
    if sanitizer.get("expected") != "SANITIZER_FAILURE" or sanitizer.get("control") != "passed":
        raise RuntimeError("sanitizer seed or clean control evidence is invalid")
    for name in ("G1", "G6", "G7"):
        value = json.loads((state / "faults" / f"{name}.json").read_text())
        if not value.get("detected") or value.get("control") != "passed":
            raise RuntimeError(f"{name} or its control did not pass")
    matrix_path = state.parent / "matrix.lock.json"
    matrix = MatrixLock.model_validate_json(matrix_path.read_text(encoding="utf-8"))
    if matrix.computed_sha256() != matrix.lock_sha256:
        raise RuntimeError("remote matrix lock self-hash differs")
    required_files = [
        state / "source.commit",
        state / "gpu-preflight.csv",
        state / "plugin" / "candidate" / "residual-rmsnorm-timeline.nsys-rep",
        state / "plugin" / "candidate" / "residual-rmsnorm-timeline.sqlite",
        state / "plugin" / "candidate" / "nsys-kernel-summary.csv",
        state / "plugin" / "candidate" / "residual-rmsnorm-kernel.ncu-rep",
        state / "plugin" / "candidate" / "ncu-kernel-summary.csv",
        state / "sbom" / "baseline.spdx.json",
        state / "sbom" / "candidate.spdx.json",
        state / "plugin" / "baseline" / "build" / "compile_commands.json",
        state / "plugin" / "candidate" / "build" / "compile_commands.json",
        state / "plugin" / "candidate" / "build" / "libupgrade_guard_residual_rmsnorm.so",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required remote artifact is absent or empty: {path}")
    log_files = sorted((state / "logs").glob("*.log"))
    if not log_files:
        raise RuntimeError("remote qualification did not retain step logs")
    artifacts = {
        path.relative_to(state).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in [*required_json.values(), *required_files, *log_files]
    }
    payload = {
        "schema_version": "upgradeguard.dev/remote-evidence/v1",
        "status": "passed",
        "generated_at": datetime.now(UTC).isoformat(),
        "gpu_uuid": matrix.gpu_uuid,
        "matrix_lock_sha256": matrix.lock_sha256,
        "source_git_commit": state.joinpath("source.commit").read_text().strip(),
        "environment_images": {
            environment.id: environment.worker_image.canonical_reference
            for environment in matrix.environments
        },
        "gate_status": {name: value["status"] for name, value in values.items()},
        "artifacts": artifacts,
        "claim_scope": (
            "Results apply to the complete locked stacks and selected GPU. "
            "They do not isolate TensorRT as the sole cause of a difference."
        ),
    }
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
