"""Polygraphy check command for the exact G7 profile-rejection predicate."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.errors import FailureCode

REPRODUCED = 86
INFRASTRUCTURE_INVALID = 87


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--control-tokens", type=Path, required=True)
    parser.add_argument("--control-mask", type=Path, required=True)
    parser.add_argument("--failure-tokens", type=Path, required=True)
    parser.add_argument("--failure-mask", type=Path, required=True)
    parser.add_argument("--workspace-bytes", type=int, required=True)
    parser.add_argument("--optimization-level", type=int, required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--observations", type=Path, required=True)
    arguments = parser.parse_args()
    model = Path("polygraphy_debug.onnx").resolve()
    record: dict[str, object] = {
        "schema_version": "upgradeguard.dev/polygraphy-profile-check/v1",
        "failure_code": FailureCode.PROFILE_REJECTED.value,
        "predicate_signature_sha256": arguments.signature,
    }
    try:
        if not model.is_file() or model.is_symlink():
            raise RuntimeError("Polygraphy did not publish its current ONNX artifact")
        record["model_sha256"] = sha256_file(model)
        with tempfile.TemporaryDirectory(prefix="ug-profile-check-", dir="/output") as name:
            root = Path(name)
            build = _run(
                (
                    "python3",
                    "-m",
                    "upgrade_guard.worker.build_engine",
                    "--model",
                    str(model),
                    "--profile",
                    str(arguments.profile),
                    "--engine",
                    str(root / "engine.plan"),
                    "--inspector",
                    str(root / "inspector.json"),
                    "--timing-cache",
                    str(root / "timing.cache"),
                    "--result",
                    str(root / "build.json"),
                    "--workspace-bytes",
                    str(arguments.workspace_bytes),
                    "--optimization-level",
                    str(arguments.optimization_level),
                )
            )
            if build.returncode != 0:
                record.update(outcome="not_reproduced", detail="candidate graph did not build")
                _append(arguments.observations, record)
                return 0
            control = _run(
                _correctness(
                    root,
                    "control",
                    arguments.control_tokens,
                    arguments.control_mask,
                )
            )
            if control.returncode != 0:
                record.update(outcome="not_reproduced", detail="candidate control did not pass")
                _append(arguments.observations, record)
                return 0
            failure = _run(
                _correctness(
                    root,
                    "failure",
                    arguments.failure_tokens,
                    arguments.failure_mask,
                )
            )
            failure_result = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            reproduced = (
                failure.returncode == 1
                and failure_result.get("status") == "failed"
                and failure_result.get("failure_code") == FailureCode.PROFILE_REJECTED.value
                and "input shape was rejected" in str(failure_result.get("message", ""))
            )
            record.update(
                outcome="reproduced" if reproduced else "not_reproduced",
                build_sha256=sha256_file(root / "build.json"),
                control_sha256=sha256_file(root / "control.json"),
                failure_sha256=sha256_file(root / "failure.json"),
            )
            _append(arguments.observations, record)
            return REPRODUCED if reproduced else 0
    except Exception as error:
        record.update(
            outcome="infrastructure_invalid",
            detail=f"{type(error).__name__}: {error}"[:4096],
        )
        _append(arguments.observations, record)
        return INFRASTRUCTURE_INVALID


def _run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        shell=False,
    )


def _correctness(
    root: Path,
    kind: str,
    tokens: Path,
    mask: Path,
) -> tuple[str, ...]:
    return (
        "python3",
        "-m",
        "upgrade_guard.worker.run_correctness",
        "--engine",
        str(root / "engine.plan"),
        "--input",
        f"tokens={tokens}",
        "--input",
        f"mask={mask}",
        "--output",
        str(root / f"{kind}-outputs"),
        "--result",
        str(root / f"{kind}.json"),
        "--repetitions",
        "2",
    )


def _append(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
