"""Hash-addressed resume state for the CUDA qualification runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

STEP_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "preflight": ("source.commit", "gpu.uuid", "gpu-preflight.csv"),
    "cpu-verify": (),
    "worker-images": ("worker-images.json",),
    "matrix-lock": ("matrix.lock.json", "full.yaml"),
    "corpus-materialization": (
        "@project/.upgrade-guard/corpora/v1-core/corpus.lock.json",
        "corpora/plugin/plugin-corpus.lock.json",
        "corpora/mobilenet/mobilenet-corpus.lock.json",
    ),
    "aa-pilot": ("aa/validation.json", "aa/engine.plan", "aa/build.json"),
    "core-qualification": ("core-run/qualification-summary.json",),
    "plugin-compile-test": (
        "plugin/baseline/build/libupgrade_guard_residual_rmsnorm.so",
        "plugin/baseline/build/upgrade_guard_kernel_tests",
        "plugin/baseline/build/compile_commands.json",
        "plugin/candidate/build/libupgrade_guard_residual_rmsnorm.so",
        "plugin/candidate/build/upgrade_guard_kernel_tests",
        "plugin/candidate/build/upgrade_guard_gpu_faults",
        "plugin/candidate/build/compile_commands.json",
    ),
    "gpu-smoke": (
        "smoke/plugin/build.json",
        "smoke/plugin/correctness.json",
        "smoke/standard/build.json",
        "smoke/standard/b1_s8/correctness.json",
        "smoke/standard/b1_s128/correctness.json",
        "smoke/validation.json",
    ),
    "plugin-benchmark": (
        "plugin/candidate/plugin-benchmark.json",
        "plugin/candidate/plugin-benchmark-validity.json",
    ),
    "plugin-matrix": ("plugin-runs/validation.json",),
    "mobilenet-matrix": ("mobilenet-runs/validation.json",),
    "fault-inputs": ("faults/inputs.json",),
    "gpu-faults": (
        "faults/validation.json",
        "faults/G1.json",
        "faults/G6.json",
        "faults/G7.json",
    ),
    "reduction-replay": ("reductions/validation.json",),
    "memory-seed": ("memory-seed/validation.json",),
    "sanitizers": ("plugin/candidate/sanitizer-seed.json",),
    "profiles": (
        "plugin/candidate/residual-rmsnorm-timeline.nsys-rep",
        "plugin/candidate/residual-rmsnorm-timeline.sqlite",
        "plugin/candidate/nsys-kernel-summary.csv",
        "plugin/candidate/residual-rmsnorm-kernel.ncu-rep",
        "plugin/candidate/ncu-kernel-summary.csv",
    ),
    "sboms": (
        "sbom/baseline.spdx.json",
        "sbom/candidate.spdx.json",
        "sbom/host.spdx.json",
    ),
    "dependency-audit": (
        "supply-chain/pip-audit.json",
        "supply-chain/worker-pip-audit.json",
    ),
    "final-evidence": ("evidence.json", "results.json", "report-model.json", "report.md"),
}

PASSING_JSON: dict[str, tuple[str, ...]] = {
    "aa-pilot": ("aa/validation.json",),
    "core-qualification": ("core-run/qualification-summary.json",),
    "gpu-smoke": (
        "smoke/plugin/build.json",
        "smoke/plugin/correctness.json",
        "smoke/standard/build.json",
        "smoke/standard/b1_s8/correctness.json",
        "smoke/standard/b1_s128/correctness.json",
        "smoke/validation.json",
    ),
    "plugin-benchmark": (
        "plugin/candidate/plugin-benchmark.json",
        "plugin/candidate/plugin-benchmark-validity.json",
    ),
    "plugin-matrix": ("plugin-runs/validation.json",),
    "mobilenet-matrix": ("mobilenet-runs/validation.json",),
    "gpu-faults": ("faults/validation.json",),
    "reduction-replay": ("reductions/validation.json",),
    "memory-seed": ("memory-seed/validation.json",),
    "final-evidence": ("evidence.json", "results.json"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("record", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--state", type=Path, required=True)
        child.add_argument("--project", type=Path, required=True)
        child.add_argument("--step", choices=tuple(STEP_ARTIFACTS), required=True)
        child.add_argument("--source", required=True)
        child.add_argument("--gpu", required=True)
        child.add_argument("--mode", choices=("full", "smoke", "sanitizer"), required=True)
    corpus = subparsers.add_parser("verify-corpus")
    corpus.add_argument("path", type=Path)
    workers = subparsers.add_parser("capture-workers")
    workers.add_argument("--output", type=Path, required=True)
    workers.add_argument("images", nargs=2)
    worker_lock = subparsers.add_parser("verify-worker-lock")
    worker_lock.add_argument("--workers", type=Path, required=True)
    worker_lock.add_argument("--matrix", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "verify-corpus":
        _verify_corpus(arguments.path.resolve(strict=True))
        return 0
    if arguments.command == "capture-workers":
        from upgrade_guard.matrix.digest import RegistryClient

        resolved = [
            RegistryClient().resolve_linux_amd64(image).image.model_dump(mode="json")
            for image in arguments.images
        ]
        _write_atomic(
            arguments.output,
            {
                "schema_version": "upgradeguard.dev/worker-images/v1",
                "images": resolved,
            },
        )
        return 0
    if arguments.command == "verify-worker-lock":
        workers_value = json.loads(arguments.workers.read_text(encoding="utf-8"))
        matrix_value = json.loads(arguments.matrix.read_text(encoding="utf-8"))
        worker_digests = {value["manifest_digest"] for value in workers_value.get("images", [])}
        matrix_digests = {
            value["worker_image"]["manifest_digest"]
            for value in matrix_value.get("environments", [])
        }
        return 0 if len(worker_digests) == 2 and worker_digests == matrix_digests else 1
    state = arguments.state.resolve(strict=True)
    project = arguments.project.resolve(strict=True)
    marker = state / "done" / f"{arguments.step}.json"
    if arguments.command == "record":
        payload = _payload(
            state,
            project,
            arguments.step,
            arguments.source,
            arguments.gpu,
            arguments.mode,
        )
        _write_atomic(marker, payload)
        return 0
    try:
        observed = json.loads(marker.read_text(encoding="utf-8"))
        expected = _payload(
            state,
            project,
            arguments.step,
            arguments.source,
            arguments.gpu,
            arguments.mode,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return 1
    return 0 if observed == expected else 1


def _payload(
    state: Path,
    project: Path,
    step: str,
    source: str,
    gpu: str,
    mode: str,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, object]] = {}
    authored_artifacts = STEP_ARTIFACTS[step]
    if step == "corpus-materialization" and mode == "sanitizer":
        authored_artifacts = authored_artifacts[:2]
    for authored in authored_artifacts:
        path = _resolve_artifact(state, project, authored)
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"required step artifact is absent or empty: {path}")
        artifacts[authored] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    for relative in PASSING_JSON.get(step, ()):
        path = state / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("status") != "passed":
            raise ValueError(f"required JSON did not pass: {path}")
    if step == "aa-pilot":
        value = json.loads((state / "aa" / "validation.json").read_text(encoding="utf-8"))
        if value.get("false_positive") is not False or value.get("accepted_pairs") != 20:
            raise ValueError("A/A pilot did not retain exactly 20 passing pairs")
    return {
        "schema_version": "upgradeguard.dev/qualification-step/v1",
        "step": step,
        "source_git_commit": source,
        "gpu_uuid": gpu,
        "mode": mode,
        "artifacts": artifacts,
    }


def _resolve_artifact(state: Path, project: Path, authored: str) -> Path:
    if authored.startswith("@project/"):
        return project / authored.removeprefix("@project/")
    return state / authored


def _verify_corpus(root: Path) -> None:
    lock_names = (
        "corpus.lock.json",
        "plugin-corpus.lock.json",
        "mobilenet-corpus.lock.json",
    )
    locks = [root / name for name in lock_names if (root / name).is_file()]
    if len(locks) != 1:
        raise ValueError("corpus must contain exactly one supported lock")
    value = json.loads(locks[0].read_text(encoding="utf-8"))
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("corpus lock contains no artifacts")
    expected: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("corpus artifact is not an object")
        relative = artifact.get("path")
        size = artifact.get("bytes")
        digest = artifact.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("corpus artifact path is unsafe")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError("corpus artifact escaped its root")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"corpus artifact differs from its lock: {relative}")
        expected.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path not in locks
    }
    if observed != expected:
        raise ValueError("corpus inventory differs from its lock")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
