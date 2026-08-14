"""Hash-addressed resume state for the CUDA qualification runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

MARKER_SCHEMA = "upgradeguard.dev/qualification-step/v2"
MODES = ("full", "smoke", "sanitizer")
_RESERVED_STATE_ROOTS = frozenset({"done", "stale", "diagnostics"})

# Alternative public names normalize to one authority owner and one marker.
STEP_ALIASES: dict[str, str] = {
    "plugin-compile-test": "plugin-build",
    "reduction-replay": "reduction-validation",
}

# The values are active paths relative to one source-run state root. A trailing slash
# denotes a required nonempty directory; every other value denotes a required file.
STEP_OWNED_PATHS: dict[str, tuple[str, ...]] = {
    "preflight": ("source.commit", "gpu.uuid", "gpu-preflight.csv"),
    "cpu-verify": (),
    "gpu-runtime-preflight": ("gpu-runtime-preflight.json",),
    "registry-bootstrap": ("registry-identity.json",),
    "capacity-preflight": ("capacity/",),
    "worker-images": ("worker-images.json",),
    "matrix-lock": ("matrix.yaml", "full.yaml", "matrix.lock.json"),
    "corpus-materialization": ("corpora.json",),
    "plugin-build": ("plugin-build/",),
    "target-readiness": ("target-readiness/",),
    "profiler-preflight": ("profiler-preflight/",),
    "aa-pilot": ("aa/",),
    "core-qualification": ("core-run/",),
    "gpu-smoke": ("smoke/",),
    "plugin-benchmark": ("plugin-benchmark/",),
    "plugin-matrix": ("plugin-runs/",),
    "mobilenet-matrix": ("mobilenet-runs/",),
    "fault-inputs": ("fault-inputs/",),
    "gpu-faults": ("gpu-faults/",),
    "reduction-prepare": ("reductions/prepared/",),
    "replay-G2": ("reductions/G2/",),
    "replay-G7": ("reductions/G7/",),
    "reduction-validation": ("reductions/validation.json",),
    "memory-seed": ("memory-seed/",),
    "sanitizers": ("sanitizers/",),
    "profiles": ("profiles/",),
    "sboms": ("sbom/",),
    "dependency-audit": ("supply-chain/",),
    "final-evidence": ("results.json", "report-model.json", "report.md", "evidence.json"),
    "terminal-cleanup": ("cleanup.json",),
}

# Direct dependency edges only. Final evidence expands to every required pre-final
# marker for its mode through _dependencies_for().
STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "preflight": (),
    "cpu-verify": ("preflight",),
    "gpu-runtime-preflight": ("cpu-verify",),
    "registry-bootstrap": ("cpu-verify",),
    "capacity-preflight": ("preflight", "registry-bootstrap"),
    "dependency-audit": ("cpu-verify",),
    "corpus-materialization": (
        "gpu-runtime-preflight",
        "capacity-preflight",
    ),
    "worker-images": (
        "cpu-verify",
        "registry-bootstrap",
        "capacity-preflight",
    ),
    "matrix-lock": ("worker-images", "gpu-runtime-preflight"),
    "plugin-build": ("matrix-lock", "corpus-materialization"),
    "target-readiness": ("plugin-build", "matrix-lock", "corpus-materialization"),
    "profiler-preflight": ("plugin-build",),
    "aa-pilot": ("matrix-lock", "corpus-materialization"),
    "core-qualification": ("aa-pilot", "matrix-lock", "corpus-materialization"),
    "gpu-smoke": ("matrix-lock", "corpus-materialization", "plugin-build"),
    "plugin-benchmark": ("plugin-build", "profiler-preflight"),
    "plugin-matrix": ("matrix-lock", "corpus-materialization", "plugin-build"),
    "mobilenet-matrix": ("matrix-lock", "corpus-materialization"),
    "fault-inputs": ("corpus-materialization",),
    "gpu-faults": (
        "matrix-lock",
        "core-qualification",
        "plugin-matrix",
        "fault-inputs",
        "plugin-build",
        "corpus-materialization",
    ),
    "reduction-prepare": (
        "gpu-faults",
        "core-qualification",
        "plugin-matrix",
        "matrix-lock",
        "corpus-materialization",
    ),
    "replay-G2": ("reduction-prepare",),
    "replay-G7": ("reduction-prepare",),
    "reduction-validation": ("reduction-prepare", "replay-G2", "replay-G7"),
    "memory-seed": (
        "gpu-faults",
        "fault-inputs",
        "plugin-build",
        "corpus-materialization",
    ),
    "sanitizers": ("plugin-build", "matrix-lock"),
    "profiles": (
        "profiler-preflight",
        "plugin-build",
        "plugin-benchmark",
        "matrix-lock",
    ),
    "sboms": ("worker-images", "matrix-lock"),
    "final-evidence": (),
    "terminal-cleanup": ("final-evidence",),
}

MODE_STEPS: dict[str, tuple[str, ...]] = {
    "full": (
        "preflight",
        "cpu-verify",
        "gpu-runtime-preflight",
        "dependency-audit",
        "registry-bootstrap",
        "capacity-preflight",
        "corpus-materialization",
        "worker-images",
        "matrix-lock",
        "plugin-build",
        "profiler-preflight",
        "target-readiness",
        "sanitizers",
        "sboms",
        "aa-pilot",
        "core-qualification",
        "plugin-benchmark",
        "plugin-matrix",
        "mobilenet-matrix",
        "fault-inputs",
        "gpu-faults",
        "reduction-prepare",
        "replay-G2",
        "replay-G7",
        "reduction-validation",
        "memory-seed",
        "profiles",
        "final-evidence",
        "terminal-cleanup",
    ),
    "smoke": (
        "preflight",
        "cpu-verify",
        "gpu-runtime-preflight",
        "registry-bootstrap",
        "capacity-preflight",
        "corpus-materialization",
        "worker-images",
        "matrix-lock",
        "plugin-build",
        "gpu-smoke",
    ),
    "sanitizer": (
        "preflight",
        "cpu-verify",
        "gpu-runtime-preflight",
        "registry-bootstrap",
        "capacity-preflight",
        "corpus-materialization",
        "worker-images",
        "matrix-lock",
        "plugin-build",
        "sanitizers",
    ),
}

# These paths were already semantic gates in marker v1. They remain a second
# line of defense in addition to exact file inventory.
PASSING_JSON: dict[str, tuple[str, ...]] = {
    "dependency-audit": ("supply-chain/triage.json",),
    "target-readiness": ("target-readiness/validation.json",),
    "profiler-preflight": ("profiler-preflight/validation.json",),
    "aa-pilot": ("aa/validation.json",),
    "core-qualification": ("core-run/qualification-summary.json",),
    "gpu-smoke": ("smoke/validation.json",),
    "plugin-benchmark": (
        "plugin-benchmark/plugin-benchmark.json",
        "plugin-benchmark/plugin-benchmark-validity.json",
    ),
    "plugin-matrix": ("plugin-runs/validation.json",),
    "mobilenet-matrix": ("mobilenet-runs/validation.json",),
    "gpu-faults": ("gpu-faults/validation.json",),
    "reduction-validation": ("reductions/validation.json",),
    "memory-seed": ("memory-seed/validation.json",),
    "sanitizers": ("sanitizers/validation.json",),
    "profiles": ("profiles/validation.json",),
    "sboms": ("sbom/validation.json",),
    "final-evidence": ("evidence.json", "results.json"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("record", "verify", "reconcile"):
        child = subparsers.add_parser(command)
        child.add_argument("--state", type=Path, required=True)
        child.add_argument("--project", type=Path, required=True)
        if command != "reconcile":
            child.add_argument("--step", choices=_public_step_names(), required=True)
        child.add_argument("--source", required=True)
        child.add_argument("--gpu", required=True)
        child.add_argument("--mode", choices=MODES, required=True)
        if command == "reconcile":
            child.add_argument("--dry-run", action="store_true")
    corpus = subparsers.add_parser("verify-corpus")
    corpus.add_argument("path", type=Path)
    materializer = corpus.add_mutually_exclusive_group()
    materializer.add_argument("--materializer-id", "--materializer-sha256", dest="materializer_id")
    materializer.add_argument("--materializer-sidecar", type=Path)
    workers = subparsers.add_parser("capture-workers")
    workers.add_argument("--output", type=Path, required=True)
    workers.add_argument("images", nargs=2)
    worker_lock = subparsers.add_parser("verify-worker-lock")
    worker_lock.add_argument("--workers", type=Path, required=True)
    worker_lock.add_argument("--matrix", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "verify-corpus":
        _verify_corpus(
            arguments.path.resolve(strict=True),
            materializer_id=arguments.materializer_id,
            materializer_sidecar=arguments.materializer_sidecar,
        )
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
        return _verify_worker_lock(arguments.workers, arguments.matrix)

    state = arguments.state.resolve(strict=True)
    project = arguments.project.resolve(strict=True)
    if arguments.command == "reconcile":
        result = reconcile(
            state,
            project,
            source=arguments.source,
            gpu=arguments.gpu,
            mode=arguments.mode,
            dry_run=arguments.dry_run,
        )
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
        return 0
    if arguments.command == "record":
        record_marker(
            state,
            project,
            arguments.step,
            arguments.source,
            arguments.gpu,
            arguments.mode,
        )
        return 0
    return (
        0
        if verify_marker(
            state,
            project,
            arguments.step,
            arguments.source,
            arguments.gpu,
            arguments.mode,
        )
        else 1
    )


def record_marker(
    state: Path,
    project: Path,
    step: str,
    source: str,
    gpu: str,
    mode: str,
) -> dict[str, Any]:
    """Validate direct dependencies and atomically publish one marker v2."""

    canonical = _canonical_step(step)
    _require_mode_step(canonical, mode)
    for dependency in _dependencies_for(canonical, mode):
        if not verify_marker(state, project, dependency, source, gpu, mode):
            raise ValueError(f"direct dependency marker is invalid: {dependency}")
    payload = _payload(state, project, step, source, gpu, mode)
    _write_atomic(_marker_path(state, canonical), payload)
    return payload


def verify_marker(
    state: Path,
    project: Path,
    step: str,
    source: str,
    gpu: str,
    mode: str,
    *,
    _memo: dict[str, bool] | None = None,
) -> bool:
    """Verify a marker, its owned inventory, and its dependency chain."""

    canonical = _canonical_step(step)
    try:
        _require_mode_step(canonical, mode)
    except ValueError:
        return False
    memo = {} if _memo is None else _memo
    if canonical in memo:
        return memo[canonical]
    memo[canonical] = False
    for dependency in _dependencies_for(canonical, mode):
        if not verify_marker(
            state,
            project,
            dependency,
            source,
            gpu,
            mode,
            _memo=memo,
        ):
            return False
    try:
        marker = _marker_path(state, canonical)
        observed = _read_json_object(marker)
        expected = _payload(state, project, step, source, gpu, mode)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    valid = observed == expected
    memo[canonical] = valid
    return valid


def _payload(
    state: Path,
    project: Path,
    step: str,
    source: str,
    gpu: str,
    mode: str,
) -> dict[str, Any]:
    """Build the deterministic marker v2 payload for one step."""

    canonical = _canonical_step(step)
    _require_mode_step(canonical, mode)
    inventory = _inventory_owned(state, canonical, invoked_step=step)
    _validate_passing_json(state, canonical)
    dependency_hashes = {
        dependency: _sha256(_marker_path(state, dependency))
        for dependency in _dependencies_for(canonical, mode)
    }
    matrix_lock_sha256 = _matrix_binding(state) if _is_matrix_bound(canonical, mode) else None
    corpus_identities = (
        _corpus_identities(
            state,
            project,
            mode,
            verify_contents=canonical == "corpus-materialization",
        )
        if _is_corpus_bound(canonical, mode)
        else []
    )
    return {
        "schema_version": MARKER_SCHEMA,
        "step": canonical,
        "source_git_commit": source,
        "gpu_uuid": gpu,
        "mode": mode,
        "inventory": inventory,
        "direct_dependency_marker_sha256s": dependency_hashes,
        "matrix_lock_sha256": matrix_lock_sha256,
        "corpus_identities": corpus_identities,
    }


def reconcile(
    state: Path,
    project: Path,
    *,
    source: str,
    gpu: str,
    mode: str,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Preserve invalid markers and owned paths without touching valid siblings."""

    order = _topological_steps(mode)
    validity: dict[str, bool] = {}
    for step in order:
        dependencies_valid = all(
            validity[dependency] for dependency in _dependencies_for(step, mode)
        )
        validity[step] = dependencies_valid and verify_marker(
            state,
            project,
            step,
            source,
            gpu,
            mode,
        )
    invalid = [step for step in order if not validity[step]]
    lineage = _observed_lineage(state, source)
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    stale_root = _unique_stale_root(state / "stale" / f"{timestamp}-{lineage}")
    moves: list[dict[str, str]] = []
    for step in invalid:
        destinations = stale_root / step
        for relative in _reconcile_owned_paths(step):
            source_path = state / relative
            if not os.path.lexists(source_path):
                continue
            destination = destinations / relative
            moves.append(
                {
                    "step": step,
                    "source": relative,
                    "destination": destination.relative_to(state).as_posix(),
                }
            )
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(destination))
        marker = _marker_path(state, step)
        if os.path.lexists(marker):
            relative = marker.relative_to(state).as_posix()
            destination = destinations / relative
            moves.append(
                {
                    "step": step,
                    "source": relative,
                    "destination": destination.relative_to(state).as_posix(),
                }
            )
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(marker), str(destination))
    return {
        "schema_version": "upgradeguard.dev/qualification-reconcile/v1",
        "mode": mode,
        "invalid_steps": invalid,
        "valid_steps": [step for step in order if validity[step]],
        "dry_run": dry_run,
        "stale_root": stale_root.relative_to(state).as_posix(),
        "moves": moves,
    }


def _inventory_owned(state: Path, step: str, *, invoked_step: str) -> list[dict[str, object]]:
    del invoked_step
    entries: dict[str, dict[str, object]] = {}
    for authored in STEP_OWNED_PATHS[step]:
        relative = authored.removesuffix("/")
        path = state / relative
        _inventory_required_path(state, path, directory=authored.endswith("/"), entries=entries)
    observed_logs = [
        state / relative for relative in _step_log_paths(step) if os.path.lexists(state / relative)
    ]
    if not observed_logs:
        raise ValueError(f"required closed step log is absent: {step}")
    for log_path in observed_logs:
        _inventory_required_path(state, log_path, directory=False, entries=entries)
    return [entries[path] for path in sorted(entries)]


def _inventory_required_path(
    state: Path,
    path: Path,
    *,
    directory: bool,
    entries: dict[str, dict[str, object]],
) -> None:
    relative = _safe_state_relative(state, path)
    _reject_symlink_parents(state, path)
    try:
        root_stat = path.lstat()
    except OSError as error:
        raise ValueError(f"required step output is absent: {relative}") from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"owned path cannot be a symlink: {relative}")
    if directory:
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"owned directory changed type: {relative}")
        before = len(entries)
        for current_root, directory_names, file_names in os.walk(path, followlinks=False):
            current = Path(current_root)
            for name in sorted(directory_names):
                child = current / name
                child_stat = child.lstat()
                child_relative = _safe_state_relative(state, child)
                if stat.S_ISLNK(child_stat.st_mode):
                    raise ValueError(f"owned path cannot be a symlink: {child_relative}")
                if not stat.S_ISDIR(child_stat.st_mode):
                    raise ValueError(f"owned path is not a regular directory: {child_relative}")
            for name in sorted(file_names):
                _inventory_regular_file(state, current / name, entries)
        if len(entries) == before:
            raise ValueError(f"required owned directory is empty: {relative}")
        return
    if not stat.S_ISREG(root_stat.st_mode):
        raise ValueError(f"owned output is not a regular file: {relative}")
    _inventory_regular_file(state, path, entries)


def _inventory_regular_file(state: Path, path: Path, entries: dict[str, dict[str, object]]) -> None:
    relative = _safe_state_relative(state, path)
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"owned output is not a regular file: {relative}")
    if relative in entries:
        raise ValueError(f"owned file appears more than once: {relative}")
    entries[relative] = {
        "path": relative,
        "bytes": file_stat.st_size,
        "sha256": _sha256(path),
    }


def _safe_state_relative(state: Path, path: Path) -> str:
    try:
        relative = path.relative_to(state)
    except ValueError as error:
        raise ValueError(f"owned path escapes state root: {path}") from error
    value = PurePosixPath(relative.as_posix())
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise ValueError(f"owned path is unsafe: {relative}")
    if value.parts[0] in _RESERVED_STATE_ROOTS:
        raise ValueError(f"owned path uses state-engine metadata: {relative}")
    return value.as_posix()


def _reject_symlink_parents(state: Path, path: Path) -> None:
    relative = path.relative_to(state)
    current = state
    for part in relative.parts[:-1]:
        current /= part
        try:
            current_stat = current.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(f"owned path has a symlink parent: {current.relative_to(state)}")


def _validate_passing_json(state: Path, step: str) -> None:
    for relative in PASSING_JSON.get(step, ()):
        value = _read_json_object(state / relative)
        if value.get("status") != "passed":
            raise ValueError(f"required JSON did not pass: {state / relative}")
    if step == "aa-pilot":
        value = _read_json_object(state / "aa" / "validation.json")
        if value.get("false_positive") is not False or value.get("accepted_pairs") != 20:
            raise ValueError("A/A pilot did not retain exactly 20 passing pairs")


def _matrix_binding(state: Path) -> str:
    from upgrade_guard.contracts.environment import MatrixLock

    path = state / "matrix.lock.json"
    try:
        lock = MatrixLock.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("matrix lock is absent or invalid") from error
    if lock.computed_sha256() != lock.lock_sha256:
        raise ValueError("matrix lock self-hash differs")
    return lock.lock_sha256


def _corpus_identities(
    state: Path,
    project: Path,
    mode: str,
    *,
    verify_contents: bool,
) -> list[dict[str, str]]:
    value = _read_json_object(state / "corpora.json")
    if value.get("schema_version") != "upgradeguard.dev/corpus-index/v1":
        raise ValueError("corpora.json has an unsupported schema")
    corpora = value.get("corpora")
    if not isinstance(corpora, dict) or not corpora:
        raise ValueError("corpora.json must contain a nonempty corpora mapping")
    expected_kinds = (
        {"core", "plugin"} if mode in {"smoke", "sanitizer"} else {"core", "plugin", "mobilenet"}
    )
    if set(corpora) != expected_kinds:
        raise ValueError(f"corpora.json kinds differ for {mode} mode")

    identities: list[dict[str, str]] = []
    required_fields = {
        "root",
        "lock",
        "lock_sha256",
        "materializer_sha256",
        "inventory_sha256",
    }
    for kind, corpus in sorted(corpora.items()):
        if not isinstance(kind, str) or not kind or not isinstance(corpus, dict):
            raise ValueError("corpus index entries must map a kind to an object")
        if set(corpus) != required_fields:
            raise ValueError(f"corpus index entry has unexpected fields: {kind}")
        root = corpus.get("root")
        lock = corpus.get("lock")
        lock_sha256 = corpus.get("lock_sha256")
        materializer_sha256 = corpus.get("materializer_sha256")
        inventory_sha256 = corpus.get("inventory_sha256")
        values = (root, lock, lock_sha256, materializer_sha256, inventory_sha256)
        if not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"corpus index entry is incomplete: {kind}")
        assert isinstance(root, str)
        assert isinstance(lock, str)
        assert isinstance(lock_sha256, str)
        assert isinstance(materializer_sha256, str)
        assert isinstance(inventory_sha256, str)
        root_path = PurePosixPath(root)
        lock_path = PurePosixPath(lock)
        if (
            not _safe_project_relative(root_path)
            or not _safe_project_relative(lock_path)
            or not lock_path.is_relative_to(root_path)
        ):
            raise ValueError(f"corpus index paths are unsafe: {kind}")
        if not all(
            _is_sha256(identity)
            for identity in (lock_sha256, materializer_sha256, inventory_sha256)
        ):
            raise ValueError("corpus lock, materializer, and inventory identities must be SHA-256")
        if verify_contents:
            resolved_root = (project / root).resolve(strict=True)
            resolved_lock = (project / lock).resolve(strict=True)
            if (
                not resolved_root.is_relative_to(project)
                or resolved_root.is_symlink()
                or not resolved_root.is_dir()
                or not resolved_lock.is_relative_to(resolved_root)
                or resolved_lock.is_symlink()
                or not resolved_lock.is_file()
            ):
                raise ValueError(f"corpus index resolves to unsafe paths: {kind}")
            _verify_corpus(resolved_root, materializer_id=materializer_sha256)
            if _sha256(resolved_lock) != lock_sha256:
                raise ValueError(f"corpus lock identity differs: {kind}")
            if _corpus_inventory_sha256(resolved_root) != inventory_sha256:
                raise ValueError(f"corpus inventory identity differs: {kind}")
        identities.append(
            {
                "kind": kind,
                "root": root,
                "lock": lock,
                "lock_sha256": lock_sha256,
                "materializer_sha256": materializer_sha256,
                "inventory_sha256": inventory_sha256,
            }
        )
    return identities


def _corpus_inventory_sha256(root: Path) -> str:
    inventory: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise ValueError("corpus inventory contains a symlink")
        if stat.S_ISDIR(path_stat.st_mode):
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("corpus inventory contains a non-regular file")
        inventory[path.relative_to(root).as_posix()] = {
            "bytes": path_stat.st_size,
            "sha256": _sha256(path),
        }
    canonical = json.dumps(
        inventory,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _verify_corpus(
    root: Path,
    *,
    materializer_id: str | None = None,
    materializer_sidecar: Path | None = None,
) -> None:
    if materializer_id is not None and materializer_sidecar is not None:
        raise ValueError("only one explicit materializer identity source is allowed")
    root_sidecar = root / "materializer.json"
    observed_materializer = _read_materializer_sidecar(root_sidecar)
    expected_materializer = observed_materializer
    if materializer_id is not None:
        expected_materializer = materializer_id
    elif materializer_sidecar is not None:
        expected_materializer = _read_materializer_sidecar(materializer_sidecar)
    if not isinstance(expected_materializer, str) or not _is_sha256(expected_materializer):
        raise ValueError("materializer identity must be an algorithm-qualified SHA-256 value")
    if observed_materializer != expected_materializer:
        raise ValueError("corpus materializer sidecar differs from the expected identity")
    if root.name != expected_materializer.removeprefix("sha256:"):
        raise ValueError("corpus path does not match its materializer identity")

    lock_names = (
        "corpus.lock.json",
        "plugin-corpus.lock.json",
        "mobilenet-corpus.lock.json",
    )
    locks = [root / name for name in lock_names if (root / name).is_file()]
    if len(locks) != 1:
        raise ValueError("corpus must contain exactly one supported lock")
    value = _read_json_object(locks[0])
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
        path = root / relative
        try:
            path_stat = path.lstat()
        except OSError as error:
            raise ValueError(f"corpus artifact is absent: {relative}") from error
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("corpus artifact must be a regular file")
        if path_stat.st_size != size or _sha256(path) != digest:
            raise ValueError(f"corpus artifact differs from its lock: {relative}")
        expected.add(relative)
    observed: set[str] = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            child = current / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError("corpus inventory contains an unsafe directory")
        for name in file_names:
            path = current / name
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("corpus inventory contains a non-regular file")
            if path not in locks and path != root_sidecar:
                observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise ValueError("corpus inventory differs from its lock")


def _read_materializer_sidecar(path: Path) -> str:
    try:
        sidecar_stat = path.lstat()
    except OSError as error:
        raise ValueError("corpus lacks its materializer sidecar") from error
    if stat.S_ISLNK(sidecar_stat.st_mode) or not stat.S_ISREG(sidecar_stat.st_mode):
        raise ValueError("materializer sidecar must be a safe regular file")
    value = _read_json_object(path)
    materializer_sha256 = value.get("materializer_sha256")
    if not isinstance(materializer_sha256, str):
        raise ValueError("materializer sidecar has no materializer_sha256")
    return materializer_sha256


def _safe_project_relative(path: PurePosixPath) -> bool:
    return (
        not path.is_absolute()
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _verify_worker_lock(workers: Path, matrix: Path) -> int:
    workers_value = _read_json_object(workers)
    matrix_value = _read_json_object(matrix)
    worker_digests = {value["manifest_digest"] for value in workers_value.get("images", [])}
    matrix_digests = {
        value["worker_image"]["manifest_digest"] for value in matrix_value.get("environments", [])
    }
    return 0 if len(worker_digests) == 2 and worker_digests == matrix_digests else 1


def _canonical_step(step: str) -> str:
    canonical = STEP_ALIASES.get(step, step)
    if canonical not in STEP_OWNED_PATHS:
        raise ValueError(f"unknown qualification step: {step}")
    return canonical


def _public_step_names() -> tuple[str, ...]:
    return tuple(sorted((*STEP_OWNED_PATHS, *STEP_ALIASES)))


def _dependencies_for(step: str, mode: str) -> tuple[str, ...]:
    if step == "final-evidence":
        return tuple(
            candidate
            for candidate in MODE_STEPS[mode]
            if candidate not in {"final-evidence", "terminal-cleanup"}
        )
    return STEP_DEPENDENCIES[step]


def _require_mode_step(step: str, mode: str) -> None:
    if mode not in MODE_STEPS:
        raise ValueError(f"unknown qualification mode: {mode}")
    if step not in MODE_STEPS[mode]:
        raise ValueError(f"step {step} is not part of {mode} mode")


def _marker_path(state: Path, step: str) -> Path:
    return state / "done" / f"{_canonical_step(step)}.json"


def _is_matrix_bound(step: str, mode: str) -> bool:
    return step == "matrix-lock" or _depends_on(step, "matrix-lock", mode)


def _is_corpus_bound(step: str, mode: str) -> bool:
    return step == "corpus-materialization" or _depends_on(step, "corpus-materialization", mode)


def _depends_on(step: str, target: str, mode: str) -> bool:
    pending = list(_dependencies_for(step, mode))
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(_dependencies_for(current, mode))
    return False


def _topological_steps(mode: str) -> list[str]:
    if mode not in MODE_STEPS:
        raise ValueError(f"unknown qualification mode: {mode}")
    remaining = set(MODE_STEPS[mode])
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            step for step in remaining if set(_dependencies_for(step, mode)).issubset(ordered)
        )
        if not ready:
            raise ValueError(f"qualification dependency graph contains a cycle in {mode} mode")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def _reconcile_owned_paths(step: str) -> tuple[str, ...]:
    paths = [path.removesuffix("/") for path in STEP_OWNED_PATHS[step]]
    paths.extend(_step_log_paths(step))
    return tuple(sorted(paths))


def _step_log_paths(step: str) -> tuple[str, ...]:
    public_names = [step, *(alias for alias, target in STEP_ALIASES.items() if target == step)]
    return tuple(f"logs/{name}.log" for name in sorted(public_names))


def _observed_lineage(state: Path, fallback: str) -> str:
    candidates = [_marker_path(state, "preflight"), *sorted((state / "done").glob("*.json"))]
    for marker in candidates:
        try:
            value = _read_json_object(marker)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        source = value.get("source_git_commit")
        if isinstance(source, str) and source:
            return _lineage_fragment(source)
    return _lineage_fragment(fallback)


def _lineage_fragment(value: str) -> str:
    normalized = "".join(character for character in value.lower() if character.isalnum())
    return (normalized or "unknown")[:16]


def _unique_stale_root(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    for index in range(1, 10_000):
        alternative = candidate.with_name(f"{candidate.name}-{index}")
        if not alternative.exists():
            return alternative
    raise ValueError("could not allocate a unique stale preservation root")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _is_sha256(value: str) -> bool:
    prefix, separator, digest = value.partition(":")
    return (
        prefix == "sha256"
        and separator == ":"
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


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


def validate_authority() -> None:
    """Reject an incoherent ownership table or dependency graph at import time."""

    nodes = set(STEP_OWNED_PATHS)
    if set(STEP_DEPENDENCIES) != nodes:
        raise ValueError("ownership and dependency step sets differ")
    if any(target not in nodes for target in STEP_ALIASES.values()):
        raise ValueError("step alias targets an unknown owner")
    owners: list[tuple[PurePosixPath, str]] = []
    for step in sorted(nodes):
        for authored in _reconcile_owned_paths(step):
            path = PurePosixPath(authored)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
                or path.parts[0] in _RESERVED_STATE_ROOTS
            ):
                raise ValueError(f"unsafe owned path for {step}: {authored}")
            for other, owner in owners:
                if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                    raise ValueError(
                        "owned paths overlap: "
                        f"{step}:{path.as_posix()} and {owner}:{other.as_posix()}"
                    )
            owners.append((path, step))
    for step, dependencies in STEP_DEPENDENCIES.items():
        if step in dependencies or any(dependency not in nodes for dependency in dependencies):
            raise ValueError(f"invalid dependencies for {step}")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"duplicate dependencies for {step}")
    for mode, steps in MODE_STEPS.items():
        if len(steps) != len(set(steps)) or any(step not in nodes for step in steps):
            raise ValueError(f"invalid step inventory for {mode} mode")
        for step in steps:
            if not set(_dependencies_for(step, mode)).issubset(steps):
                raise ValueError(f"{mode} mode omits a dependency of {step}")
        _topological_steps(mode)


validate_authority()


if __name__ == "__main__":
    raise SystemExit(main())
