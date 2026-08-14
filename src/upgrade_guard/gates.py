"""Authoritative qualification step and terminal-publication ordering."""

from __future__ import annotations

from typing import Literal

MARKER_SCHEMA = "upgradeguard.dev/qualification-step/v3"

DOMAIN_FAILURE_STEPS = (
    "core-qualification",
    "plugin-matrix",
    "mobilenet-matrix",
)

FULL_MODE_STEPS = (
    "preflight",
    "cpu-verify",
    "gpu-runtime-preflight",
    "dependency-audit",
    "registry-bootstrap",
    "capacity-preflight",
    "reference-environment",
    "corpus-materialization",
    "worker-images",
    "matrix-lock",
    "matrix-live-verification",
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
    "public-failure",
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
)

SMOKE_MODE_STEPS = (
    "preflight",
    "cpu-verify",
    "gpu-runtime-preflight",
    "registry-bootstrap",
    "capacity-preflight",
    "reference-environment",
    "corpus-materialization",
    "worker-images",
    "matrix-lock",
    "matrix-live-verification",
    "plugin-build",
    "gpu-smoke",
)

SANITIZER_MODE_STEPS = (
    "preflight",
    "cpu-verify",
    "gpu-runtime-preflight",
    "registry-bootstrap",
    "capacity-preflight",
    "reference-environment",
    "corpus-materialization",
    "worker-images",
    "matrix-lock",
    "matrix-live-verification",
    "plugin-build",
    "sanitizers",
)

MODE_STEPS = {
    "full": FULL_MODE_STEPS,
    "smoke": SMOKE_MODE_STEPS,
    "sanitizer": SANITIZER_MODE_STEPS,
}

# These files are created only after the terminal publication inventory is closed.
# They are exact lifecycle metadata, not qualification evidence.
POST_PUBLICATION_ARTIFACTS = frozenset(
    {
        "cleanup.json",
        "done/final-evidence.json",
        "done/terminal-cleanup.json",
        "logs/final-evidence.log",
        "logs/terminal-cleanup.log",
    }
)

# Alternative public names normalize to one authority owner and one marker.
STEP_ALIASES: dict[str, str] = {
    "plugin-compile-test": "plugin-build",
    "reduction-replay": "reduction-validation",
}

# Values are paths relative to one source-run state root. A trailing slash denotes
# a required nonempty directory. Every other value denotes a required file.
STEP_OWNED_PATHS: dict[str, tuple[str, ...]] = {
    "preflight": ("source.commit", "gpu.uuid", "gpu-preflight.csv"),
    "cpu-verify": (),
    "gpu-runtime-preflight": ("gpu-runtime-preflight.json",),
    "registry-bootstrap": ("registry-identity.json",),
    "capacity-preflight": ("capacity/",),
    "reference-environment": ("reference-environment.lock.json",),
    "worker-images": ("worker-images.json",),
    "matrix-lock": ("matrix.yaml", "full.yaml", "core.yaml", "matrix.lock.json"),
    "matrix-live-verification": ("matrix-live-verification.json",),
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
    "public-failure": ("public-failure/",),
    "fault-inputs": ("fault-inputs/",),
    "gpu-faults": ("gpu-faults/",),
    "reduction-prepare": (
        "reductions/prepared/",
        "reductions/candidate-reduction-work/",
    ),
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

# Direct dependency edges only. Dynamic terminal dependencies are resolved by the
# state engine and public publication validator.
STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "preflight": (),
    "cpu-verify": ("preflight",),
    "gpu-runtime-preflight": ("cpu-verify",),
    "registry-bootstrap": ("cpu-verify",),
    "capacity-preflight": ("preflight", "registry-bootstrap"),
    "dependency-audit": ("cpu-verify",),
    "reference-environment": (
        "cpu-verify",
        "registry-bootstrap",
        "capacity-preflight",
    ),
    "corpus-materialization": (
        "gpu-runtime-preflight",
        "capacity-preflight",
        "reference-environment",
    ),
    "worker-images": (
        "cpu-verify",
        "registry-bootstrap",
        "capacity-preflight",
    ),
    "matrix-lock": ("worker-images", "gpu-runtime-preflight"),
    "matrix-live-verification": ("matrix-lock",),
    "plugin-build": ("matrix-live-verification", "corpus-materialization"),
    "target-readiness": (
        "plugin-build",
        "matrix-live-verification",
        "corpus-materialization",
    ),
    "profiler-preflight": ("plugin-build",),
    "aa-pilot": ("matrix-lock", "corpus-materialization"),
    "core-qualification": (
        "aa-pilot",
        "matrix-live-verification",
        "corpus-materialization",
    ),
    "gpu-smoke": ("matrix-live-verification", "corpus-materialization", "plugin-build"),
    "plugin-benchmark": ("plugin-build", "profiler-preflight"),
    "plugin-matrix": (
        "matrix-live-verification",
        "corpus-materialization",
        "plugin-build",
    ),
    "mobilenet-matrix": ("matrix-live-verification", "corpus-materialization"),
    "public-failure": DOMAIN_FAILURE_STEPS,
    "fault-inputs": ("corpus-materialization",),
    "gpu-faults": (
        "matrix-live-verification",
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
        "matrix-live-verification",
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
    "sanitizers": ("plugin-build", "matrix-live-verification"),
    "profiles": (
        "profiler-preflight",
        "plugin-build",
        "plugin-benchmark",
        "matrix-live-verification",
    ),
    "sboms": ("worker-images", "matrix-live-verification"),
    "final-evidence": (),
    "terminal-cleanup": ("final-evidence",),
}

_NON_GATE_STEPS = frozenset({"public-failure", "final-evidence", "terminal-cleanup"})


def expected_publication_steps(
    status: Literal["passed", "failed"],
    *,
    failure_step: str | None = None,
) -> tuple[str, ...]:
    """Return the exact ordered marker set a full terminal publication must retain."""

    candidates = tuple(step for step in FULL_MODE_STEPS if step not in _NON_GATE_STEPS)
    if status == "passed":
        if failure_step is not None:
            raise ValueError("passing publication cannot name a failed step")
        return candidates
    if failure_step not in DOMAIN_FAILURE_STEPS:
        raise ValueError("failed publication must name one domain failure step")
    return (*candidates[: candidates.index(failure_step) + 1], "public-failure")


def direct_step_dependencies(step: str, *, failure_step: str | None = None) -> tuple[str, ...]:
    """Return the exact direct dependency set retained in a terminal marker."""

    if step == "public-failure":
        if failure_step not in DOMAIN_FAILURE_STEPS:
            raise ValueError("public failure marker requires one failed domain step")
        return (failure_step,)
    try:
        return STEP_DEPENDENCIES[step]
    except KeyError as error:
        raise ValueError(f"unknown qualification step: {step}") from error


def step_is_bound_to(step: str, target: str, *, failure_step: str | None = None) -> bool:
    """Return whether a marker is transitively bound to another step."""

    pending = list(direct_step_dependencies(step, failure_step=failure_step))
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(direct_step_dependencies(current, failure_step=failure_step))
    return False
