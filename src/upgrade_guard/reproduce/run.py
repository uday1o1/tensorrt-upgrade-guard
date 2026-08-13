"""Trust-gated reproduction preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from upgrade_guard.errors import UnsupportedEnvironmentError
from upgrade_guard.reproduce.verify import verify_bundle


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Verified typed replay plan that never invokes bundled scripts."""

    bundle_id: str
    selected_gpu_uuid: str | None
    worker_images: tuple[str, ...]
    build_commands: tuple[tuple[str, ...], ...]
    source_paths: tuple[str, ...]
    included_engine_trusted: bool


def prepare_replay(
    source: Path,
    *,
    trust_source_code: bool,
    trust_included_engine: bool,
) -> ReplayPlan:
    """Verify identity and require explicit trust for executable artifacts."""

    verified = verify_bundle(source)
    manifest = verified.manifest
    if verified.source_code_present and not trust_source_code:
        source_paths = (
            tuple(artifact.path for artifact in manifest.source_build.sources)
            if manifest.source_build
            else ()
        )
        raise UnsupportedEnvironmentError(
            "bundle contains source code; pass --trust-source-code after review",
            details={"source_paths": list(source_paths)},
        )
    if verified.engine_present and not trust_included_engine:
        raise UnsupportedEnvironmentError(
            "bundle contains a serialized engine; pass --trust-included-engine after review",
            details={"engine": manifest.included_engine.path if manifest.included_engine else None},
        )
    source_build = manifest.source_build
    return ReplayPlan(
        bundle_id=manifest.id,
        selected_gpu_uuid=source_build.selected_gpu_uuid if source_build else None,
        worker_images=((source_build.worker_image_manifest_digest,) if source_build else ()),
        build_commands=((source_build.command,) if source_build else ()),
        source_paths=(
            tuple(artifact.path for artifact in source_build.sources) if source_build else ()
        ),
        included_engine_trusted=verified.engine_present and trust_included_engine,
    )


def require_gpu_for_replay() -> None:
    """Mark the exact boundary between safe preparation and GPU execution."""

    raise UnsupportedEnvironmentError(
        "verified replay preparation is complete; GPU worker execution is required"
    )
