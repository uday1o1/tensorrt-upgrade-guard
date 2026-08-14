"""Trust-gated typed reproduction preparation and execution."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.contracts.base import StrictModel
from upgrade_guard.contracts.environment import EnvironmentLock
from upgrade_guard.errors import InvalidInputError, UnsupportedEnvironmentError
from upgrade_guard.reproduce.verify import materialize_verified_bundle, verify_bundle

JsonScalar = bool | int | float | str | None


class ReplayStep(StrictModel):
    """One typed command and its machine-checkable expected evidence."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    command: tuple[str, ...]
    timeout_seconds: int = Field(default=600, ge=1, le=3600)
    accepted_returncodes: tuple[int, ...] = (0,)
    result_file: str | None = None
    expected_result_status: Literal["passed", "failed"] | None = None
    result_message_contains: str | None = None
    stdout_json_equals: dict[str, JsonScalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution_contract(self) -> ReplayStep:
        if not self.command or any(not item or "\x00" in item for item in self.command):
            raise ValueError("replay commands must be nonempty NUL-free argument arrays")
        if (
            not self.accepted_returncodes
            or len(self.accepted_returncodes) != len(set(self.accepted_returncodes))
            or any(code < 0 for code in self.accepted_returncodes)
        ):
            raise ValueError("accepted return codes must be unique nonnegative values")
        if (self.result_file is None) != (self.expected_result_status is None):
            raise ValueError("result_file and expected_result_status must be authored together")
        if self.result_message_contains is not None and self.result_file is None:
            raise ValueError("result_message_contains requires result_file")
        if self.result_file is not None:
            _safe_relative(self.result_file)
        return self


class ReplayRecipe(StrictModel):
    """Hash-verified command sequence stored inside a source-bearing bundle."""

    schema_version: Literal["upgradeguard.dev/replay-recipe/v1"]
    expected_failure_code: str
    steps: tuple[ReplayStep, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_unique_steps(self) -> ReplayRecipe:
        identifiers = [step.id for step in self.steps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("replay step identifiers must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Verified typed replay plan that never invokes bundled scripts."""

    bundle_id: str
    selected_gpu_uuid: str | None
    worker_images: tuple[str, ...]
    build_commands: tuple[tuple[str, ...], ...]
    source_paths: tuple[str, ...]
    included_engine_trusted: bool


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Successful clean replay evidence emitted by the public CLI."""

    schema_version: str
    status: str
    bundle_id: str
    bundle_manifest_sha256: str
    worker_image: str
    selected_gpu_uuid: str
    expected_failure_code: str
    step_results: tuple[str, ...]


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


def execute_replay(
    source: Path,
    output: Path,
    *,
    trust_source_code: bool,
    trust_included_engine: bool,
    worker: DockerGpuWorker | None = None,
) -> ReplayResult:
    """Materialize a verified bundle and execute only its typed recipe."""

    plan = prepare_replay(
        source,
        trust_source_code=trust_source_code,
        trust_included_engine=trust_included_engine,
    )
    if output.exists() or output.is_symlink():
        raise InvalidInputError("refusing to overwrite replay output")
    if source.is_dir() and output.resolve().is_relative_to(source.resolve()):
        raise InvalidInputError("replay output must be outside the verified bundle directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        bundle_root = staging / "bundle"
        verified = materialize_verified_bundle(source, bundle_root)
        manifest = verified.manifest
        source_build = manifest.source_build
        if source_build is None:
            raise UnsupportedEnvironmentError(
                "V1 replay requires a source-bearing engine rebuild recipe"
            )
        try:
            environment = EnvironmentLock.model_validate_json(
                (bundle_root / manifest.candidate_environment.path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValidationError) as error:
            raise InvalidInputError("candidate environment lock is invalid") from error
        if source_build.worker_image_manifest_digest != environment.worker_image.manifest_digest:
            raise InvalidInputError("replay worker identity differs from candidate environment")
        if source_build.selected_gpu_uuid != environment.probe.gpu.uuid:
            raise InvalidInputError("replay GPU identity differs from candidate environment")
        recipe_path = bundle_root / "commands" / "replay.json"
        if "commands/replay.json" not in {item.path for item in manifest.files}:
            raise InvalidInputError("bundle has no hash-verified typed replay recipe")
        try:
            recipe = ReplayRecipe.model_validate_json(recipe_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError) as error:
            raise InvalidInputError("typed replay recipe is invalid") from error
        if recipe.expected_failure_code != manifest.expected_failure.code.value:
            raise InvalidInputError("replay recipe failure code differs from bundle manifest")
        if recipe.steps[0].command != source_build.command:
            raise InvalidInputError(
                "first replay step differs from the reviewed source build command"
            )

        work = staging / "work"
        logs = staging / "steps"
        work.mkdir()
        logs.mkdir()
        gpu_worker = worker or DockerGpuWorker()
        completed: list[str] = []
        for step in recipe.steps:
            result = gpu_worker.run(
                image=environment.worker_image.canonical_reference,
                gpu_uuid=source_build.selected_gpu_uuid,
                mounts=WorkerMounts(source=bundle_root, corpus=bundle_root, output=work),
                command=step.command,
                timeout_seconds=step.timeout_seconds,
                accepted_returncodes=step.accepted_returncodes,
            )
            _validate_step_evidence(step, result.stdout, work)
            record = {
                "schema_version": "upgradeguard.dev/replay-step/v1",
                "status": "passed",
                "id": step.id,
                "command": list(step.command),
                "accepted_returncodes": list(step.accepted_returncodes),
                "observed_returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            (logs / f"{step.id}.json").write_text(
                json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            completed.append(step.id)
        replay = ReplayResult(
            schema_version="upgradeguard.dev/replay-result/v1",
            status="passed",
            bundle_id=plan.bundle_id,
            bundle_manifest_sha256=manifest.manifest_sha256,
            worker_image=environment.worker_image.canonical_reference,
            selected_gpu_uuid=source_build.selected_gpu_uuid,
            expected_failure_code=manifest.expected_failure.code.value,
            step_results=tuple(completed),
        )
        (staging / "replay-result.json").write_text(
            json.dumps(asdict(replay), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
        return replay
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_step_evidence(step: ReplayStep, stdout: str, work: Path) -> None:
    if step.result_file is not None:
        result_path = work / _safe_relative(step.result_file)
        if not result_path.is_file() or result_path.is_symlink():
            raise InvalidInputError(f"replay step did not produce its result: {step.id}")
        value = _json_object(result_path.read_text(encoding="utf-8"), step.id)
        if value.get("status") != step.expected_result_status:
            raise InvalidInputError(f"replay step result status differed: {step.id}")
        if step.result_message_contains is not None and step.result_message_contains not in str(
            value.get("message", "")
        ):
            raise InvalidInputError(f"replay step failed for a different reason: {step.id}")
    if step.stdout_json_equals:
        value = _json_object(stdout, step.id)
        for authored_path, expected in step.stdout_json_equals.items():
            observed: object = value
            for component in authored_path.split("."):
                if not isinstance(observed, Mapping) or component not in observed:
                    raise InvalidInputError(
                        f"replay stdout lacks {authored_path!r} for step {step.id}"
                    )
                observed = observed[component]
            if observed != expected:
                raise InvalidInputError(
                    f"replay stdout predicate differed for {authored_path!r}: {step.id}"
                )


def _json_object(text: str, step_id: str) -> dict[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidInputError(f"replay step did not emit valid JSON: {step_id}") from error
    if not isinstance(value, dict):
        raise InvalidInputError(f"replay step did not emit a JSON object: {step_id}")
    return value


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("replay result path must be a safe relative path")
    return path


def require_gpu_for_replay() -> None:
    """Retained compatibility boundary for CPU-only callers and stored fixtures."""

    raise UnsupportedEnvironmentError(
        "verified replay preparation is complete; GPU worker execution is required"
    )
