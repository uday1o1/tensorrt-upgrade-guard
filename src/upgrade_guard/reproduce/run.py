"""Trust-gated typed reproduction preparation and execution."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from csv import reader as csv_reader
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from upgrade_guard.containers.commands import CommandRunner, Runner, command_sha256
from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.containers.security import validate_locked_image
from upgrade_guard.contracts.base import StrictModel, sha256_bytes, sha256_file
from upgrade_guard.contracts.bundle import (
    CudaArchitectureBuild,
    LocalWorkerBuild,
    ReplayRequirements,
    SourceBuildRequest,
    canonical_cmake_cuda_architecture,
    is_cmake_configure_command,
    validate_cmake_cuda_command,
)
from upgrade_guard.contracts.common import ArtifactReference
from upgrade_guard.contracts.environment import EnvironmentLock, Sha256Digest
from upgrade_guard.contracts.matrix import GpuUuid
from upgrade_guard.errors import (
    FailureCode,
    InfrastructureError,
    InvalidInputError,
    UnsupportedEnvironmentError,
)
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
    expected_failure_code: FailureCode | None = None
    failure_code_source: Literal["result_file", "stdout"] | None = None

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
        if (self.expected_failure_code is None) != (self.failure_code_source is None):
            raise ValueError(
                "expected_failure_code and failure_code_source must be authored together"
            )
        if self.failure_code_source == "result_file" and self.result_file is None:
            raise ValueError("result_file failure-code evidence requires result_file")
        if self.expected_result_status == "failed" and self.expected_failure_code is None:
            raise ValueError("failed replay results require an expected failure code")
        if self.expected_result_status == "passed" and self.expected_failure_code is not None:
            raise ValueError("passing replay results cannot expect a failure code")
        if self.result_file is not None:
            _safe_relative(self.result_file)
        return self


class ReplayRecipe(StrictModel):
    """Hash-verified command sequence stored inside a source-bearing bundle."""

    schema_version: Literal["upgradeguard.dev/replay-recipe/v1"]
    expected_failure_code: FailureCode
    steps: tuple[ReplayStep, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_unique_steps(self) -> ReplayRecipe:
        identifiers = [step.id for step in self.steps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("replay step identifiers must be unique")
        observed = tuple(
            step.expected_failure_code
            for step in self.steps
            if step.expected_failure_code is not None
        )
        if observed != (self.expected_failure_code,):
            raise ValueError(
                "replay recipe requires exactly one step for its expected failure code"
            )
        return self


class ReplayTarget(StrictModel):
    """Observed replay GPU selected independently from original qualification provenance."""

    gpu_uuid: GpuUuid
    platform: Literal["linux/amd64"] = "linux/amd64"
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    driver_version: str = Field(min_length=1, max_length=128)
    vram_mib: int = Field(gt=0)


class ReplayTargetReview(StrictModel):
    """Selected replay GPU and its result against the bundled requirements."""

    selected_gpu: ReplayTarget | None
    requirements: ReplayRequirements
    status: Literal["passed", "failed", "not_evaluated"]
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_consistent_status(self) -> ReplayTargetReview:
        if self.selected_gpu is None:
            if self.status != "not_evaluated" or self.reasons:
                raise ValueError("an unselected replay GPU cannot have compatibility evidence")
        elif self.status == "not_evaluated":
            raise ValueError("a selected replay GPU requires compatibility evidence")
        elif (self.status == "passed") == bool(self.reasons):
            raise ValueError("replay GPU compatibility status and reasons differ")
        return self


class ReplayReviewInventory(StrictModel):
    """Complete source-build inventory shown before explicit code trust."""

    schema_version: Literal["upgradeguard.dev/replay-source-review/v1"]
    bundle_id: str
    bundle_manifest_sha256: Sha256Digest
    sources: tuple[ArtifactReference, ...]
    original_worker_image_manifest_digest: Sha256Digest
    original_gpu_uuid: GpuUuid
    cuda_architecture: CudaArchitectureBuild | None
    worker_rebuild_recipe: LocalWorkerBuild
    worker_rebuild_recipe_sha256: Sha256Digest
    build_command: tuple[str, ...]
    build_command_sha256: Sha256Digest
    target_compatibility: ReplayTargetReview

    @model_validator(mode="after")
    def validate_review_identity(self) -> ReplayReviewInventory:
        paths = [artifact.path for artifact in self.sources]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("source review paths must be nonempty, unique, and sorted")
        if self.worker_rebuild_recipe_sha256 != self.worker_rebuild_recipe.computed_sha256():
            raise ValueError("source review rebuild recipe hash differs")
        if self.build_command_sha256 != command_sha256(self.build_command):
            raise ValueError("source review build command hash differs")
        return self


def observe_replay_target(
    gpu_uuid: str | None,
    *,
    runner: Runner | None = None,
) -> ReplayTarget:
    """Observe one replay GPU instead of trusting operator-entered compatibility facts."""

    executor = runner or CommandRunner()
    platform = executor.run(
        ("docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"),
        timeout_seconds=30,
    )
    if platform.returncode != 0:
        raise InfrastructureError("Docker platform could not be observed for replay")
    observed_platform = platform.stdout.strip()
    if observed_platform not in {"linux/amd64", "linux/x86_64"}:
        raise UnsupportedEnvironmentError(
            "V1 replay requires a linux/amd64 Docker server",
            details={"observed_platform": observed_platform},
        )
    query = executor.run(
        (
            "nvidia-smi",
            "--query-gpu=uuid,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        timeout_seconds=30,
    )
    if query.returncode != 0:
        raise InfrastructureError("replay GPU properties could not be observed")
    rows = tuple(csv_reader(query.stdout.splitlines()))
    parsed: list[ReplayTarget] = []
    for row in rows:
        if len(row) != 4:
            raise InfrastructureError("nvidia-smi replay GPU output is malformed")
        uuid, capability, memory, driver = (item.strip() for item in row)
        try:
            memory_decimal = Decimal(memory)
        except InvalidOperation as error:
            raise InfrastructureError("nvidia-smi replay VRAM output is malformed") from error
        if memory_decimal <= 0 or memory_decimal != memory_decimal.to_integral_value():
            raise InfrastructureError("nvidia-smi replay VRAM output is malformed")
        try:
            parsed.append(
                ReplayTarget(
                    gpu_uuid=uuid,
                    compute_capability=capability,
                    driver_version=driver,
                    vram_mib=int(memory_decimal),
                )
            )
        except ValidationError as error:
            raise InfrastructureError("nvidia-smi replay GPU output is invalid") from error
    selected = [target for target in parsed if gpu_uuid is None or target.gpu_uuid == gpu_uuid]
    if not selected:
        raise InvalidInputError("the selected replay GPU UUID is not visible")
    if len(selected) != 1:
        raise InvalidInputError("multiple replay GPUs are visible; pass --gpu with one UUID")
    return selected[0]


class ReplayImageBuilder(Protocol):
    """Build a local immutable worker from the verified bundle recipe."""

    def build(
        self,
        *,
        bundle_root: Path,
        request: SourceBuildRequest,
        timeout_seconds: int,
    ) -> RebuiltWorkerImage: ...


class RebuiltWorkerImage(StrictModel):
    """Immutable local build output bound to the reviewed bundle recipe."""

    canonical_reference: str
    recipe_sha256: Sha256Digest
    build_log_sha256: Sha256Digest
    build_log: str


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Verified typed replay plan that never invokes bundled scripts."""

    bundle_id: str
    original_gpu_uuid: str | None
    selected_replay_gpu_uuid: str | None
    original_worker_image_manifest_digest: str | None
    base_image: str | None
    worker_rebuild_recipe_sha256: str | None
    replay_requirements: ReplayRequirements | None
    cuda_architecture: CudaArchitectureBuild | None
    build_commands: tuple[tuple[str, ...], ...]
    worker_build_arguments: tuple[tuple[str, str], ...]
    source_paths: tuple[str, ...]
    included_engine_trusted: bool
    review_inventory: ReplayReviewInventory | None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Successful clean replay evidence emitted by the public CLI."""

    schema_version: str
    status: str
    bundle_id: str
    bundle_manifest_sha256: str
    worker_image: str
    worker_rebuild_recipe_sha256: str
    worker_build_log_sha256: str
    worker_build_log: ArtifactReference
    original_gpu_uuid: str
    selected_gpu_uuid: str
    expected_failure_code: FailureCode
    observed_failure_code: FailureCode
    step_results: tuple[str, ...]


def prepare_replay(
    source: Path,
    *,
    trust_source_code: bool,
    trust_included_engine: bool,
    replay_target: ReplayTarget | None = None,
) -> ReplayPlan:
    """Verify identity and require explicit trust for executable artifacts."""

    verified = verify_bundle(source)
    manifest = verified.manifest
    source_build = manifest.source_build
    review_inventory = (
        _source_review_inventory(
            manifest.id,
            manifest.manifest_sha256,
            source_build,
            replay_target,
        )
        if source_build is not None
        else None
    )
    if verified.source_code_present and not trust_source_code:
        assert review_inventory is not None
        raise UnsupportedEnvironmentError(
            "bundle contains source code; pass --trust-source-code after review",
            details={"review_inventory": review_inventory.model_dump(mode="json")},
        )
    if verified.engine_present and not trust_included_engine:
        raise UnsupportedEnvironmentError(
            "bundle contains a serialized engine; pass --trust-included-engine after review",
            details={"engine": manifest.included_engine.path if manifest.included_engine else None},
        )
    if source_build is not None and replay_target is not None:
        _validate_replay_target(
            replay_target,
            source_build.replay_requirements,
            source_build.cuda_architecture,
        )
    return ReplayPlan(
        bundle_id=manifest.id,
        original_gpu_uuid=source_build.original_gpu_uuid if source_build else None,
        selected_replay_gpu_uuid=replay_target.gpu_uuid if replay_target else None,
        original_worker_image_manifest_digest=(
            source_build.original_worker_image_manifest_digest if source_build else None
        ),
        base_image=source_build.local_worker_build.base_image if source_build else None,
        worker_rebuild_recipe_sha256=(
            source_build.local_worker_build.computed_sha256() if source_build else None
        ),
        replay_requirements=source_build.replay_requirements if source_build else None,
        cuda_architecture=source_build.cuda_architecture if source_build else None,
        build_commands=((source_build.command,) if source_build else ()),
        worker_build_arguments=(
            tuple(
                (argument.name, argument.value)
                for argument in source_build.local_worker_build.build_arguments
            )
            if source_build
            else ()
        ),
        source_paths=(
            tuple(sorted(artifact.path for artifact in source_build.sources))
            if source_build
            else ()
        ),
        included_engine_trusted=verified.engine_present and trust_included_engine,
        review_inventory=review_inventory,
    )


def execute_replay(
    source: Path,
    output: Path,
    *,
    trust_source_code: bool,
    trust_included_engine: bool,
    worker: DockerGpuWorker | None = None,
    replay_target: ReplayTarget | None = None,
    image_builder: ReplayImageBuilder | None = None,
) -> ReplayResult:
    """Materialize a verified bundle and execute only its typed recipe."""

    plan = prepare_replay(
        source,
        trust_source_code=trust_source_code,
        trust_included_engine=trust_included_engine,
        replay_target=replay_target,
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
        if replay_target is None:
            raise UnsupportedEnvironmentError(
                "source replay requires an explicitly selected compatible replay GPU"
            )
        if image_builder is None:
            raise UnsupportedEnvironmentError("source replay requires a local worker image builder")
        try:
            environment = EnvironmentLock.model_validate_json(
                (bundle_root / manifest.candidate_environment.path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValidationError) as error:
            raise InvalidInputError("candidate environment lock is invalid") from error
        if (
            source_build.original_worker_image_manifest_digest
            != environment.worker_image.manifest_digest
        ):
            raise InvalidInputError("original replay worker provenance differs from environment")
        if source_build.original_gpu_uuid != environment.probe.gpu.uuid:
            raise InvalidInputError("original replay GPU provenance differs from environment")
        if (
            source_build.cuda_architecture is not None
            and source_build.cuda_architecture.original_compute_capability
            != environment.probe.gpu.compute_capability
        ):
            raise InvalidInputError("replay CUDA architecture differs from candidate environment")
        if (
            source_build.local_worker_build.base_image_manifest_digest
            != environment.base_image.manifest_digest
        ):
            raise InvalidInputError("replay base image differs from candidate environment")
        _validate_replay_target(
            replay_target,
            source_build.replay_requirements,
            source_build.cuda_architecture,
        )
        recipe_path = bundle_root / "commands" / "replay.json"
        if "commands/replay.json" not in {item.path for item in manifest.files}:
            raise InvalidInputError("bundle has no hash-verified typed replay recipe")
        try:
            recipe = ReplayRecipe.model_validate_json(recipe_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError) as error:
            raise InvalidInputError("typed replay recipe is invalid") from error
        if recipe.expected_failure_code is not manifest.expected_failure.code:
            raise InvalidInputError("replay recipe failure code differs from bundle manifest")
        if recipe.steps[0].command != source_build.command:
            raise InvalidInputError(
                "first replay step differs from the reviewed source build command"
            )
        _validate_recipe_cuda_architecture(recipe, source_build.cuda_architecture)

        work = staging / "work"
        logs = staging / "steps"
        work.mkdir()
        logs.mkdir()
        rebuilt = image_builder.build(
            bundle_root=bundle_root,
            request=source_build,
            timeout_seconds=1800,
        )
        recipe_sha256 = source_build.local_worker_build.computed_sha256()
        if rebuilt.recipe_sha256 != recipe_sha256:
            raise InvalidInputError("rebuilt worker evidence differs from bundle recipe")
        if sha256_bytes(rebuilt.build_log.encode("utf-8")) != rebuilt.build_log_sha256:
            raise InvalidInputError("rebuilt worker log differs from its declared identity")
        rebuild_logs = staging / "logs"
        rebuild_logs.mkdir()
        rebuild_log_path = rebuild_logs / "worker-build.log"
        rebuild_log_path.write_text(rebuilt.build_log, encoding="utf-8")
        rebuild_log = ArtifactReference(
            path="logs/worker-build.log",
            sha256=sha256_file(rebuild_log_path),
            bytes=rebuild_log_path.stat().st_size,
            media_type="text/plain",
        )
        replay_image = validate_locked_image(rebuilt.canonical_reference)
        gpu_worker = worker or DockerGpuWorker()
        completed: list[str] = []
        observed_failure_codes: list[FailureCode] = []
        for step in recipe.steps:
            result = gpu_worker.run(
                image=replay_image,
                gpu_uuid=replay_target.gpu_uuid,
                mounts=WorkerMounts(source=bundle_root, corpus=bundle_root, output=work),
                command=step.command,
                timeout_seconds=step.timeout_seconds,
                accepted_returncodes=step.accepted_returncodes,
            )
            observed_failure_code = _validate_step_evidence(step, result.stdout, work)
            if observed_failure_code is not None:
                observed_failure_codes.append(observed_failure_code)
            record = {
                "schema_version": "upgradeguard.dev/replay-step/v1",
                "status": "passed",
                "id": step.id,
                "command": list(step.command),
                "accepted_returncodes": list(step.accepted_returncodes),
                "observed_returncode": result.returncode,
                "expected_failure_code": (
                    step.expected_failure_code.value
                    if step.expected_failure_code is not None
                    else None
                ),
                "observed_failure_code": (
                    observed_failure_code.value if observed_failure_code is not None else None
                ),
                "duration_seconds": result.duration_seconds,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            (logs / f"{step.id}.json").write_text(
                json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            completed.append(step.id)
        if observed_failure_codes != [recipe.expected_failure_code]:
            raise InvalidInputError(
                "replay did not observe exactly its declared typed failure code"
            )
        replay = ReplayResult(
            schema_version="upgradeguard.dev/replay-result/v1",
            status="passed",
            bundle_id=plan.bundle_id,
            bundle_manifest_sha256=manifest.manifest_sha256,
            worker_image=replay_image,
            worker_rebuild_recipe_sha256=recipe_sha256,
            worker_build_log_sha256=rebuilt.build_log_sha256,
            worker_build_log=rebuild_log,
            original_gpu_uuid=source_build.original_gpu_uuid,
            selected_gpu_uuid=replay_target.gpu_uuid,
            expected_failure_code=recipe.expected_failure_code,
            observed_failure_code=observed_failure_codes[0],
            step_results=tuple(completed),
        )
        replay_value = asdict(replay)
        replay_value["worker_build_log"] = rebuild_log.model_dump(mode="json")
        (staging / "replay-result.json").write_text(
            json.dumps(replay_value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
        return replay
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _source_review_inventory(
    bundle_id: str,
    bundle_manifest_sha256: str,
    source_build: SourceBuildRequest,
    replay_target: ReplayTarget | None,
) -> ReplayReviewInventory:
    sources = tuple(sorted(source_build.sources, key=lambda artifact: artifact.path))
    recipe = source_build.local_worker_build
    return ReplayReviewInventory(
        schema_version="upgradeguard.dev/replay-source-review/v1",
        bundle_id=bundle_id,
        bundle_manifest_sha256=bundle_manifest_sha256,
        sources=sources,
        original_worker_image_manifest_digest=(source_build.original_worker_image_manifest_digest),
        original_gpu_uuid=source_build.original_gpu_uuid,
        cuda_architecture=source_build.cuda_architecture,
        worker_rebuild_recipe=recipe,
        worker_rebuild_recipe_sha256=recipe.computed_sha256(),
        build_command=source_build.command,
        build_command_sha256=command_sha256(source_build.command),
        target_compatibility=_replay_target_review(
            replay_target,
            source_build.replay_requirements,
            source_build.cuda_architecture,
        ),
    )


def _replay_target_review(
    target: ReplayTarget | None,
    requirements: ReplayRequirements,
    cuda_architecture: CudaArchitectureBuild | None,
) -> ReplayTargetReview:
    if target is None:
        return ReplayTargetReview(
            selected_gpu=None,
            requirements=requirements,
            status="not_evaluated",
        )
    reasons = _replay_target_compatibility_reasons(target, requirements, cuda_architecture)
    return ReplayTargetReview(
        selected_gpu=target,
        requirements=requirements,
        status="failed" if reasons else "passed",
        reasons=reasons,
    )


def _validate_replay_target(
    target: ReplayTarget,
    requirements: ReplayRequirements,
    cuda_architecture: CudaArchitectureBuild | None,
) -> None:
    reasons = _replay_target_compatibility_reasons(target, requirements, cuda_architecture)
    if reasons:
        raise UnsupportedEnvironmentError(
            "selected replay GPU does not satisfy bundle requirements",
            details={"reasons": list(reasons), "selected_gpu_uuid": target.gpu_uuid},
        )


def _replay_target_compatibility_reasons(
    target: ReplayTarget,
    requirements: ReplayRequirements,
    cuda_architecture: CudaArchitectureBuild | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if target.platform != requirements.platform:
        reasons.append(f"platform {target.platform} differs from {requirements.platform}")
    if _version_tuple(target.compute_capability) < _version_tuple(
        requirements.minimum_compute_capability
    ):
        reasons.append(
            f"compute capability {target.compute_capability} is below "
            f"{requirements.minimum_compute_capability}"
        )
    if _version_tuple(target.driver_version) < _version_tuple(requirements.minimum_driver):
        reasons.append(f"driver {target.driver_version} is below {requirements.minimum_driver}")
    if target.vram_mib < requirements.minimum_vram_mib:
        reasons.append(f"VRAM {target.vram_mib} MiB is below {requirements.minimum_vram_mib} MiB")
    if cuda_architecture is not None:
        try:
            target_architecture = canonical_cmake_cuda_architecture(target.compute_capability)
        except ValueError:
            reasons.append(f"compute capability {target.compute_capability} is not canonical")
        else:
            if target_architecture != cuda_architecture.cmake_cuda_architecture:
                reasons.append(
                    f"CUDA architecture {target_architecture} differs from locked "
                    f"{cuda_architecture.cmake_cuda_architecture}"
                )
    return tuple(reasons)


def _validate_recipe_cuda_architecture(
    recipe: ReplayRecipe, architecture: CudaArchitectureBuild | None
) -> None:
    try:
        for step in recipe.steps:
            if is_cmake_configure_command(step.command):
                validate_cmake_cuda_command(step.command, architecture)
    except ValueError as error:
        raise InvalidInputError("replay CMake CUDA architecture is invalid") from error


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = tuple(int(number) for number in re.findall(r"\d+", value))
    return numbers or (0,)


def _validate_step_evidence(step: ReplayStep, stdout: str, work: Path) -> FailureCode | None:
    result_value: dict[str, object] | None = None
    stdout_value: dict[str, object] | None = None
    if step.result_file is not None:
        result_path = work / _safe_relative(step.result_file)
        if not result_path.is_file() or result_path.is_symlink():
            raise InvalidInputError(f"replay step did not produce its result: {step.id}")
        result_value = _json_object(result_path.read_text(encoding="utf-8"), step.id)
        if result_value.get("status") != step.expected_result_status:
            raise InvalidInputError(f"replay step result status differed: {step.id}")
        if step.result_message_contains is not None and step.result_message_contains not in str(
            result_value.get("message", "")
        ):
            raise InvalidInputError(f"replay step failed for a different reason: {step.id}")
        if step.expected_result_status == "passed" and result_value.get("failure_code") is not None:
            raise InvalidInputError(f"passing replay step reported a failure code: {step.id}")
    if step.stdout_json_equals:
        stdout_value = _json_object(stdout, step.id)
        for authored_path, expected in step.stdout_json_equals.items():
            observed: object = stdout_value
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
    if step.expected_failure_code is None:
        return None
    if step.failure_code_source == "result_file":
        assert result_value is not None
        evidence = result_value
    else:
        if stdout_value is None:
            stdout_value = _json_object(stdout, step.id)
        evidence = stdout_value
    raw_failure_code = evidence.get("failure_code")
    try:
        if not isinstance(raw_failure_code, str):
            raise ValueError("failure code is not a string")
        observed_failure_code = FailureCode(raw_failure_code)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(
            f"replay step did not emit a typed failure code: {step.id}"
        ) from error
    if observed_failure_code is not step.expected_failure_code:
        raise InvalidInputError(f"replay step failure code differed: {step.id}")
    return observed_failure_code


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
