"""Transactional matrix-lock orchestration tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

import upgrade_guard.matrix.lock as lock_module
from tests.factories import FIXED_TIME, digest, resolved_image, supported_doctor, worker_probe
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.containers.gpu_runtime import observe_nvidia_container_toolkit_version
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.doctor import DoctorIssue
from upgrade_guard.contracts.environment import NvidiaContainerToolkitVersionObservation
from upgrade_guard.contracts.matrix import MatrixSpec
from upgrade_guard.errors import (
    InfrastructureError,
    InvalidInputError,
    UnsupportedEnvironmentError,
)
from upgrade_guard.matrix.digest import ResolvedArtifact
from upgrade_guard.matrix.lock import (
    WORKER_BASE_DIGEST_LABEL,
    MatrixLocker,
    _host_inventory,
    _host_observation,
    _parse_matrix,
    _read_matrix,
    _validate_pair,
    lock_json,
)
from upgrade_guard.matrix.probe import ProbeExecution

GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"


class ToolkitRunner:
    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        if command == ("nvidia-container-cli", "--version"):
            return CommandResult(command, 0, "cli-version: 1.18.0\n", "", 0.01)
        return CommandResult(command, 127, "", "missing", 0.01)


class FakeResolver:
    def __init__(self, artifacts: Mapping[str, ResolvedArtifact]) -> None:
        self.artifacts = artifacts
        self.references: list[str] = []

    def resolve_linux_amd64(self, authored_reference: str) -> ResolvedArtifact:
        self.references.append(authored_reference)
        return self.artifacts[authored_reference]


class FakeProber:
    def __init__(self, probes: Mapping[str, ProbeExecution]) -> None:
        self.probes = probes
        self.images: list[str] = []
        self.toolkit_observations: list[NvidiaContainerToolkitVersionObservation] = []

    def run(
        self,
        image: object,
        gpu_uuid: str,
        *,
        toolkit_observation: NvidiaContainerToolkitVersionObservation,
    ) -> ProbeExecution:
        manifest_digest = image.manifest_digest  # type: ignore[attr-defined]
        self.images.append(manifest_digest)
        self.toolkit_observations.append(toolkit_observation)
        assert gpu_uuid == GPU_UUID
        return self.probes[manifest_digest]


class FailSecondProber(FakeProber):
    def run(
        self,
        image: object,
        gpu_uuid: str,
        *,
        toolkit_observation: NvidiaContainerToolkitVersionObservation,
    ) -> ProbeExecution:
        if len(self.images) == 1:
            self.images.append(image.manifest_digest)  # type: ignore[attr-defined]
            raise InfrastructureError("second exact worker could not launch")
        return super().run(
            image,
            gpu_uuid,
            toolkit_observation=toolkit_observation,
        )


def matrix_spec() -> MatrixSpec:
    return MatrixSpec.model_validate(
        {
            "api_version": "upgradeguard.dev/v1alpha1",
            "kind": "EnvironmentMatrix",
            "gpu_uuid": GPU_UUID,
            "environments": [
                {
                    "id": "baseline",
                    "base_image": "registry.example/base:v1",
                    "worker_image": "registry.example/worker:v1",
                },
                {
                    "id": "candidate",
                    "base_image": "registry.example/base:v2",
                    "worker_image": "registry.example/worker:v2",
                },
            ],
        }
    )


def lock_dependencies() -> tuple[FakeResolver, FakeProber]:
    base_one = resolved_image(
        reference="registry.example/base:v1",
        manifest_character="1",
        config_character="a",
    )
    worker_one = resolved_image(
        reference="registry.example/worker:v1",
        manifest_character="2",
        config_character="b",
    )
    base_two = resolved_image(
        reference="registry.example/base:v2",
        manifest_character="3",
        config_character="c",
    )
    worker_two = resolved_image(
        reference="registry.example/worker:v2",
        manifest_character="4",
        config_character="d",
    )
    artifacts = {
        base_one.authored_reference: ResolvedArtifact(base_one, {"config": {}}),
        worker_one.authored_reference: ResolvedArtifact(
            worker_one,
            {"config": {"Labels": {WORKER_BASE_DIGEST_LABEL: base_one.manifest_digest}}},
        ),
        base_two.authored_reference: ResolvedArtifact(base_two, {"config": {}}),
        worker_two.authored_reference: ResolvedArtifact(
            worker_two,
            {"config": {"Labels": {WORKER_BASE_DIGEST_LABEL: base_two.manifest_digest}}},
        ),
    }
    probes = {
        worker_one.manifest_digest: ProbeExecution(
            worker_probe(manifest_digest=worker_one.manifest_digest),
            digest("e"),
            digest("f"),
        ),
        worker_two.manifest_digest: ProbeExecution(
            worker_probe(manifest_digest=worker_two.manifest_digest),
            digest("e"),
            digest("f"),
        ),
    }
    return FakeResolver(artifacts), FakeProber(probes)


def locker() -> MatrixLocker:
    resolver, prober = lock_dependencies()
    return MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )


def test_build_lock_captures_exact_pair_and_self_hash() -> None:
    resolver, prober = lock_dependencies()
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )
    result = subject.build(matrix_spec(), source_sha256=digest("9"))
    assert [environment.id for environment in result.environments] == [
        "baseline",
        "candidate",
    ]
    assert result.gpu_uuid == GPU_UUID
    assert result.lock_sha256 == result.computed_sha256()
    assert all(environment.compatibility.compatible for environment in result.environments)
    assert result.environments[0].base_image.manifest_digest != (
        result.environments[0].worker_image.manifest_digest
    )
    assert len(prober.toolkit_observations) == 2
    assert all(observation.status == "observed" for observation in prober.toolkit_observations)
    host = result.environments[0].host
    assert host.docker_cdi_spec_dirs == ("/etc/cdi", "/var/run/cdi")
    assert [device.id for device in host.docker_discovered_devices] == [
        f"nvidia.com/gpu={GPU_UUID}"
    ]
    assert result.environments[1].host == host


def test_verify_reprobes_complete_lock_and_rejects_tool_drift() -> None:
    expected = locker().build(matrix_spec(), source_sha256=digest("9"))
    resolver, prober = lock_dependencies()
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )
    observed = subject.verify(expected)
    assert observed.gpu_uuid == expected.gpu_uuid
    assert len(prober.images) == 2

    first = next(iter(prober.probes))
    execution = prober.probes[first]
    prober.probes[first] = ProbeExecution(
        execution.probe.model_copy(update={"tensorrt": "drifted"}),
        execution.command_sha256,
        execution.output_sha256,
    )
    with pytest.raises(InfrastructureError, match="differs from the immutable lock") as error:
        subject.verify(expected)
    assert "baseline.probe" in error.value.details["drifted_fields"]


def test_lock_writes_only_complete_valid_json(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        """
api_version: upgradeguard.dev/v1alpha1
kind: EnvironmentMatrix
gpu_uuid: GPU-11111111-1111-1111-1111-111111111111
environments:
  - id: baseline
    base_image: registry.example/base:v1
    worker_image: registry.example/worker:v1
  - id: candidate
    base_image: registry.example/base:v2
    worker_image: registry.example/worker:v2
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "matrix.lock.json"
    result = locker().lock(matrix, output)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["lock_sha256"] == result.lock_sha256
    assert stored["source_matrix_sha256"] == sha256_bytes(matrix.read_bytes())
    assert not list(tmp_path.glob(".matrix.lock.json.*.tmp"))


def test_second_worker_failure_leaves_no_lock_file(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        """
api_version: upgradeguard.dev/v1alpha1
kind: EnvironmentMatrix
gpu_uuid: GPU-11111111-1111-1111-1111-111111111111
environments:
  - id: baseline
    base_image: registry.example/base:v1
    worker_image: registry.example/worker:v1
  - id: candidate
    base_image: registry.example/base:v2
    worker_image: registry.example/worker:v2
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "matrix.lock.json"
    resolver, normal_prober = lock_dependencies()
    prober = FailSecondProber(normal_prober.probes)
    subject = MatrixLocker(
        runner=FallbackToolkitRunner(succeeds=False),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(InfrastructureError, match="second exact worker"):
        subject.lock(matrix, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".matrix.lock.json.*.tmp"))


def test_lock_refuses_to_rewrite_existing_output(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("not used", encoding="utf-8")
    output = tmp_path / "matrix.lock.json"
    output.write_text("user data", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="refusing to overwrite"):
        locker().lock(matrix, output)
    assert output.read_text(encoding="utf-8") == "user data"


def test_worker_base_provenance_mismatch_stops_before_probe() -> None:
    resolver, prober = lock_dependencies()
    artifact = resolver.artifacts["registry.example/worker:v1"]
    resolver.artifacts["registry.example/worker:v1"] = ResolvedArtifact(
        artifact.image,
        {"config": {"Labels": {WORKER_BASE_DIGEST_LABEL: digest("f")}}},
    )
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(InvalidInputError, match="does not declare"):
        subject.build(matrix_spec(), source_sha256=digest("9"))
    assert prober.images == []


def test_unsupported_host_stops_before_registry_resolution() -> None:
    resolver, prober = lock_dependencies()
    doctor = supported_doctor().model_copy(
        update={
            "outcome": "unsupported",
            "issues": (
                DoctorIssue(
                    code="HOST_OS_UNSUPPORTED",
                    category="unsupported",
                    message="wrong host",
                ),
            ),
        }
    )
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=lambda: doctor,
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(UnsupportedEnvironmentError, match="qualification boundary"):
        subject.build(matrix_spec(), source_sha256=digest("9"))
    assert resolver.references == []


def test_missing_required_extended_tool_rejects_pair() -> None:
    resolver, prober = lock_dependencies()
    first_key = next(iter(prober.probes))
    probe_execution = prober.probes[first_key]
    prober.probes[first_key] = ProbeExecution(
        worker_probe(
            manifest_digest=probe_execution.probe.image_manifest_digest,
            compute_sanitizer=False,
        ),
        probe_execution.command_sha256,
        probe_execution.output_sha256,
    )
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(UnsupportedEnvironmentError, match="incompatible"):
        subject.build(matrix_spec(), source_sha256=digest("9"))


def test_infrastructure_preflight_and_missing_selected_gpu_are_distinct() -> None:
    resolver, prober = lock_dependencies()
    infrastructure_doctor = supported_doctor().model_copy(
        update={
            "outcome": "infrastructure_invalid",
            "issues": (
                DoctorIssue(
                    code="DOCKER_UNAVAILABLE",
                    category="infrastructure",
                    message="daemon unavailable",
                ),
            ),
        }
    )
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=lambda: infrastructure_doctor,
    )
    with pytest.raises(InfrastructureError, match="inconclusive"):
        subject.build(matrix_spec(), source_sha256=digest("9"))

    no_gpu = supported_doctor().model_copy(update={"gpus": ()})
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=lambda: no_gpu,
    )
    with pytest.raises(UnsupportedEnvironmentError, match="not uniquely visible"):
        subject.build(matrix_spec(), source_sha256=digest("9"))


def test_worker_driver_must_match_host() -> None:
    resolver, prober = lock_dependencies()
    first_key = next(iter(prober.probes))
    execution = prober.probes[first_key]
    prober.probes[first_key] = ProbeExecution(
        worker_probe(
            manifest_digest=execution.probe.image_manifest_digest,
            driver="581.0",
        ),
        execution.command_sha256,
        execution.output_sha256,
    )
    subject = MatrixLocker(
        runner=ToolkitRunner(),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
    )
    with pytest.raises(InfrastructureError, match="different NVIDIA driver"):
        subject.build(matrix_spec(), source_sha256=digest("9"))


def test_matrix_read_and_parse_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="could not read"):
        _read_matrix(tmp_path / "missing.yaml")
    with pytest.raises(InvalidInputError, match="strict schema"):
        _parse_matrix(b"not: [valid")
    with pytest.raises(InvalidInputError, match="strict schema"):
        _parse_matrix(b"unknown: true\n")


class FallbackToolkitRunner:
    def __init__(self, *, succeeds: bool) -> None:
        self.succeeds = succeeds

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        if command == ("nvidia-ctk", "--version") and self.succeeds:
            return CommandResult(command, 0, "NVIDIA Container Toolkit 1.18\n", "", 0.01)
        return CommandResult(command, 127, "", "missing", 0.01)


class ExactToolkitRunner:
    def __init__(self, target: tuple[str, ...], output: str) -> None:
        self.target = target
        self.output = output

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        if command == self.target:
            return CommandResult(command, 0, self.output, "", 0.01)
        if command == ("nvidia-container-cli", "--version"):
            raise InfrastructureError("probe timed out")
        return CommandResult(command, 127, "", "missing", 0.01)


def test_host_observation_falls_back_to_nvidia_ctk() -> None:
    doctor = supported_doctor().model_copy(
        update={"docker": supported_doctor().docker.model_copy(update={"runtimes": ("runc",)})}
    )
    inventory = _host_inventory(doctor, FallbackToolkitRunner(succeeds=True))
    assert inventory.docker_runtime_inventory == ("runc",)
    assert inventory.docker_cdi_spec_dirs == ("/etc/cdi", "/var/run/cdi")
    assert [device.source for device in inventory.docker_discovered_devices] == ["cdi"]
    assert inventory.toolkit_version.status == "observed"
    assert inventory.toolkit_version.source == "nvidia-ctk"
    observation = _host_observation(inventory)
    assert observation.docker_runtime_inventory == ("runc",)
    assert observation.docker_cdi_spec_dirs == ("/etc/cdi", "/var/run/cdi")
    assert observation.docker_discovered_devices == inventory.docker_discovered_devices
    assert observation.gpu_injection_verified is True


def test_host_inventory_keeps_unavailable_toolkit_version_truthfully() -> None:
    doctor = supported_doctor().model_copy(
        update={"docker": supported_doctor().docker.model_copy(update={"runtimes": ("runc",)})}
    )
    inventory = _host_inventory(doctor, FallbackToolkitRunner(succeeds=False))
    assert inventory.docker_runtime_inventory == ("runc",)
    assert inventory.toolkit_version.status == "unavailable"
    assert inventory.toolkit_version.version is None
    assert inventory.toolkit_version.source is None
    assert len(inventory.toolkit_version.attempts) == 9
    assert {attempt.source for attempt in inventory.toolkit_version.attempts} == {
        "nvidia-container-cli",
        "nvidia-ctk",
        "nvidia-container-runtime",
        "dpkg",
        "rpm",
    }


@pytest.mark.parametrize(
    ("command", "output"),
    [
        (
            ("nvidia-container-runtime", "--version"),
            "NVIDIA Container Runtime version 1.18.0\n",
        ),
        (
            (
                "dpkg-query",
                "--show",
                "--showformat=${binary:Package}=${Version}\\n",
                "nvidia-container-toolkit",
            ),
            "nvidia-container-toolkit=1.18.0-1\n",
        ),
        (
            (
                "dpkg-query",
                "--show",
                "--showformat=${binary:Package}=${Version}\\n",
                "libnvidia-container1",
            ),
            "libnvidia-container1=1.18.0-1\n",
        ),
        (
            (
                "rpm",
                "--query",
                "--queryformat",
                "%{NAME}=%{VERSION}-%{RELEASE}\\n",
                "nvidia-container-toolkit",
            ),
            "nvidia-container-toolkit=1.18.0-1\n",
        ),
    ],
)
def test_toolkit_version_uses_runtime_and_package_metadata_fallbacks(
    command: tuple[str, ...], output: str
) -> None:
    observation = observe_nvidia_container_toolkit_version(ExactToolkitRunner(command, output))
    assert observation.status == "observed"
    assert observation.version == output.strip()
    assert observation.attempts[-1].outcome == "observed"


def test_toolkit_observation_rejects_inconsistent_states() -> None:
    attempt = observe_nvidia_container_toolkit_version(ToolkitRunner()).attempts[0]
    with pytest.raises(ValidationError, match="require a version and source"):
        NvidiaContainerToolkitVersionObservation(
            status="observed",
            version=None,
            source=None,
            attempts=(attempt,),
        )
    with pytest.raises(ValidationError, match="cannot contain observed details"):
        NvidiaContainerToolkitVersionObservation(
            status="unavailable",
            version="invented",
            source="nvidia-container-cli",
            attempts=(attempt,),
        )


def test_host_is_finalized_only_after_both_exact_worker_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, prober = lock_dependencies()
    finalized_after: list[int] = []
    original = lock_module._host_observation

    def observe(inventory: object) -> object:
        finalized_after.append(len(prober.images))
        return original(inventory)  # type: ignore[arg-type]

    monkeypatch.setattr(lock_module, "_host_observation", observe)
    subject = MatrixLocker(
        runner=FallbackToolkitRunner(succeeds=False),
        resolver=resolver,
        prober=prober,
        doctor=supported_doctor,
        clock=lambda: FIXED_TIME,
    )
    result = subject.build(matrix_spec(), source_sha256=digest("9"))
    assert finalized_after == [2]
    assert result.environments[0].host.nvidia_container_toolkit_version.status == "unavailable"


def test_pair_validation_rejects_gpu_driver_and_host_drift() -> None:
    result = locker().build(matrix_spec(), source_sha256=digest("9"))
    first, second = result.environments
    changed_gpu = second.model_copy(
        update={
            "probe": second.probe.model_copy(
                update={"gpu": second.probe.gpu.model_copy(update={"name": "Different GPU"})}
            )
        }
    )
    with pytest.raises(InfrastructureError, match="GPU properties"):
        _validate_pair([first, changed_gpu])

    changed_driver = second.model_copy(
        update={"probe": second.probe.model_copy(update={"observed_driver": "999.0"})}
    )
    with pytest.raises(InfrastructureError, match="host drivers"):
        _validate_pair([first, changed_driver])

    changed_host = second.model_copy(
        update={"host": second.host.model_copy(update={"kernel": "different"})}
    )
    with pytest.raises(InfrastructureError, match="one host lock"):
        _validate_pair([first, changed_host])
    assert json.loads(lock_json(result))["lock_sha256"] == result.lock_sha256
