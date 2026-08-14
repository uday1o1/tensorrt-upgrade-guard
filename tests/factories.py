"""Typed test-data factories for Milestone 0 contracts."""

from __future__ import annotations

from datetime import UTC, datetime

from upgrade_guard.contracts.common import FailureRecord, Phase, ResultStatus
from upgrade_guard.contracts.doctor import (
    DockerDiscoveredDevice,
    DoctorDocker,
    DoctorGpu,
    DoctorResult,
)
from upgrade_guard.contracts.environment import (
    CompatibilityEvidence,
    EnvironmentLock,
    GpuObservation,
    HostObservation,
    NvidiaContainerToolkitVersionAttempt,
    NvidiaContainerToolkitVersionObservation,
    PlatformIdentity,
    ResolvedImage,
    ToolObservation,
    TrtexecObservation,
    WorkerProbe,
)
from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock
from upgrade_guard.contracts.results import HardwareObservation, RunResult
from upgrade_guard.errors import FailureCode


def digest(character: str) -> str:
    return f"sha256:{character * 64}"


def resolved_image(
    *,
    reference: str = "registry.example/upgrade/worker:v1",
    manifest_character: str = "1",
    config_character: str = "2",
) -> ResolvedImage:
    registry_repository = reference.split("@", maxsplit=1)[0].split(":", maxsplit=1)[0]
    registry, repository = registry_repository.split("/", maxsplit=1)
    return ResolvedImage(
        authored_reference=reference,
        registry=registry,
        repository=repository,
        authored_tag="v1",
        requested_digest=None,
        index_digest=digest("0"),
        manifest_digest=digest(manifest_character),
        config_digest=digest(config_character),
        manifest_media_type="application/vnd.oci.image.manifest.v1+json",
        config_media_type="application/vnd.oci.image.config.v1+json",
        platform=PlatformIdentity(os="linux", architecture="amd64"),
    )


def reference_environment_lock() -> ReferenceEnvironmentLock:
    """Return one self-valid independent CPU reference lock."""

    lock = ReferenceEnvironmentLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ReferenceEnvironmentLock",
        id="onnxruntime-cpu-reference",
        image=resolved_image(
            reference="registry.example/reference:v1",
            manifest_character="7",
            config_character="8",
        ),
        operating_system="Debian GNU/Linux 12",
        architecture="x86_64",
        python="3.12.13",
        onnx="1.22.0",
        onnxruntime="1.28.0",
        execution_provider="CPUExecutionProvider",
        provider_options={
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_DISABLE_ALL",
        },
        numpy="2.4.2",
        pytorch=None,
        intra_op_threads=1,
        inter_op_threads=1,
        probe_command_sha256=digest("9"),
        probe_output_sha256=digest("a"),
        probed_at=FIXED_TIME,
        lock_sha256=digest("0"),
    )
    return lock.model_copy(update={"lock_sha256": lock.computed_sha256()})


def available_tool(
    name: str = "tool",
    version: str = "tool version 1.0",
) -> ToolObservation:
    return ToolObservation(
        available=True,
        path=f"/usr/bin/{name}",
        version=version,
        sha256=digest("a"),
    )


def worker_probe(
    *,
    manifest_digest: str | None = None,
    gpu_uuid: str = "GPU-11111111-1111-1111-1111-111111111111",
    driver: str = "580.80.01",
    cuda_runtime: str = "13.0",
    compute_capability: str = "8.9",
    compute_sanitizer: bool = True,
) -> WorkerProbe:
    tool = available_tool()
    sanitizer = tool if compute_sanitizer else ToolObservation(available=False)
    return WorkerProbe(
        schema_version="upgradeguard.dev/worker-probe/v1",
        image_manifest_digest=manifest_digest or digest("1"),
        gpu=GpuObservation(
            name="NVIDIA RTX Test",
            uuid=gpu_uuid,
            compute_capability=compute_capability,
            vram_mib=24576,
            vbios_version="95.00.00.00.00",
            power_limit_watts=300.0,
        ),
        observed_driver=driver,
        cuda_runtime=cuda_runtime,
        cuda_toolkit="13.0",
        tensorrt="11.2.1",
        python="3.12.10",
        polygraphy="0.49.26",
        onnx="1.18.0",
        onnxruntime="1.22.1",
        operating_system="Ubuntu 24.04",
        kernel="6.8.0",
        trtexec=TrtexecObservation(
            available=True,
            path="/usr/src/tensorrt/bin/trtexec",
            version="TensorRT.trtexec 11.2.1",
            sha256=digest("b"),
            help_sha256=digest("c"),
            options=("--onnx", "--shapes"),
        ),
        compute_sanitizer=sanitizer,
        nsight_systems=tool,
        nsight_compute=tool,
        c_compiler=tool,
        cxx_compiler=tool,
        cuda_compiler=tool,
        cmake=available_tool("cmake", "cmake version 3.31.0"),
        ninja=tool,
        cuda_headers=("/usr/local/cuda/include/cuda_runtime_api.h",),
        tensorrt_headers=("/usr/include/x86_64-linux-gnu/NvInfer.h",),
    )


def supported_doctor(
    gpu_uuid: str = "GPU-11111111-1111-1111-1111-111111111111",
) -> DoctorResult:
    return DoctorResult(
        schema_version="upgradeguard.dev/doctor/v1",
        outcome="supported",
        host_os="Linux",
        host_release="6.8.0",
        host_architecture="x86_64",
        python_version="3.12.10",
        docker=DoctorDocker(
            available=True,
            client_version="29.0.0",
            server_version="29.0.0",
            server_os="linux",
            server_architecture="x86_64",
            runtimes=("nvidia", "runc"),
            cdi_spec_dirs=("/etc/cdi", "/var/run/cdi"),
            discovered_devices=(
                DockerDiscoveredDevice(source="cdi", id=f"nvidia.com/gpu={gpu_uuid}"),
            ),
            context="default",
        ),
        gpus=(
            DoctorGpu(
                name="NVIDIA RTX Test",
                uuid=gpu_uuid,
                compute_capability="8.9",
                vram_mib=24576,
                driver_version="580.80.01",
            ),
        ),
        issues=(),
    )


def environment_lock(
    *,
    environment_id: str = "candidate",
    worker_manifest_character: str = "1",
    gpu_uuid: str = "GPU-11111111-1111-1111-1111-111111111111",
) -> EnvironmentLock:
    """Return one internally consistent immutable test environment."""

    base = resolved_image(
        reference="registry.example/base:v1",
        manifest_character="3",
        config_character="4",
    )
    worker = resolved_image(
        reference="registry.example/worker:v1",
        manifest_character=worker_manifest_character,
        config_character="5",
    )
    return EnvironmentLock(
        id=environment_id,
        base_image=base,
        worker_image=worker,
        declared_base_manifest_digest=base.manifest_digest,
        probe=worker_probe(manifest_digest=worker.manifest_digest, gpu_uuid=gpu_uuid),
        host=HostObservation(
            operating_system="Ubuntu 24.04",
            kernel="6.8.0",
            architecture="x86_64",
            docker_client_version="29.0.0",
            docker_server_version="29.0.0",
            docker_runtime_inventory=("nvidia", "runc"),
            docker_cdi_spec_dirs=("/etc/cdi", "/var/run/cdi"),
            docker_discovered_devices=(
                DockerDiscoveredDevice(source="cdi", id=f"nvidia.com/gpu={gpu_uuid}"),
            ),
            gpu_injection_interface="docker-gpus",
            gpu_injection_verified=True,
            nvidia_container_toolkit_version=NvidiaContainerToolkitVersionObservation(
                status="observed",
                version="1.18.0",
                source="nvidia-container-cli",
                attempts=(
                    NvidiaContainerToolkitVersionAttempt(
                        source="nvidia-container-cli",
                        command=("nvidia-container-cli", "--version"),
                        outcome="observed",
                        returncode=0,
                        detail="1.18.0",
                    ),
                ),
            ),
        ),
        compatibility=CompatibilityEvidence(
            policy_version="test-v1",
            source_urls=("https://docs.nvidia.com/",),
            checked_at=FIXED_TIME,
            minimum_driver="580.0",
            minimum_compute_capability="8.0",
            compatible=True,
            reasons=(),
        ),
        probe_command_sha256=digest("6"),
        probe_output_sha256=digest("7"),
        probed_at=FIXED_TIME,
    )


FIXED_TIME = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def failure_record(
    code: FailureCode = FailureCode.NUMERICAL_REGRESSION,
) -> FailureRecord:
    return FailureRecord(
        code=code,
        phase=Phase.CORRECTNESS,
        environment_id="candidate",
        model_id="fixture-model",
        precision="fp32",
        shape_id="fixture-shape",
        input_fixture_id="fixture-input",
        output_name="output",
        gate="candidate_to_reference",
        observed="maximum absolute error 0.2",
        threshold="atol=0.001, rtol=0.001",
        evidence=(),
        signature_sha256=digest("7"),
    )


def run_result(
    *,
    status: ResultStatus = ResultStatus.PASSED,
    failure: FailureRecord | None = None,
) -> RunResult:
    if status is not ResultStatus.PASSED and failure is None:
        code = {
            ResultStatus.UNSUPPORTED: FailureCode.PREFLIGHT_UNSUPPORTED,
            ResultStatus.INCONCLUSIVE: FailureCode.INCONCLUSIVE,
            ResultStatus.INFRASTRUCTURE_INVALID: FailureCode.INFRASTRUCTURE_INVALID,
            ResultStatus.FAILED: FailureCode.NUMERICAL_REGRESSION,
        }[status]
        failure = failure_record(code)
    return RunResult(
        api_version="upgradeguard.dev/v1alpha1",
        kind="RunResult",
        id=f"fixture-{status.value}",
        case_manifest_sha256=digest("1"),
        build_manifest_sha256=digest("2"),
        environment_lock_sha256=digest("3"),
        hardware_sha256=digest("4"),
        command=("worker", "run"),
        command_sha256=digest("5"),
        output_schema=(),
        output_artifacts=(),
        numerical=(),
        determinism=None,
        timing_blocks=(),
        memory=None,
        hardware=HardwareObservation(
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            driver="580.80.01",
            environment_lock_sha256=digest("3"),
            valid=True,
            invalid_reasons=(),
        ),
        started_at=FIXED_TIME,
        ended_at=FIXED_TIME,
        status=status,
        failure=failure,
        logs=(),
        warnings=(),
        diagnostics=(),
    )
