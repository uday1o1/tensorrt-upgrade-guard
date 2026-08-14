"""Fail early when Docker cannot accept the selected NVIDIA GPU request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from upgrade_guard.containers.commands import CommandRunner, command_sha256
from upgrade_guard.containers.gpu_runtime import (
    bounded_redacted_output,
    docker_gpu_failure_error,
    observe_nvidia_container_toolkit_version,
)
from upgrade_guard.doctor import run_doctor
from upgrade_guard.errors import ExitCode, InfrastructureError, UpgradeGuardError

SCHEMA_VERSION = "upgradeguard.dev/docker-gpu-preflight/v1"


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _selected_gpu(doctor: Any, gpu_uuid: str) -> Any:
    selected = [gpu for gpu in doctor.gpus if gpu.uuid == gpu_uuid]
    if len(selected) != 1:
        raise InfrastructureError(
            "selected GPU UUID is not uniquely visible during Docker GPU preflight",
            details={"selected_gpu_uuid": gpu_uuid},
        )
    return selected[0]


def _device_request_matches(stdout: str, gpu_uuid: str) -> bool:
    try:
        requests = json.loads(stdout)
    except json.JSONDecodeError:
        return False
    if not isinstance(requests, list):
        return False
    for request in requests:
        if not isinstance(request, dict):
            continue
        device_ids = request.get("DeviceIDs")
        capabilities = request.get("Capabilities")
        if device_ids == [gpu_uuid] and isinstance(capabilities, list):
            return any(isinstance(group, list) and "gpu" in group for group in capabilities)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    runner = CommandRunner()
    doctor = run_doctor(runner)
    toolkit = observe_nvidia_container_toolkit_version(runner)
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selected_gpu_uuid": arguments.gpu,
        "image": arguments.image,
        "docker": doctor.docker.model_dump(mode="json"),
        "toolkit_version_observation": toolkit.model_dump(mode="json"),
    }
    try:
        if doctor.outcome != "supported":
            raise InfrastructureError(
                "host doctor did not pass before Docker GPU preflight",
                details={"issues": [issue.model_dump(mode="json") for issue in doctor.issues]},
            )
        selected = _selected_gpu(doctor, arguments.gpu)
        local = runner.run(
            ("docker", "image", "inspect", "--format", "{{json .}}", arguments.image),
            timeout_seconds=30,
        )
        if local.returncode != 0:
            pull = runner.run(("docker", "pull", arguments.image), timeout_seconds=600)
            if pull.returncode != 0:
                raise InfrastructureError(
                    "Docker could not obtain the pinned GPU preflight image",
                    details={"error": bounded_redacted_output(pull.stderr, pull.stdout)},
                )
        name_hash = hashlib.sha256(arguments.gpu.encode()).hexdigest()[:12]
        container_name = f"upgrade-guard-gpu-preflight-{name_hash}"
        runner.run(
            ("docker", "container", "rm", "--force", container_name),
            timeout_seconds=30,
        )
        create_command = (
            "docker",
            "container",
            "create",
            "--name",
            container_name,
            "--gpus",
            f"device={arguments.gpu}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "/bin/true",
            arguments.image,
        )
        created = runner.run(create_command, timeout_seconds=60)
        if created.returncode != 0:
            raise docker_gpu_failure_error(
                "Docker rejected the selected GPU request",
                stdout=created.stdout,
                stderr=created.stderr,
                toolkit_observation=toolkit,
                details={"command_sha256": command_sha256(create_command)},
            )
        inspected = runner.run(
            (
                "docker",
                "container",
                "inspect",
                "--format",
                "{{json .HostConfig.DeviceRequests}}",
                container_name,
            ),
            timeout_seconds=30,
        )
        if inspected.returncode != 0 or not _device_request_matches(
            inspected.stdout, arguments.gpu
        ):
            raise InfrastructureError(
                "Docker did not retain the exact selected GPU device request",
                details={"command_sha256": command_sha256(create_command)},
            )
        payload = {
            **base,
            "status": "passed",
            "diagnosis_code": None,
            "gpu": {
                "uuid": selected.uuid,
                "driver_version": selected.driver_version,
            },
            "gpu_injection_interface": "docker-gpus",
            "gpu_request_verified": True,
            "command_sha256": command_sha256(create_command),
        }
        exit_code = int(ExitCode.SUCCESS)
    except UpgradeGuardError as error:
        payload = {
            **base,
            "status": (
                "unsupported"
                if error.exit_code == ExitCode.UNSUPPORTED
                else "infrastructure_invalid"
            ),
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details,
        }
        exit_code = int(error.exit_code)
    finally:
        if "container_name" in locals():
            try:
                runner.run(
                    ("docker", "container", "rm", "--force", container_name),
                    timeout_seconds=30,
                )
            except UpgradeGuardError:
                pass
    _write_atomic(arguments.output, payload)
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
