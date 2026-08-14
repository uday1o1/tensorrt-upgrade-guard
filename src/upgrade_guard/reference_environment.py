"""Immutable independent CPU reference-environment locking."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from upgrade_guard.containers.commands import CommandRunner, Runner, command_sha256
from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock
from upgrade_guard.errors import InfrastructureError, InvalidInputError
from upgrade_guard.matrix.digest import (
    RegistryClient,
    credentials_from_environment,
    parse_image_reference,
)


class ReferenceEnvironmentLocker:
    """Resolve and probe an exact non-TensorRT CPU reference image."""

    def __init__(
        self,
        runner: Runner | None = None,
        *,
        clock: Any | None = None,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.clock = clock or (lambda: datetime.now(UTC))

    def lock(self, image_reference: str, output: Path) -> ReferenceEnvironmentLock:
        """Probe and atomically publish one strict reference lock."""

        if output.exists() or output.is_symlink():
            raise InvalidInputError("refusing to overwrite reference environment lock")
        parts = parse_image_reference(image_reference)
        credentials = credentials_from_environment(parts.registry)
        image = RegistryClient(credentials=credentials).resolve_linux_amd64(image_reference).image
        canonical = image.canonical_reference
        pull = self.runner.run(
            ("docker", "pull", "--platform", "linux/amd64", canonical),
            timeout_seconds=1800,
        )
        if pull.returncode != 0:
            raise InfrastructureError("could not pull the independent reference image")
        container_name = f"upgrade-guard-reference-{uuid.uuid4().hex[:12]}"
        user_id = os.getuid()
        group_id = os.getgid()
        command = (
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--platform",
            "linux/amd64",
            "--user",
            f"{user_id}:{group_id}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",  # noqa: S108
            "--entrypoint",
            "",
            canonical,
            "python3",
            "-c",
            _PROBE,
        )
        try:
            result = self.runner.run(command, timeout_seconds=300)
            if result.returncode != 0:
                raise InfrastructureError("independent reference image probe failed")
            payload: Any = json.loads(result.stdout)
            lock = _lock_from_probe(
                payload,
                image=image,
                command=command,
                output=result.stdout,
                probed_at=self.clock(),
            )
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as error:
            self._cleanup(container_name)
            raise InfrastructureError(
                "independent reference probe returned invalid evidence"
            ) from error
        except BaseException:
            self._cleanup(container_name)
            raise
        _write_atomic(output, lock.model_dump_json(indent=2) + "\n")
        return lock

    def _cleanup(self, container_name: str) -> None:
        try:
            self.runner.run(
                ("docker", "container", "rm", "--force", container_name),
                timeout_seconds=30,
            )
        except Exception:
            return


_PROBE = """import json,platform,sys
import numpy,onnx,onnxruntime
providers=onnxruntime.get_available_providers()
assert providers == ['CPUExecutionProvider'] or 'CPUExecutionProvider' in providers
print(json.dumps({
    'schema_version':'upgradeguard.dev/reference-probe/v1',
    'operating_system':platform.platform(),
    'architecture':platform.machine(),
    'python':sys.version.split()[0],
    'onnx':onnx.__version__,
    'onnxruntime':onnxruntime.__version__,
    'numpy':numpy.__version__,
    'providers':providers,
    'intra_op_threads':1,
    'inter_op_threads':1,
},sort_keys=True))
"""


def _lock_from_probe(
    payload: Any,
    *,
    image: Any,
    command: tuple[str, ...],
    output: str,
    probed_at: datetime,
) -> ReferenceEnvironmentLock:
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "upgradeguard.dev/reference-probe/v1"
    ):
        raise ValueError("reference probe schema differs")
    providers = payload["providers"]
    if not isinstance(providers, list) or "CPUExecutionProvider" not in providers:
        raise ValueError("CPUExecutionProvider is unavailable")
    zero = "sha256:" + ("0" * 64)
    lock = ReferenceEnvironmentLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="ReferenceEnvironmentLock",
        id="onnxruntime-cpu-reference",
        image=image,
        operating_system=str(payload["operating_system"]),
        architecture=str(payload["architecture"]),
        python=str(payload["python"]),
        onnx=str(payload["onnx"]),
        onnxruntime=str(payload["onnxruntime"]),
        execution_provider="CPUExecutionProvider",
        provider_options={
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_DISABLE_ALL",
        },
        numpy=str(payload["numpy"]),
        pytorch=None,
        intra_op_threads=int(payload["intra_op_threads"]),
        inter_op_threads=int(payload["inter_op_threads"]),
        probe_command_sha256=command_sha256(command),
        probe_output_sha256=sha256_bytes(output.encode()),
        probed_at=probed_at,
        lock_sha256=zero,
    )
    return lock.model_copy(update={"lock_sha256": lock.computed_sha256()})


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise InfrastructureError("could not publish reference environment lock") from error
