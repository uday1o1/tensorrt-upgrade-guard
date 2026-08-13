"""CPU tests for the self-contained in-container probe."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import pytest

from tests.factories import digest
from upgrade_guard.matrix import worker_probe


def test_run_and_file_hash_use_argument_arrays(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"artifact")
    returncode, output = worker_probe._run((sys.executable, "-c", "print('worker-output')"))
    assert returncode == 0
    assert output == "worker-output"
    assert worker_probe._sha256_file(str(path)).startswith("sha256:")


def test_run_handles_missing_command_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    returncode, output = worker_probe._run(("definitely-not-a-command",))
    assert (returncode, output) == (127, "")

    def timeout(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise worker_probe.subprocess.TimeoutExpired(("tool",), 1)

    monkeypatch.setattr(worker_probe.subprocess, "run", timeout)
    assert worker_probe._run(("tool",)) == (127, "")


def test_tool_and_trtexec_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_probe.shutil, "which", lambda command: None)
    assert worker_probe._tool("missing") == {"available": False}

    monkeypatch.setattr(worker_probe.shutil, "which", lambda command: sys.executable)
    tool = worker_probe._tool("python", ("--version",))
    assert tool["available"]
    assert tool["path"] == str(Path(sys.executable).resolve())

    monkeypatch.setattr(
        worker_probe,
        "_tool",
        lambda command, version_args=("--version",): {
            "available": True,
            "path": "/usr/bin/trtexec",
            "version": "TensorRT 11.2",
            "sha256": digest("a"),
        },
    )
    monkeypatch.setattr(
        worker_probe,
        "_run",
        lambda args: (0, "usage: trtexec --onnx=model --shapes=input:1x8"),
    )
    trtexec = worker_probe._trtexec()
    assert trtexec["options"] == ["onnx", "shapes"]
    assert trtexec["help_sha256"].startswith("sha256:")


def test_trtexec_fails_closed_without_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker_probe,
        "_tool",
        lambda command: {"available": True, "path": "/bin/trtexec"},
    )
    monkeypatch.setattr(worker_probe, "_run", lambda args: (1, ""))
    assert worker_probe._trtexec() == {"available": False}


def test_package_version_uses_distribution_and_module_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert worker_probe._package_version("pydantic")

    def missing(distribution: str) -> str:
        raise worker_probe.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(worker_probe.importlib.metadata, "version", missing)
    monkeypatch.setattr(
        worker_probe.importlib,
        "import_module",
        lambda module: type("Module", (), {"__version__": "9.9"})(),
    )
    assert worker_probe._package_version("missing", "fallback") == "9.9"
    assert worker_probe._package_version("missing") == ""


def test_cuda_runtime_and_toolkit_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCudaRuntime:
        @staticmethod
        def cudaRuntimeGetVersion(pointer: object) -> int:  # noqa: N802
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 12090
            return 0

    monkeypatch.setattr(worker_probe.ctypes, "CDLL", lambda candidate: FakeCudaRuntime())
    assert worker_probe._cuda_runtime_version() == "12.9"
    assert (
        worker_probe._cuda_toolkit_version(
            {"available": True, "version": "Cuda compilation tools, release 13.0, V13.0"}
        )
        == "13.0"
    )
    assert worker_probe._cuda_toolkit_version({"available": False}) == ""


def test_gpu_parser_requires_exactly_one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker_probe,
        "_run",
        lambda args: (
            0,
            (
                "NVIDIA RTX Test, GPU-11111111-1111-1111-1111-111111111111, "
                "8.9, 24576, 95.00, 580.80.01, 300.0"
            ),
        ),
    )
    gpu, driver = worker_probe._gpu()
    assert gpu["vram_mib"] == 24576
    assert driver == "580.80.01"
    monkeypatch.setattr(worker_probe, "_run", lambda args: (0, ""))
    with pytest.raises(RuntimeError, match="exactly one"):
        worker_probe._gpu()


def test_headers_find_existing_files(tmp_path: Path) -> None:
    header = tmp_path / "NvInfer.h"
    header.write_text("header", encoding="utf-8")
    assert worker_probe._headers((str(header),)) == [str(header)]


def test_collect_probe_requires_manifest_and_builds_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UG_WORKER_MANIFEST_DIGEST", digest("1"))
    monkeypatch.setattr(
        worker_probe,
        "_gpu",
        lambda: (
            {
                "name": "GPU",
                "uuid": "GPU-11111111-1111-1111-1111-111111111111",
                "compute_capability": "8.9",
                "vram_mib": 24576,
                "vbios_version": "95.00",
                "power_limit_watts": 300.0,
            },
            "580.80.01",
        ),
    )
    monkeypatch.setattr(
        worker_probe,
        "_tool",
        lambda command, version_args=("--version",): {"available": False},
    )
    monkeypatch.setattr(worker_probe, "_trtexec", lambda: {"available": False})
    monkeypatch.setattr(worker_probe, "_package_version", lambda distribution, module=None: "1.0")
    monkeypatch.setattr(worker_probe, "_cuda_runtime_version", lambda: "13.0")
    monkeypatch.setattr(worker_probe, "_cuda_toolkit_version", lambda compiler: "13.0")
    monkeypatch.setattr(worker_probe, "_headers", lambda patterns: ["/header"])
    monkeypatch.setattr(worker_probe, "_operating_system", lambda: "Ubuntu")
    result = worker_probe.collect_probe()
    assert result["schema_version"] == "upgradeguard.dev/worker-probe/v1"
    assert result["image_manifest_digest"] == digest("1")
    assert result["observed_driver"] == "580.80.01"


def test_collect_probe_and_main_fail_closed_on_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("UG_WORKER_MANIFEST_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="must identify"):
        worker_probe.collect_probe()
    assert worker_probe.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert "UG_WORKER_MANIFEST_DIGEST" in payload["error"]


def test_main_prints_success_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(worker_probe, "collect_probe", lambda: {"ok": True})
    assert worker_probe.main() == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
