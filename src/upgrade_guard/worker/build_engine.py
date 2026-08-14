"""Build and reload one strongly typed TensorRT engine inside a worker."""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path
from typing import Any

from upgrade_guard.worker.common import (
    load_json,
    process_memory_evidence,
    sha256_file,
    write_json_atomic,
)


def build_engine(arguments: argparse.Namespace) -> dict[str, Any]:
    """Build, serialize, reload, inspect, and describe one trusted engine."""

    import tensorrt as trt  # type: ignore[import-not-found]

    started = time.time()
    for plugin in arguments.plugin:
        ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    logger = _capturing_logger(trt, verbose=arguments.verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(_strongly_typed_network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    model_bytes = arguments.model.read_bytes()
    parsed = parser.parse(model_bytes)
    parser_errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    if not parsed or parser_errors:
        raise RuntimeError("ONNX parser rejected model: " + " | ".join(parser_errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, arguments.workspace_bytes)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = arguments.optimization_level
    profile_specification = load_json(arguments.profile)
    profile = builder.create_optimization_profile()
    for name, shapes in profile_specification.items():
        accepted = profile.set_shape(
            name, tuple(shapes["min"]), tuple(shapes["opt"]), tuple(shapes["max"])
        )
        if not accepted:
            raise RuntimeError(f"optimization profile rejected input {name}")
    profile_index = config.add_optimization_profile(profile)
    if profile_index < 0:
        raise RuntimeError("TensorRT rejected optimization profile")
    timing_cache_input_sha256 = None
    observed_cache_state = "warm" if arguments.timing_cache.is_file() else "cold"
    if arguments.cache_state != "auto" and arguments.cache_state != observed_cache_state:
        raise RuntimeError(
            f"timing cache state differs: expected={arguments.cache_state}, "
            f"observed={observed_cache_state}"
        )
    if arguments.timing_cache.is_file():
        timing_cache_input_sha256 = sha256_file(arguments.timing_cache)
        cache = config.create_timing_cache(arguments.timing_cache.read_bytes())
        if not config.set_timing_cache(cache, False):
            raise RuntimeError("TensorRT rejected environment-local timing cache")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned no serialized engine")
    arguments.engine.parent.mkdir(parents=True, exist_ok=True)
    arguments.engine.write_bytes(bytes(serialized))
    cache = config.get_timing_cache()
    if cache is not None:
        arguments.timing_cache.parent.mkdir(parents=True, exist_ok=True)
        arguments.timing_cache.write_bytes(bytes(cache.serialize()))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(arguments.engine.read_bytes())
    if engine is None:
        raise RuntimeError("freshly built engine failed same-environment reload")
    inspector = engine.create_engine_inspector()
    inspector_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    arguments.inspector.parent.mkdir(parents=True, exist_ok=True)
    arguments.inspector.write_text(inspector_text + "\n", encoding="utf-8")
    device_memory = (
        engine.device_memory_size_v2
        if hasattr(engine, "device_memory_size_v2")
        else engine.device_memory_size
    )
    ended = time.time()
    return {
        "schema_version": "upgradeguard.dev/worker-build/v1",
        "status": "passed",
        "model": {"path": str(arguments.model), "sha256": sha256_file(arguments.model)},
        "engine": {
            "path": str(arguments.engine),
            "sha256": sha256_file(arguments.engine),
            "bytes": arguments.engine.stat().st_size,
            "device_memory_bytes": int(device_memory),
        },
        "memory_diagnostics": {
            "engine_reported_device_memory_bytes": int(device_memory),
            "execution_context_allocation_bytes": None,
            "builder": process_memory_evidence(),
        },
        "inspector": {
            "path": str(arguments.inspector),
            "sha256": sha256_file(arguments.inspector),
        },
        "timing_cache": {
            "path": str(arguments.timing_cache),
            "input_sha256": timing_cache_input_sha256,
            "output_sha256": sha256_file(arguments.timing_cache),
        },
        "parser_errors": parser_errors,
        "builder_messages": logger.messages,
        "builder_warnings": [
            message for message in logger.messages if message["severity"] == "WARNING"
        ],
        "timing_cache_state": observed_cache_state,
        "tensorrt_version": trt.__version__,
        "started_unix_seconds": started,
        "ended_unix_seconds": ended,
        "duration_seconds": ended - started,
        "strongly_typed": True,
    }


def _strongly_typed_network_flags(trt: Any) -> int:
    """Require the strong-typing API before parsing or building a network."""

    flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if flag is None:
        raise RuntimeError("TensorRT does not expose STRONGLY_TYPED network creation")
    return 1 << int(flag)


def _capturing_logger(trt: Any, *, verbose: bool) -> Any:
    """Create the documented ILogger subclass and retain bounded structured messages."""

    class CapturingLogger(trt.ILogger):  # type: ignore[misc]
        def __init__(self) -> None:
            trt.ILogger.__init__(self)
            self.messages: list[dict[str, str]] = []

        def log(self, severity: Any, message: str) -> None:
            warning_or_higher = severity in {
                trt.ILogger.INTERNAL_ERROR,
                trt.ILogger.ERROR,
                trt.ILogger.WARNING,
            }
            if verbose or warning_or_higher:
                name = next(
                    (
                        authored
                        for authored in (
                            "INTERNAL_ERROR",
                            "ERROR",
                            "WARNING",
                            "INFO",
                            "VERBOSE",
                        )
                        if severity == getattr(trt.ILogger, authored)
                    ),
                    str(severity),
                )
                self.messages.append({"severity": name, "message": str(message)[:4000]})

    return CapturingLogger()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--timing-cache", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, action="append", default=[])
    parser.add_argument("--workspace-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--optimization-level", type=int, choices=range(6), default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-state", choices=("auto", "cold", "warm"), default="auto")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        result = build_engine(arguments)
    except Exception as error:
        write_json_atomic(
            arguments.result,
            {
                "schema_version": "upgradeguard.dev/worker-build/v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    write_json_atomic(arguments.result, result)


if __name__ == "__main__":
    main()
