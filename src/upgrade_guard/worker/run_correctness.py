"""Execute one trusted TensorRT engine with explicit CUDA buffers."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_guard.contracts.results import WorkerCorrectnessResult
from upgrade_guard.worker.common import (
    command_evidence,
    process_memory_evidence,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)


class InputIntegrityError(RuntimeError):
    """A named device input changed during one correctness repetition."""

    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


def _checked(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {int(error)}")
    return tuple(values)


def _cuda_runtime() -> Any:
    try:
        from cuda.bindings import runtime  # type: ignore[import-not-found]
    except ImportError:
        from cuda import cudart as runtime  # type: ignore[import-not-found]
    return runtime


def run_engine(arguments: argparse.Namespace) -> dict[str, Any]:
    """Deserialize only a same-run trusted engine and retain every repetition hash."""

    import tensorrt as trt  # type: ignore[import-not-found]

    tactic_diagnostic_path = arguments.result.parent / "tactic-diagnostics.jsonl"
    if arguments.plugin:
        tactic_diagnostic_path.unlink(missing_ok=True)
        os.environ["UPGRADE_GUARD_TACTIC_DIAGNOSTIC"] = str(tactic_diagnostic_path)
    for plugin in arguments.plugin:
        try:
            ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise RuntimeError(f"plugin load failed: {plugin}: {error}") from error
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(arguments.engine.read_bytes())
    if engine is None:
        raise RuntimeError("trusted same-run engine deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT execution-context creation failed")
    input_paths = _parse_inputs(arguments.input)
    inputs = {name: np.load(path, allow_pickle=False) for name, path in input_paths.items()}
    tensor_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    input_names = [
        name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
    ]
    if set(inputs) != set(input_names):
        raise RuntimeError(f"input names differ: expected={input_names}, observed={sorted(inputs)}")
    for name in input_names:
        expected_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        if inputs[name].dtype != expected_dtype:
            observed_dtype = inputs[name].dtype
            raise RuntimeError(
                f"input {name} dtype differs: expected={expected_dtype}, observed={observed_dtype}"
            )
        if not context.set_input_shape(name, inputs[name].shape):
            raise RuntimeError(f"input shape was rejected for {name}: {inputs[name].shape}")
    unresolved = tuple(context.infer_shapes())
    if unresolved:
        raise RuntimeError(f"not every dynamic tensor was specified: {unresolved}")
    cudart = _cuda_runtime()
    (stream,) = _checked(cudart.cudaStreamCreate(), "cudaStreamCreate")
    allocations: list[int] = []
    host_outputs: dict[str, np.ndarray[Any, Any]] = {}
    output_pointers: dict[str, int] = {}
    input_pointers: dict[str, int] = {}
    contiguous_inputs: dict[str, np.ndarray[Any, Any]] = {}
    input_value_sha256: dict[str, str] = {}
    io_device_allocation_bytes = 0
    try:
        for name in input_names:
            array = np.ascontiguousarray(inputs[name])
            contiguous_inputs[name] = array
            input_value_sha256[name] = sha256_bytes(array.tobytes(order="C"))
            (pointer,) = _checked(cudart.cudaMalloc(array.nbytes), f"cudaMalloc({name})")
            pointer_value = int(pointer)
            input_pointers[name] = pointer_value
            allocations.append(pointer_value)
            io_device_allocation_bytes += array.nbytes
            if not context.set_tensor_address(name, pointer_value):
                raise RuntimeError(f"TensorRT rejected input address for {name}")
        for name in output_names:
            shape = tuple(int(item) for item in context.get_tensor_shape(name))
            if any(item < 0 for item in shape):
                raise RuntimeError(f"unresolved output shape for {name}: {shape}")
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            array = np.empty(shape, dtype=dtype)
            (pointer,) = _checked(cudart.cudaMalloc(array.nbytes), f"cudaMalloc({name})")
            pointer_value = int(pointer)
            allocations.append(pointer_value)
            io_device_allocation_bytes += array.nbytes
            host_outputs[name] = array
            output_pointers[name] = pointer_value
            if not context.set_tensor_address(name, pointer_value):
                raise RuntimeError(f"TensorRT rejected output address for {name}")
        repetitions: list[dict[str, Any]] = []
        for repetition in range(arguments.repetitions):
            for name, array in contiguous_inputs.items():
                _checked(
                    cudart.cudaMemcpyAsync(
                        input_pointers[name],
                        array.ctypes.data,
                        array.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    ),
                    f"cudaMemcpyAsync H2D({name})",
                )
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            for name, array in host_outputs.items():
                _checked(
                    cudart.cudaMemcpyAsync(
                        array.ctypes.data,
                        output_pointers[name],
                        array.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    f"cudaMemcpyAsync D2H({name})",
                )
            verified_inputs: dict[str, np.ndarray[Any, Any]] = {}
            for name, array in contiguous_inputs.items():
                verified = np.empty_like(array)
                verified_inputs[name] = verified
                _checked(
                    cudart.cudaMemcpyAsync(
                        verified.ctypes.data,
                        input_pointers[name],
                        verified.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    f"cudaMemcpyAsync D2H input verification({name})",
                )
            _checked(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            repetition_inputs = []
            for name in input_names:
                device_sha256 = sha256_bytes(verified_inputs[name].tobytes(order="C"))
                stable = device_sha256 == input_value_sha256[name]
                repetition_inputs.append(
                    {
                        "name": name,
                        "source_sha256": sha256_file(input_paths[name]),
                        "host_value_sha256": input_value_sha256[name],
                        "device_value_sha256": device_sha256,
                        "stable": stable,
                    }
                )
                if not stable:
                    raise InputIntegrityError(
                        f"device input changed during repetition {repetition}: {name}",
                        {"repetition": repetition, "inputs": repetition_inputs},
                    )
            repetition_outputs = []
            for name, array in host_outputs.items():
                if not np.all(np.isfinite(array)):
                    raise RuntimeError(f"output {name} contains nonfinite values")
                output_path = arguments.output / f"{name}.repetition-{repetition:02d}.npy"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, array, allow_pickle=False)
                repetition_outputs.append(
                    {
                        "name": name,
                        "path": str(output_path),
                        "sha256": sha256_file(output_path),
                        "bytes": output_path.stat().st_size,
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                    }
                )
            repetitions.append(
                {"index": repetition, "inputs": repetition_inputs, "outputs": repetition_outputs}
            )
        engine_sha256 = sha256_file(arguments.engine)
        tactic_diagnostic = (
            _tactic_evidence(
                tactic_diagnostic_path,
                engine_sha256,
                contiguous_inputs,
                expected_enqueue_count=arguments.repetitions,
            )
            if arguments.plugin
            else None
        )
        return {
            "schema_version": "upgradeguard.dev/worker-correctness/v1",
            "status": "passed",
            "engine_sha256": engine_sha256,
            "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
            "repetitions": repetitions,
            "input_integrity_stable": True,
            "tactic_diagnostic": tactic_diagnostic,
            "memory_diagnostics": {
                "execution_context_device_memory_bytes": int(
                    context.update_device_memory_size_for_shapes()
                    if hasattr(context, "update_device_memory_size_for_shapes")
                    else (
                        engine.device_memory_size_v2
                        if hasattr(engine, "device_memory_size_v2")
                        else engine.device_memory_size
                    )
                ),
                "io_device_allocation_bytes": io_device_allocation_bytes,
                "process": process_memory_evidence(),
            },
            "tensorrt_version": trt.__version__,
        }
    finally:
        for pointer in reversed(allocations):
            _checked(cudart.cudaFree(pointer), "cudaFree")
        _checked(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in parsed:
            raise ValueError(f"invalid or duplicate --input: {value}")
        parsed[name] = Path(path)
    return parsed


def _tactic_evidence(
    path: Path,
    engine_sha256: str,
    inputs: dict[str, np.ndarray[Any, Any]],
    *,
    expected_enqueue_count: int,
) -> dict[str, object]:
    """Bind the plugin's runtime-selected tactic to this engine and shape."""

    if not path.is_file() or path.is_symlink():
        raise RuntimeError("plugin execution emitted no selected-tactic diagnostic")
    try:
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("plugin selected-tactic diagnostic is invalid") from error
    enqueue = [record for record in records if record.get("event") == "enqueue"]
    if not enqueue:
        raise RuntimeError("plugin selected-tactic diagnostic has no enqueue event")
    if len(enqueue) != expected_enqueue_count:
        raise RuntimeError("plugin selected-tactic enqueue count differs from repetitions")
    tactics = {record.get("tactic") for record in enqueue}
    try:
        shapes = {
            (int(record.get("rows", -1)), int(record.get("hidden", -1))) for record in enqueue
        }
    except (TypeError, ValueError) as error:
        raise RuntimeError("plugin selected-tactic diagnostic shape is invalid") from error
    activations = [value for value in inputs.values() if value.ndim in (2, 3)]
    if not activations:
        raise RuntimeError("plugin tactic evidence has no activation input")
    activation = activations[0]
    expected_shape = (int(np.prod(activation.shape[:-1])), int(activation.shape[-1]))
    if len(tactics) != 1 or tactics - {"kSCALAR_REFERENCE", "kVECTORIZED_WARP"}:
        raise RuntimeError("plugin selected more than one or an unknown tactic")
    if shapes != {expected_shape}:
        raise RuntimeError("plugin selected-tactic shape differs from executed activation")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "engine_sha256": engine_sha256,
        "selected_tactic": next(iter(tactics)),
        "rows": expected_shape[0],
        "hidden": expected_shape[1],
        "enqueue_count": len(enqueue),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--plugin", type=Path, action="append", default=[])
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    started = time.time()
    try:
        result = run_engine(arguments)
    except Exception as error:
        from upgrade_guard.classify import classify_worker_error

        ended = time.time()
        failure: dict[str, object] = {
            "schema_version": "upgradeguard.dev/worker-correctness/v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "failure_code": classify_worker_error("correctness", str(error)).value,
            "started_unix_seconds": started,
            "ended_unix_seconds": ended,
            "duration_seconds": ended - started,
        }
        if isinstance(error, InputIntegrityError):
            failure["input_integrity_stable"] = False
            failure["input_integrity_evidence"] = error.evidence
        failure.update(command_evidence("upgrade_guard.worker.run_correctness", sys.argv[1:]))
        typed_failure = WorkerCorrectnessResult.model_validate(failure)
        write_json_atomic(arguments.result, typed_failure.model_dump(mode="json"))
        raise
    ended = time.time()
    result.update(
        {
            **command_evidence("upgrade_guard.worker.run_correctness", sys.argv[1:]),
            "started_unix_seconds": started,
            "ended_unix_seconds": ended,
            "duration_seconds": ended - started,
        }
    )
    typed_result = WorkerCorrectnessResult.model_validate(result)
    write_json_atomic(arguments.result, typed_result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
