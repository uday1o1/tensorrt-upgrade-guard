"""Execute one trusted TensorRT engine with explicit CUDA buffers."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_guard.worker.common import sha256_file, write_json_atomic


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

    for plugin in arguments.plugin:
        ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
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
    try:
        for name in input_names:
            array = np.ascontiguousarray(inputs[name])
            (pointer,) = _checked(cudart.cudaMalloc(array.nbytes), f"cudaMalloc({name})")
            pointer_value = int(pointer)
            allocations.append(pointer_value)
            _checked(
                cudart.cudaMemcpyAsync(
                    pointer_value,
                    array.ctypes.data,
                    array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                f"cudaMemcpyAsync H2D({name})",
            )
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
            host_outputs[name] = array
            output_pointers[name] = pointer_value
            if not context.set_tensor_address(name, pointer_value):
                raise RuntimeError(f"TensorRT rejected output address for {name}")
        repetitions: list[dict[str, Any]] = []
        for repetition in range(arguments.repetitions):
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
            _checked(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
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
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                    }
                )
            repetitions.append({"index": repetition, "outputs": repetition_outputs})
        return {
            "schema_version": "upgradeguard.dev/worker-correctness/v1",
            "status": "passed",
            "engine_sha256": sha256_file(arguments.engine),
            "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
            "repetitions": repetitions,
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
    try:
        result = run_engine(arguments)
    except Exception as error:
        write_json_atomic(
            arguments.result,
            {
                "schema_version": "upgradeguard.dev/worker-correctness/v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    write_json_atomic(arguments.result, result)


if __name__ == "__main__":
    main()
