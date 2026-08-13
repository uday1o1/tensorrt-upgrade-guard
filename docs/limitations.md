# Limitations

This page defines the current repository scope rather than limitations of NVIDIA TensorRT or CUDA.

## Evidence scope

No GPU, numerical, performance, sanitizer, or profiler pass is public until the corresponding hash-addressed evidence from a real run is reviewed.
Local macOS tests verify the host control plane and simulated orchestration only.

Every measured result applies to the complete locked baseline and candidate stacks on one selected GPU.
The result does not isolate TensorRT as the sole cause of a change.

## Workload scope

V1 covers the frozen mini transformer, dynamic MobileNetV3 Small 0.75, and project-owned `ResidualRMSNorm` micrograph.
It does not claim coverage for arbitrary ONNX graphs, plugins, precision modes, or production traffic distributions.

Q/DQ transformer qualification remains excluded until the locked CPU reference provider proves support for the exact artifact.

## Platform scope

The complete workflow targets Linux x86-64 with Docker and NVIDIA Container Toolkit.
Jetson, Windows, Triton Server, Dynamo, Kubernetes, and multi-GPU execution are outside V1.

## Reproduction scope

V1 reproduces from ONNX, frozen inputs, environment locks, and reviewed plugin source.
TensorRT API Capture and Replay remains post-V1 because the project has no C++ builder adapter.

## Statistical scope

The paired bootstrap supports a decision for the declared shapes and policy.
It does not predict every latency percentile, concurrency level, power mode, or deployment environment.
