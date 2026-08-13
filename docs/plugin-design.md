# `ResidualRMSNorm` plugin design

The custom TensorRT plugin computes residual addition followed by RMS normalization and gamma scaling.
It exists to exercise dynamic-shape, serialization, tactic, CUDA correctness, sanitizer, and profiler workflows in one bounded component.

## Contract

The plugin implements `IPluginV3`, `IPluginV3OneCore`, `IPluginV3OneBuild`, and `IPluginV3OneRuntime`.
It accepts two equal FP32 or FP16 activation tensors plus a one-dimensional FP32 gamma tensor.
The output type matches the activation type.

Activation rank must be two or three.
The final dimension must be positive, fit in a 32-bit integer, and match the gamma length.

## Serialization

The serialized fields include a version, epsilon, and optional extra workspace bytes.
The timing-cache identity and metadata include the same semantic fields.

Engine reload tests verify that epsilon and output behavior survive serialization.
The G6 seed intentionally fails to restore epsilon and must fail beside a clean control.

## Tactics

`kScalarReference` uses straightforward scalar operations and FP32 accumulation.
It prioritizes readability and correct tail behavior.

`kVectorizedWarp` uses warp reduction, `float4` loads for aligned FP32 inputs, `half2` loads for aligned FP16 inputs, and scalar fallback for tails or misalignment.
FP16 input still accumulates in FP32.

The focused benchmark requires a measured benefit on at least one declared workload and no regression beyond five percent on either required benchmark shape.

## Safety tests

Kernel tests cover FP32, FP16, aligned hidden size 256, tail hidden size 259, and reference agreement.
Plugin smoke tests cover invalid rank, invalid gamma, invalid type, profile transitions, tactic selection, cloning, and serialization fields.

Compute Sanitizer runs `memcheck`, `racecheck`, `initcheck`, and `synccheck` on clean targets.
A quarantined G4 target supplies one bounded out-of-bounds tail for expected memcheck detection.
