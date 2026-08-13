# CUDA optimization narrative

The `ResidualRMSNorm` optimization starts from a readable scalar reference and changes one execution tactic without changing the mathematical contract.

## Baseline tactic

The scalar path loads each activation and residual value, accumulates squared sums in FP32, normalizes with epsilon, applies FP32 gamma, and writes the output type declared by TensorRT.
It handles every positive hidden size in the declared dynamic profile.

## Optimized tactic

The optimized path assigns one warp to a row and reduces squared sums with shuffle operations.
It uses `float4` only for aligned FP32 addresses with a hidden size divisible by four.
It uses `half2` only for aligned FP16 addresses with an even hidden size.
Every other case uses the scalar warp path, including hidden size 259.

FP16 activation values accumulate in FP32.
No tactic assumes that a dynamic hidden size is always aligned.

## Measurement gate

The focused unprofiled benchmark compares scalar and optimized kernels at hidden sizes 256 and 259 with 4096 rows.
It requires a measured benefit on at least one case and rejects more than five percent regression on either case.

No measured outcome is published in this repository yet.
The candidate GPU run must produce `plugin-benchmark.json` with `"profiled": false` before this narrative can include a result.

## Diagnostic gate

After the unprofiled result, Nsight Systems captures only the NVTX-marked optimized range.
Nsight Compute then collects four focused section groups for one `residualRmsNormFloat4` launch.

The final narrative can discuss achieved occupancy, memory behavior, and launch shape only when the retained `.ncu-rep`, `.nsys-rep`, CSV exports, source hash, and tool versions support those claims.
