# How to collect focused CUDA profiles

**Goal**: Capture diagnostic evidence for the optimized `ResidualRMSNorm` tactic without contaminating the primary benchmark.

**Prerequisites**: Complete the unprofiled plugin benchmark on the selected GPU first.

## Run the automated profile step

Run the full resumable qualification or resume through the profile step.

```bash
UG_THROUGH_STEP=profiles bash scripts/run_cuda_pm_qualification.sh
```

The runner records `nsys --version` and `ncu --version` inside the exact candidate worker.

## Nsight Systems capture

The benchmark marks only the optimized tactic with the NVTX range `upgrade_guard/residual_rmsnorm_optimized`.
Nsight Systems starts capture at that range and traces CUDA plus NVTX activity without CPU sampling.

The runner retains the `.nsys-rep` file, exports SQLite, and writes a small CUDA kernel summary CSV.

## Nsight Compute capture

Nsight Compute filters to one `residualRmsNormFloat4` launch.
It collects Speed of Light, Memory Workload Analysis, Launch Statistics, and Occupancy sections.

The runner retains the `.ncu-rep` file and a CSV detail export.

## Verify the boundary

Confirm that the primary benchmark JSON contains `"profiled": false`.
Confirm that profiler files appear only under the candidate plugin evidence directory.

Profiler evidence explains the measured behavior.
It does not replace, amend, or pass the unprofiled performance gate.
