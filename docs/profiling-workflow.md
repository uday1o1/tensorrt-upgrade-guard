# How to collect focused CUDA profiles

**Goal**: Capture diagnostic evidence for the optimized `ResidualRMSNorm` tactic without contaminating the primary benchmark.

**Prerequisites**: Complete the unprofiled plugin benchmark on the selected GPU first.

## Run the automated profile step

Run the full resumable qualification or resume through the profile step.

```bash
UG_THROUGH_STEP=profiles bash scripts/run_cuda_pm_qualification.sh
```

The runner records `nsys --version` and `ncu --version` inside the exact candidate worker.
Before statistical qualification, it checks the exact pinned tools for every CLI option and Nsight Compute section used later.
It then runs one bounded `SpeedOfLight` collection for the selected kernel to prove protected-counter permission before expensive qualification.
The retained result is labeled capability-only, not a benchmark or diagnostic profile.

## Nsight Systems capture

The benchmark marks only the optimized tactic with the NVTX range `upgrade_guard/residual_rmsnorm_optimized`.
The range message is registered in the `upgrade_guard` domain so the default Nsight Systems capture policy can match it.
Nsight Systems starts capture at that range and traces CUDA plus NVTX activity without CPU sampling.

The runner retains the `.nsys-rep` file, exports SQLite, and writes a small CUDA kernel summary CSV.

## Nsight Compute capture

Nsight Compute filters to one `residualRmsNormFloat4` launch.
It collects Speed of Light, Memory Workload Analysis, Launch Statistics, and Occupancy sections.

The runner retains the `.ncu-rep` file and a CSV detail export.
The CSV must contain the selected kernel and a finite value under a recognized metric-value header.
If protected counters are disabled, the runner preserves bounded stdout and stderr and reports the exact administrator prerequisite.

## Verify the boundary

Confirm that the primary benchmark JSON contains `"profiled": false`.
Confirm that the early capability report appears under `profiler-preflight` and the final diagnostic files appear under `profiles`.

Profiler evidence explains the measured behavior.
It does not replace, amend, or pass the unprofiled performance gate.
