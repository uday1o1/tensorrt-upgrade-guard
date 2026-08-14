# Benchmarking methodology

UpgradeGuard measures the candidate relative to the baseline with adjacent, unprofiled blocks on one selected GPU.
The protocol aims to distinguish a practical regression from ordinary shared-system and thermal variation.

## Primary measurement

Each shape uses one inference stream, disabled CUDA Graphs, disabled host-to-device transfers during the timed section, 500 milliseconds of warmup, and one second of measurement.
The `trtexec` adapter selects `--infStreams=1` or the legacy `--streams=1` only when the locked help inventory supports it.

## Paired blocks

The runner uses a precommitted seeded and balanced randomized order with ten observations in each order for a 20-pair focused benchmark.
It accepts at least 20 adjacent pairs per required shape.
It retries rejected blocks up to the authored bound but never converts an invalid block into passing evidence.

Before and after each block, the host records temperature, graphics and memory clocks, power draw, power limit, utilization, and competing compute processes.
The block is rejected when those observations violate the locked validity policy.

## Statistical decision

For each accepted pair, the implementation computes the log of the candidate-to-baseline median ratio.
A seeded paired bootstrap generates two-sided and one-sided confidence bounds.

The gate passes when the one-sided upper bound is within the practical allowance.
It reports a regression when the one-sided lower bound exceeds the allowance.
Every other statistically valid result is inconclusive.

The weighted workload result resamples each shape independently and combines log ratios using authored positive weights that sum to one.

## False-positive pilot

The A/A pilot runs the baseline against itself for 20 accepted pairs under the same stream and validity rules.
The checked-in `scripts/validate_aa.py` rejects an unstable or policy-violating pilot.

## Memory and build evidence

Three independent engine builds record engine bytes, engine-reported device memory, build duration, cold or warm timing-cache state, inspector output, warnings, builder host RSS, and coarse builder GPU-process observations.
Correctness runs separately record execution-context device memory, I/O device allocation, host RSS, and coarse GPU-process observations.
Runtime memory, engine file size, engine-reported device memory, execution-context memory, coarse GPU memory, and host memory remain separately labeled fields.

## Profiling boundary

The full diagnostic Nsight Systems and Nsight Compute runs start only after the unprofiled benchmark.
Profiler measurements never update the primary qualification result.
An early exact-worker preflight validates required CLI options and section names and runs one bounded Nsight Compute capability probe.
The probe collects one `SpeedOfLight` section for the selected kernel solely to prove protected-counter permission before expensive qualification.
Its report is labeled capability-only and is excluded from benchmark and diagnostic-profile claims.
The focused post-benchmark capture remains the only profiling evidence used to explain measured behavior.
