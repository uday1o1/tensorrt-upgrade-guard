# How to run a remote qualification

**Goal**: Run every hardware-dependent gate with one resumable command.

**Prerequisites**: Use a clean Linux x86-64 checkout with Docker, `nvidia-smi`, working Docker GPU injection, access to the locked NGC images, and at least 20 GiB free in both the workspace and Docker-volume storage.

An unprivileged host may not expose an NVIDIA Container Toolkit binary or package record.
That version observation may remain unavailable only when both exact immutable workers pass the real `docker run --gpus device=<UUID>` probe.

## Select the repository

```bash
cd ~/tensorrt-upgrade-guard
```

The runner refuses tracked or untracked source-tree changes because the evidence must name one exact Git commit.

## Run the workflow

```bash
bash scripts/run_cuda_pm_qualification.sh
```

The command creates `.upgrade-guard/cuda-pm/runs/<commit>` for logs, builds, run data, reports, profiles, SBOMs, and completion markers.
Generated corpora are published once under `.upgrade-guard/corpora/by-id/<kind>/<materializer-hash>` and referenced by a source-run corpus index.
Each source commit has an isolated state directory, so a resumed run cannot mix artifacts from different revisions.
One process-wide file lock prevents two qualification invocations from racing the registry, state reconciliation, or cleanup.

## Resume after a failure

Fix the reported external prerequisite or implementation defect, keep the same source revision when appropriate, and run the same command again.

```bash
bash scripts/run_cuda_pm_qualification.sh
```

The runner skips only a step whose versioned marker, complete file inventory, selected GPU, mode, source commit, direct dependency markers, matrix lock, and corpus identities all verify.
Invalid or incomplete owned outputs move under the source run's `stale/` lineage before the step reruns.
Successful G2 and G7 clean replays have independent markers, so an interruption after G2 does not repeat it.

The local failure diagnostic under `diagnostics/` reports the failed step, stable classification, safe log pointers, and the same resume command without copying log contents or environment variables.

## Select a GPU

```bash
UG_GPU_INDEX=1 bash scripts/run_cuda_pm_qualification.sh
```

CI also sets `UG_EXPECTED_GPU_UUID` so a scheduler or topology change cannot silently select a different device.

## Stop after one step

```bash
UG_THROUGH_STEP=profiles bash scripts/run_cuda_pm_qualification.sh
```

Valid step names appear at the bottom of the runner script.
The full workflow probes a real Nsight Compute counter immediately after the CUDA plugin build.
If the host restricts performance counters, it stops before the long qualification gates with `NSIGHT_COMPUTE_COUNTER_PERMISSION_UNAVAILABLE` and the administrator prerequisite.

## Run the trusted smoke path

```bash
UG_SMOKE_ONLY=1 bash scripts/run_cuda_pm_qualification.sh
```

The smoke path locks the environment, materializes the frozen corpora, compiles both workers, runs CTest, builds a standard candidate engine, executes two transformer shapes, and executes the plugin tail shape 20 times.

## Verify completion

The final success line names `.upgrade-guard/cuda-pm/runs/<commit>/evidence.json`.
Inspect its `source_git_commit`, `gpu_uuid`, `matrix_lock_sha256`, `environment_images`, `gate_status`, and artifact hashes before publishing a result.

An infrastructure failure, timeout, unsupported tool, or inconclusive statistical result is not a passing qualification.

The project-owned local registry container is removed at terminal cleanup.
Its source-specific Docker volume is retained because the matrix lock and reproduction evidence refer to worker manifests stored there.
Remove that volume only after the evidence has been reviewed or the source run has been intentionally abandoned.
