# How to run a remote qualification

**Goal**: Run every hardware-dependent gate with one resumable command.

**Prerequisites**: Use a clean Linux x86-64 checkout with Docker, `nvidia-smi`, working Docker GPU injection, access to the locked NGC images, and sufficient workspace and Docker-volume storage.

The default capacity policy requires 20 GiB and 100,000 free inodes for each storage root.
When both roots share one filesystem, the runner requires the aggregate 40 GiB and 200,000 free inodes instead of counting the same free space twice.

An unprivileged host may not expose an NVIDIA Container Toolkit binary or package record.
That version observation may remain unavailable only when both exact immutable workers pass the real `docker run --gpus device=<UUID>` probe.
If Docker reports that it cannot discover an NVIDIA GPU vendor from CDI and every toolkit source is unavailable, an administrator must install and configure NVIDIA Container Toolkit for the rootful Docker daemon.
The runner classifies that state as `PREFLIGHT_UNSUPPORTED` before it builds workers or starts qualification.

## Select the repository

```bash
cd ~/tensorrt-upgrade-guard
```

The runner refuses tracked or untracked source-tree changes because the evidence must name one exact Git commit.

## Run the workflow

```bash
run_root=".upgrade-guard/qualification/runs/$(git rev-parse HEAD)"
uv run --frozen upgrade-guard qualify qualification/full.yaml \
  --project-root . --out "${run_root}"
```

The public command invokes the checked-in resumable runner without shell interpolation.

The command creates `.upgrade-guard/qualification/runs/<commit>` for logs, builds, run data, reports, profiles, SBOMs, and completion markers.
Generated corpora are published once under `.upgrade-guard/corpora/by-id/<kind>/<materializer-hash>` and referenced by a source-run corpus index.
Each source commit has an isolated state directory, so a resumed run cannot mix artifacts from different revisions.
One process-wide file lock prevents two qualification invocations from racing the registry, state reconciliation, or cleanup.

## Resume after a failure

Fix the reported external prerequisite or implementation defect, keep the same source revision when appropriate, and run the same command again.

```bash
uv run --frozen upgrade-guard qualify qualification/full.yaml \
  --project-root . --out "${run_root}"
```

The runner skips only a step whose versioned marker, complete file inventory, selected GPU, mode, source commit, direct dependency markers, matrix lock, and corpus identities all verify.
Invalid or incomplete owned outputs move under the source run's `stale/` lineage before the step reruns.
Successful G2 and G7 clean replays have independent markers, so an interruption after G2 does not repeat it.

The local failure diagnostic under `diagnostics/` reports the failed step, stable classification, safe log pointers, and the same resume command without copying log contents or environment variables.

When the diagnostic is `NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE`, the missing operation requires an administrator.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

After that change, rerun the same public qualification command.

## Select a GPU

```bash
UG_GPU_INDEX=1 bash scripts/run_gpu_qualification.sh
```

Set `UG_EXPECTED_GPU_UUID` when invoking the lower-level runner directly so a scheduler or topology change cannot silently select a different device.

## Stop after one step

```bash
UG_THROUGH_STEP=profiles bash scripts/run_gpu_qualification.sh
```

Valid step names appear at the bottom of the runner script.
The full workflow verifies exact Nsight CLI options and required section names immediately after the CUDA plugin build.
It then runs one bounded candidate-worker `SpeedOfLight` collection to prove protected-counter permission before expensive qualification.
That capability-only report is excluded from performance and diagnostic-profile claims.
Both the early probe and the focused post-benchmark profile search both Nsight Compute streams for `ERR_NVGPUCTRPERM` and report `NSIGHT_COMPUTE_COUNTER_PERMISSION_UNAVAILABLE` with the administrator prerequisite.

## Run the trusted smoke path

```bash
UG_SMOKE_ONLY=1 bash scripts/run_gpu_qualification.sh
```

The smoke path locks the environment, materializes the frozen corpora, compiles both workers, runs CTest, builds a standard candidate engine, executes two transformer shapes, and executes the plugin tail shape 20 times.

Full mode additionally runs a bounded readiness gate in both exact workers before A/A.
That gate builds, reloads, and executes the standard transformer, the FP32 plugin tail case, and one dynamic MobileNet case against frozen CPU references.
Compute Sanitizer controls and worker SBOM validation also complete before the long statistical gates.

## Verify completion

The final success line names `.upgrade-guard/qualification/runs/<commit>/evidence.json`.
Inspect its `source_git_commit`, `gpu_uuid`, `matrix_lock_sha256`, `environment_images`, `gate_status`, and artifact hashes before publishing a result.

An infrastructure failure, timeout, unsupported tool, or inconclusive statistical result is not a passing qualification.

The project-owned local registry container is removed at terminal cleanup.
Its source-specific Docker volume is retained because the matrix lock and reproduction evidence refer to worker manifests stored there.
Remove that volume only after the evidence has been reviewed or the source run has been intentionally abandoned.
