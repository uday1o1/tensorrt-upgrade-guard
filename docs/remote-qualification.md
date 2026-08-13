# How to run a remote qualification

**Goal**: Run every hardware-dependent gate with one resumable command.

**Prerequisites**: Use a clean Linux x86-64 checkout with Docker, NVIDIA Container Toolkit, `nvidia-smi`, and access to the locked NGC images.

## Select the repository

```bash
cd ~/tensorrt-upgrade-guard
```

The runner refuses tracked or untracked source-tree changes because the evidence must name one exact Git commit.

## Run the workflow

```bash
bash scripts/run_cuda_pm_qualification.sh
```

The command creates `.upgrade-guard/cuda-pm/runs/<commit>` for logs, generated corpora, builds, run data, reports, profiles, SBOMs, and completion markers.
Each source commit has an isolated state directory, so a resumed run cannot mix artifacts from different revisions.

## Resume after a failure

Fix the reported external prerequisite or implementation defect, keep the same source revision when appropriate, and run the same command again.

```bash
bash scripts/run_cuda_pm_qualification.sh
```

The runner skips only steps whose marker belongs to the current Git commit.

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

## Run the trusted smoke path

```bash
UG_SMOKE_ONLY=1 bash scripts/run_cuda_pm_qualification.sh
```

The smoke path locks the environment, materializes the plugin corpus, compiles both workers, runs CTest, builds one candidate engine, and executes one case 20 times.

## Verify completion

The final success line names `.upgrade-guard/cuda-pm/runs/<commit>/evidence.json`.
Inspect its `source_git_commit`, `gpu_uuid`, `matrix_lock_sha256`, `environment_images`, `gate_status`, and artifact hashes before publishing a result.

An infrastructure failure, timeout, unsupported tool, or inconclusive statistical result is not a passing qualification.
