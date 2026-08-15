# Short demonstration

This demonstration shows the public control-plane path in about two minutes before a longer GPU run begins.

## Verify the repository

```bash
uv sync --frozen
make verify
```

Point out that the same command runs on macOS and Ubuntu without a host TensorRT or CUDA import.

## Inspect the host

```bash
uv run --frozen upgrade-guard doctor --json
```

On a supported Linux GPU host, highlight the selected GPU UUID, Docker runtime, and absence of mutable stack guesses.
On an unsupported host, highlight the stable fail-closed result instead of presenting it as a pass.

## Start the GPU smoke path

```bash
UG_SMOKE_ONLY=1 bash scripts/run_gpu_qualification.sh
```

Explain that the script resolves exact images, compiles the plugin in both workers, runs CTest, builds one candidate engine, and executes one bounded case.

## Show the full evidence boundary

```bash
run_root=".upgrade-guard/qualification/runs/$(git rev-parse HEAD)"
uv run --frozen upgrade-guard qualify qualification/full.yaml \
  --project-root . --out "${run_root}"
```

The same command resumes source-revision-specific work and ends only after correctness, determinism, unprofiled performance, seeded faults, sanitizer, profiles, SBOMs, dependency audit, and final evidence succeed.

Do not show a fabricated timing or GPU pass.
Use `.upgrade-guard/qualification/runs/<commit>/evidence.json` only after the real run completes.
