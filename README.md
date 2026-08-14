# TensorRT UpgradeGuard

TensorRT UpgradeGuard helps inference engineers decide whether a complete TensorRT stack upgrade is safe for a frozen workload.
It rebuilds the same ONNX artifacts in baseline and candidate containers, runs identical inputs on one NVIDIA GPU, separates correctness from performance, and retains hash-addressed evidence.

The repository is useful to reviewers who want a concrete example of TensorRT, CUDA, dynamic-shape, statistical benchmarking, sanitizer, profiler, and reproducibility engineering.

## Readiness

| Area | Current evidence |
| --- | --- |
| CPU control plane | Locally verified on Python 3.12 with the complete configured test suite and 90 percent coverage gate. |
| Environment resolution | Exact OCI index, manifest, configuration, base, and derived-worker identities are implemented and tested with adversarial fixtures. |
| GPU qualification | Implemented, but no public pass or performance claim exists until the checked-in revision completes on a compatible NVIDIA GPU. |
| CUDA plugin track | Source, scalar and optimized tactics, fault seeds, sanitizer commands, and focused profiler commands are implemented; real GPU gates remain pending. |

The authoritative scope and acceptance gates are in [BUILD_PLAN.md](BUILD_PLAN.md).

## What the qualification covers

The core workload uses a project-owned dynamic mini transformer in FP32 and explicit FP16 modes.
The extended workload adds a frozen dynamic MobileNetV3 Small graph and a custom `ResidualRMSNorm` `IPluginV3` micrograph.

The primary decision keeps these evidence types separate:

- baseline-to-reference and candidate-to-reference numerical checks;
- candidate-to-baseline drift checks;
- 20 repeated outputs for determinism;
- three independent engine builds for build and memory evidence;
- at least 20 accepted adjacent timing pairs per shape;
- before-and-after temperature, clocks, power, utilization, and competing-process checks;
- seeded failures with nearby clean controls;
- unprofiled qualification data kept separate from Nsight diagnostics.

## Requirements

### CPU development

- Python 3.12;
- `uv` 0.11.23;
- Git;
- macOS or Linux for the host control plane.

### Complete GPU qualification

- Linux x86-64;
- one NVIDIA GPU visible through `nvidia-smi`;
- Docker with working `--gpus device=<UUID>` NVIDIA GPU injection;
- access to the exact container digests in `matrices/examples/controlled-minor.yaml`;
- enough local space for two worker images and generated evidence;
- an idle selected GPU during accepted performance blocks.

TensorRT and CUDA Python run only inside locked worker containers.
The host package does not import them.

## Verify the CPU control plane

Install the locked environment and run the same gate used by CPU CI.

```bash
uv sync --frozen
make verify
```

Expected success ends with all tests passing, coverage at or above 90 percent, and this message:

```text
Documentation links and action pins are valid.
```

## Run the complete qualification

Run the preflight on the Linux GPU host.

```bash
uv run --frozen upgrade-guard doctor --json
```

Then run the resumable qualification from a clean tracked checkout.

```bash
run_root=".upgrade-guard/cuda-pm/runs/$(git rev-parse HEAD)"
uv run --frozen upgrade-guard qualify qualification/full.yaml \
  --project-root . --out "${run_root}"
```

The checked-in `scripts/run_cuda_pm_qualification.sh` command is the lower-level resumable runner used by the public CLI and trusted GPU workflows.

The runner locks the selected GPU and worker identities, materializes the corpora, and proves both exact workers can build and reload representative engines before long statistical gates.
It then runs the core and extended gates, executes sanitizer seeds and controls, captures post-benchmark focused profiles, validates worker SBOMs, and writes a final evidence index.

A successful run ends with a line shaped like this:

```text
COMPLETE evidence=<repository>/.upgrade-guard/cuda-pm/runs/<commit>/evidence.json
```

The runner stores source-revision-specific completion markers.
Each marker binds its complete owned artifact inventory, direct dependency markers, matrix lock, corpus identities, selected GPU, run mode, and source commit.
Run the same command again after an interruption to resume at the next incomplete step.

Host-side NVIDIA Container Toolkit version provenance may be unavailable on a managed machine.
The matrix gate still fails closed unless both exact immutable workers execute successfully on the selected GPU UUID.
If preflight reports `NVIDIA_CONTAINER_TOOLKIT_UNAVAILABLE`, an administrator must configure the Docker runtime before the same command can resume.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Select a different visible GPU by index when needed.

```bash
UG_GPU_INDEX=1 bash scripts/run_cuda_pm_qualification.sh
```

Refer to [Run a remote qualification](docs/remote-qualification.md) for the full procedure and troubleshooting signals.

## Public CLI

```text
upgrade-guard doctor [--json]
upgrade-guard matrix lock MATRIX.yaml --out MATRIX.lock.json [--json]
upgrade-guard matrix verify MATRIX.lock.json [--json]
upgrade-guard corpus materialize RECIPE.yaml --reference-lock REFERENCE.lock.json --out DIR [--json]
upgrade-guard qualify QUALIFICATION.yaml --project-root REPOSITORY --out DIR [--json]
upgrade-guard compare RUN_DIR [--json]
upgrade-guard reduce FAILURE_DIR --out DIR [--json]
upgrade-guard reproduce verify BUNDLE [--json]
upgrade-guard reproduce run BUNDLE --out DIR [--gpu GPU-UUID] [--local-registry HOST:PORT] [--trust-included-engine] [--trust-source-code] [--json]
upgrade-guard report RUN_DIR --format text|json|html
```

`reproduce verify` checks paths, file types, sizes, hashes, inventory, and the manifest self-hash.
`reproduce run` never executes the bundle's `reproduce.sh` file.
It rebuilds the engine and evaluates clean-control and expected-failure predicates from the hash-verified typed replay recipe.
It observes the selected GPU's compute capability, VRAM, driver, and Docker platform directly instead of accepting those compatibility facts from command-line input.
The full qualification runner provides its project-owned local registry.
Standalone replay requires an operator-owned Docker Registry v2 endpoint at `--local-registry`.
The replay output must be a new directory outside a directory-form bundle.

Public failure handling is deliberately narrower than the failure taxonomy.
Genuine `NUMERICAL_REGRESSION` decisions from the core, plugin, and MobileNet domains are confirmed, reduced through the locked worker boundary, exported as source-bearing bundles, and replayed from an empty directory before failed evidence is published.
The G5 seeded `PERFORMANCE_REGRESSION` separately proves the paired-performance reducer and clean replay path.
Other genuine V1 failure classes receive an explicit typed `not_applicable` disposition with a precise unsupported-reducer reason instead of a fabricated reproduction claim.
See [the reproduction format](docs/reproduction-format.md#public-failure-reduction-coverage) for the exact support boundary.

## Evidence and reports

Generated evidence stays under `.upgrade-guard/` and remains outside Git.
The final evidence index records the source commit, selected GPU UUID, matrix-lock hash, exact worker images, gate statuses, artifact sizes, and artifact hashes.

The repository does not publish numerical or performance results before a reviewer imports and checks the corresponding machine-readable evidence.
Refer to [reports/published/README.md](reports/published/README.md) for the publication rule.

## Security

Containers, source-bearing bundles, and serialized TensorRT engines are executable trust boundaries.
Review their provenance before use.
Run third-party source-bearing reproductions only on an ephemeral GPU host.

The worker has no network, no Docker socket, no added capabilities, a read-only root filesystem, and narrow bind mounts during qualification.
Refer to [SECURITY.md](SECURITY.md) and [the security model](docs/security-model.md) before running an external bundle.

## Documentation

- [Architecture](docs/architecture.md)
- [Compatibility contract](docs/compatibility.md)
- [Corpus and attribution](docs/corpus.md)
- [Numerical policy](docs/numerical-policy.md)
- [Determinism policy](docs/determinism-policy.md)
- [Benchmarking methodology](docs/benchmarking-methodology.md)
- [`ResidualRMSNorm` plugin design](docs/plugin-design.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Short demonstration](docs/demo.md)
- [Seeded failure walkthroughs](docs/seeded-failures.md)
- [CUDA optimization narrative](docs/cuda-optimization.md)
- [Reproduction format](docs/reproduction-format.md)
- [Limitations](docs/limitations.md)

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the local verification and change rules.

## License

The project is licensed under Apache License 2.0.
Third-party model redistribution follows the separate attribution and review rules in [corpus/attribution.yaml](corpus/attribution.yaml).
