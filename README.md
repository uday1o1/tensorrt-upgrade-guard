# TensorRT UpgradeGuard

TensorRT UpgradeGuard helps inference engineers decide whether a complete TensorRT stack upgrade is safe for a frozen workload.
It rebuilds the same ONNX artifacts in baseline and candidate containers, runs identical inputs on one NVIDIA GPU, separates correctness from performance, and retains hash-addressed evidence.

The repository is useful to reviewers who want a concrete example of TensorRT, CUDA, dynamic-shape, statistical benchmarking, sanitizer, profiler, and reproducibility engineering.

## Readiness

| Area | Current evidence |
| --- | --- |
| CPU control plane | Locally verified on Python 3.12 with 192 tests and the configured 90 percent coverage gate. |
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
- Docker with NVIDIA Container Toolkit GPU support;
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
bash scripts/run_cuda_pm_qualification.sh
```

The runner locks the selected GPU and worker identities, materializes the corpora, runs core and extended gates, executes sanitizer seeds and controls, captures focused profiles, creates worker SBOMs, and writes a final evidence index.

A successful run ends with a line shaped like this:

```text
COMPLETE evidence=<repository>/.upgrade-guard/cuda-pm/runs/<commit>/evidence.json
```

The runner stores source-revision-specific completion markers.
Run the same command again after an interruption to resume at the next incomplete step.

Select a different visible GPU by index when needed.

```bash
UG_GPU_INDEX=1 bash scripts/run_cuda_pm_qualification.sh
```

Refer to [Run a remote qualification](docs/remote-qualification.md) for the full procedure and troubleshooting signals.

## Public CLI

```text
upgrade-guard doctor [--json]
upgrade-guard matrix lock MATRIX.yaml --out MATRIX.lock.json [--json]
upgrade-guard corpus materialize RECIPE.yaml --out DIR [--json]
upgrade-guard qualify QUALIFICATION.yaml --out DIR [--json]
upgrade-guard compare RUN_DIR [--json]
upgrade-guard reduce FAILURE_DIR --out DIR [--json]
upgrade-guard reproduce verify BUNDLE [--json]
upgrade-guard reproduce run BUNDLE --out DIR [--trust-included-engine] [--trust-source-code] [--json]
upgrade-guard report RUN_DIR --format text|json|html
```

`reproduce verify` checks paths, file types, sizes, hashes, inventory, and the manifest self-hash.
`reproduce run` never executes the bundle's `reproduce.sh` file.

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
