# TensorRT UpgradeGuard Build Plan

## 1. Document purpose and authority

This file is the implementation authority for TensorRT UpgradeGuard V1.
An implementation agent should read the entire file before changing the repository.
The repository now contains the complete host control plane, GPU worker path, CUDA plugin, qualification runner, fault corpus, tests, and project documentation described below.
The agent must preserve milestone acceptance order and must not publish benchmark claims until the hardware and environment validity gates pass.
The only unresolved external input is the exact NVIDIA GPU and host driver available for final qualification.
That uncertainty is handled by Milestone 0 and must not be guessed away.

### 1.1 Current implementation and acceptance status

Implementation progress and milestone acceptance are tracked separately.
All locally implementable work for Milestones 0 through 10 is present in the repository.
The complete local verification command passes on the development workstation.
No real GPU qualification evidence has been accepted or published.

| Milestone | Implementation state | Acceptance state |
| --- | --- | --- |
| M0 | Hardware discovery, Docker GPU preflight, image resolution, compatibility checks, and matrix locking are implemented. | Deferred until both immutable workers launch on one compatible Linux x86-64 NVIDIA GPU. |
| M1 | Contracts, schemas, failure taxonomy, CLI fixtures, reports, and bundle safety are implemented. | Local macOS verification passes; the same local gate must be rerun on the Linux qualification host. |
| M2-M6 | Worker orchestration, corpus materialization, qualification, statistics, failure reduction, and replay are implemented with CPU fixtures and simulated boundaries. | Target acceptance is deferred because the real workers, engines, and GPU measurements do not exist yet. |
| M7-M9 | `IPluginV3`, CUDA tactics, fault fixtures, sanitizer commands, profiler commands, and evidence validation are implemented. | Target correctness, sanitizer, optimization, and profiler gates are deferred. |
| M10 | Portfolio documentation, demo instructions, evidence contracts, and publication safeguards are implemented. | Final publication remains deferred until the complete real GPU evidence index passes review. |

Resume from M0 on a clean `main` checkout rather than repeating local implementation.

```bash
git switch main
git pull --ff-only origin main
uv sync --frozen
make verify
uv run --frozen upgrade-guard doctor --json
run_root=".upgrade-guard/qualification/runs/$(git rev-parse HEAD)"
uv run --frozen upgrade-guard qualify qualification/full.yaml \
  --project-root . --out "${run_root}"
```

If the target run exposes an implementation defect, fix only the failing boundary, rerun `make verify`, resume the same qualification command, and preserve the generated hardware evidence outside Git.
After a coherent fix passes, commit and push directly to `main` with ordinary `git add`, `git commit`, and `git push origin main` commands.

## 2. Product definition

TensorRT UpgradeGuard is a host-controlled release-qualification laboratory for TensorRT stack upgrades.
It rebuilds identical frozen model artifacts inside a baseline and candidate environment, executes identical inputs and dynamic shapes on the same GPU, separates correctness from performance gates, and produces a reduced, hash-verified reproduction when an upgrade fails.
It also includes a real `IPluginV3` CUDA plugin with reference and optimized tactics so the portfolio demonstrates inference integration, CUDA correctness, profiling, and optimization rather than Python orchestration alone.

The defensible one-sentence claim is:

> TensorRT UpgradeGuard qualifies inference-stack upgrades against frozen ONNX artifacts, numerical and determinism policies, workload-shaped performance gates, reproducible environment locks, and reduced failure bundles.

This is the primary NVIDIA-facing portfolio project.
Its strongest interview evidence should be C++, CUDA, TensorRT, dynamic-shape reasoning, benchmark discipline, Nsight analysis, sanitizer use, compatibility handling, and reproducible systems engineering.

## 3. Problem being solved

A TensorRT upgrade can change model parsing, engine building, tactic selection, numerical output, dynamic-shape acceptance, memory use, or latency.
An average throughput number does not show whether a production shape distribution still satisfies its SLO.
Serialized engines are not portable enough to compare by copying one plan across unrelated stacks.
Profiling every run changes cost and can perturb the measurement that is supposed to qualify the release.
Existing tools solve important parts of the problem, but teams still need a version-aware protocol that freezes inputs, rebuilds per environment, classifies gates consistently, and emits a clean reproduction.

UpgradeGuard supplies that protocol without claiming to be a new compiler, general neural-network fuzzer, or replacement for NVIDIA tooling.

## 4. Relationship to existing tools

UpgradeGuard orchestrates and extends established tools instead of duplicating them.

| Existing tool | Role in UpgradeGuard |
| --- | --- |
| TensorRT and `trtexec` | Engine build, execution, inspection, and primary unprofiled benchmark. |
| Polygraphy | Cross-runner numerical comparison, model inspection, debugging, and graph reduction. |
| ONNX Runtime CPU | Reference execution for supported frozen ONNX models. |
| TensorRT API Capture and Replay | Post-V1 optional build-phase reproduction path for supported Linux C++ builders. |
| Compute Sanitizer | Focused CUDA memory, race, initialization, and synchronization validation. |
| Nsight Systems | Timeline diagnosis after a regression is confirmed. |
| Nsight Compute | Focused kernel diagnosis and optimization evidence. |
| NNSmith | Optional source of pre-generated frozen ONNX cases, never a core runtime dependency. |

TensorRT 11.2's API Capture and Replay records supported engine-build API calls.
It is not an inference-run recorder, does not replace frozen ONNX cases, is Linux-only, and has limitations around Python plugins and dynamic registration.
V1 does not implement it because the core worker builder is Python-oriented and no separate C++ build adapter is justified yet.
A post-V1 extension may add a bounded C++ builder and use capture only as a secondary diagnostic for a confirmed supported build failure.
For ordinary ONNX failures, a reduced ONNX artifact remains the primary reproduction.

## 5. Scope contract

### 5.1 Required V1 capabilities

- Run the control plane on macOS or Linux without importing TensorRT.
- Probe an NVIDIA Linux host and lock its GPU, driver, container runtime, and tool compatibility.
- Resolve human-readable environment tags to immutable OCI digests.
- Probe versions inside each container instead of inferring them from its tag.
- Freeze model, input, plugin source, configuration, and workload hashes before comparison.
- Rebuild engines independently inside baseline and candidate environments.
- Support strongly typed TensorRT networks.
- Keep frozen FP32 and explicit FP16 or Q/DQ artifacts separate.
- Validate parser and build success, numerical output, nonfinite output, output schema, repeated-run determinism, runtime performance, and memory observations.
- Exercise every declared dynamic shape instead of editing only ONNX metadata.
- Benchmark each shape independently and aggregate only through a declared request-shape distribution.
- Alternate baseline and candidate benchmark blocks on the same GPU.
- Reject infrastructure-invalid measurements.
- Reduce model, shape, inputs, options, and environment pair while preserving a stable failure predicate.
- Implement a dynamic `ResidualRMSNorm` `IPluginV3` with a correctness-first and optimized CUDA tactic.
- Run focused Compute Sanitizer checks and bounded Nsight investigations.
- Export a hash-verified reproduction bundle and static report.

### 5.2 Explicit non-goals

- V1 will not run Triton Inference Server, NVIDIA Dynamo, Kubernetes, multi-node, or multi-GPU serving.
- V1 will not include DLA.
- V1 will not make INT8 or FP8 part of the core qualification matrix.
- V1 will not build a learned cost model.
- V1 will not perform general neural-network fuzzing.
- V1 will not claim that a cross-container regression proves TensorRT alone caused it.
- V1 will not execute arbitrary third-party serialized engines.
- V1 will not auto-file GitHub or NVIDIA issues.
- V1 will not distribute a universal precompiled plugin binary.
- V1 will not claim one GPU generalizes to every architecture.
- V1 will not use profiled timings as the primary performance gate.
- V1 will not share timing caches or serialized engines across TensorRT versions in the core experiment.

## 6. Environment and hardware boundary

The development workstation is Apple silicon and has no NVIDIA runtime.
The control plane, schemas, report generation, failure classification, numerical statistics, bundle verification, and CPU fixtures must work there.
All TensorRT, CUDA, sanitizer, and Nsight execution occurs in locked Linux GPU workers.

Final GPU evidence requires:

- Linux x86-64.
- A TensorRT-supported NVIDIA GPU, with SM 7.5 or newer as the researched 2026 starting boundary.
- At least 8 GB VRAM for the proposed corpus, with 16 GB preferred.
- One physical GPU selected by UUID.
- No MIG for the reference publication run.
- A host driver that supports both locked worker environments.
- Exclusive or otherwise verified idle access during benchmark blocks.
- Stable power and clock behavior or a policy that rejects unstable blocks.
- Adequate local disk for two worker images, engines, raw timings, and diagnostic reports.

The implementation must not hardcode a final container pair before `upgrade-guard doctor` probes the real host.
Illustrative matrix files may use nearby NGC TensorRT container tags, but `matrix lock` must verify existence, compatibility, immutable digest, and observed versions.
The first real comparison should use a controlled minor stack change with the same CUDA major line when possible.
A cross-major comparison should be a separate later qualification.

## 7. Strong typing and frozen artifact policy

TensorRT 11 requires strongly typed network workflows for the planned core path.
Legacy weak-typing precision flags must not be used as if they still define precision policy.

The corpus stores separate frozen artifacts:

- A frozen FP32 ONNX graph.
- A frozen explicit FP16 or Q/DQ ONNX graph where that model supports the mode.

Precision transformation occurs once in a separate locked exporter environment.
The resulting ONNX SHA-256 is identical for baseline and candidate.
Neither worker may independently mutate precision, opset, constants, shapes, or graph structure.

The engine is rebuilt inside each worker because serialized plans are platform and version dependent.
Version-compatible engine experiments, when added, are a separate suite with their own claims and gates.

## 8. Trust and artifact policy

TensorRT serialized engines contain executable content and are treated as trusted artifacts.
UpgradeGuard never deserializes an engine received from an untrusted source.
Normal reproduction bundles rebuild engines from verified model and source artifacts.
If a trusted bundle intentionally includes an engine, replay requires an explicit `--trust-included-engine` flag after hash verification and environment review.

Worker containers receive one GPU, a read-only corpus mount, a writable per-run output mount, no Docker socket, no privileged mode, no host devices beyond the selected GPU, and no network after required materials are available.
The control plane accepts only trusted qualification manifests and container images.
Third-party models should be inspected and materialized in an isolated environment.

## 9. System architecture

```text
Host control plane
  -> validates QualificationSpec
  -> resolves and probes EnvironmentLock entries
  -> materializes and verifies frozen corpus
  -> launches one isolated GPU worker at a time
  -> requests build, correctness, and unprofiled benchmark phases
  -> receives typed result artifacts
  -> compares baseline and candidate
  -> confirms failures
  -> invokes bounded reduction
  -> optionally runs sanitizer or Nsight diagnosis
  -> exports report and reproduction bundle
```

The Python host package must not import `tensorrt`, CUDA Python, PyCUDA, or another GPU runtime at module import time.
GPU operations live behind worker commands executed inside containers.
Worker output crosses the boundary through versioned JSON and files rather than Python object serialization.

## 10. Stable contracts

All authored YAML decodes into strict Pydantic models with unknown fields forbidden.
Machine outputs validate against checked-in JSON Schemas.

### 10.1 `QualificationSpec`

Required fields include:

- Specification version.
- Baseline and candidate environment IDs.
- Corpus lock ID.
- Required model cases.
- Precision modes.
- Optimization profiles.
- Concrete shape cases.
- Input fixture IDs.
- Builder option policy.
- Numerical policy.
- Determinism policy.
- Performance policy.
- Memory policy.
- Hardware validity policy.
- Required confirmations.
- Reduction budget.
- Artifact retention policy.

### 10.2 `EnvironmentLock`

Required fields include:

- Registry, repository, and authored image tag.
- OCI index digest when the reference resolves to a multi-platform index.
- Selected Linux x86-64 manifest digest.
- Selected image-configuration digest.
- Base-image manifest digest for a derived worker.
- Final derived-worker manifest and configuration digests.
- Image platform.
- Observed TensorRT version.
- Observed CUDA runtime and toolkit versions.
- Required minimum host driver and observed host driver.
- Python version.
- Polygraphy version.
- ONNX and ONNX Runtime versions.
- `trtexec` path, hash where possible, and supported option inventory.
- Compute Sanitizer version.
- Nsight Systems and Nsight Compute versions.
- C and C++ compiler versions.
- CMake and Ninja versions.
- GPU name, UUID, compute capability, VRAM, VBIOS where available, and power-limit policy.
- Docker, NVIDIA Container Toolkit, kernel, and host operating-system details.
- Probe timestamp and probe command hashes.

### 10.3 `ReferenceEnvironmentLock`

The reference environment is independent of both TensorRT workers and has its own immutable lock.
It records image index, platform manifest, configuration and derived-image digests, operating system, Python, ONNX, ONNX Runtime, execution provider, provider options, NumPy, PyTorch when used for a project-owned formula, CPU architecture, thread settings, and probe command hashes.
Materialization executes every frozen artifact through its declared reference runner before candidate qualification.
An ONNX Runtime CPU reference must prove that the exact graph, dtype, operators, inputs, and outputs execute without implicit dtype rewriting.
A reduced-precision graph unsupported by the locked CPU provider is excluded from V1 before baseline execution unless a separately authored and locked alternative reference environment has already been approved.
Silent upcasting is forbidden.
The plugin micrograph uses the project-owned FP32-accumulation formula rather than pretending ONNX Runtime understands the custom operator.

### 10.4 `CaseManifest`

Required fields include:

- Model ID and source attribution.
- ONNX SHA-256, opset, IR version, inputs, outputs, and external-data files.
- Exporter environment hash.
- Precision artifact identity.
- Optimization profile.
- Concrete input shape.
- Input fixture hashes.
- Reference runner and version.
- Reference environment lock hash and capability-probe result.
- Expected output names, dtypes, and shapes.
- Numerical, semantic, and determinism policies.
- Workload weight.

### 10.5 `BuildManifest`

Required fields include:

- Case and environment hashes.
- Exact build command.
- Parser warnings and errors.
- Builder configuration.
- Plugin source and binary hashes.
- Plugin compiler command and build log.
- Timing-cache mode and hash.
- Build start, end, and duration.
- Engine hash and byte size.
- Engine inspector output.
- Reported device memory.
- Build warnings and failure classification.

### 10.6 `RunResult`

Required fields include:

- Case, build, environment, and hardware hashes.
- Exact execution command.
- Output schema and hashes.
- Numerical summary.
- Repetition-level determinism summary.
- Raw timing artifact path and hash.
- Per-iteration and block metadata.
- Memory observations by source.
- Hardware validity observations.
- Classification and stable failure code.
- Logs, warnings, and diagnostic artifact references.

Schema versions use a project namespace such as `upgradeguard.dev/v1alpha1` until the seeded corpus is stable.

## 11. Failure taxonomy

Stable failure codes are:

- `PREFLIGHT_UNSUPPORTED`
- `CORPUS_INVALID`
- `PLUGIN_COMPILE_FAILED`
- `ONNX_PARSE_FAILED`
- `ENGINE_BUILD_FAILED`
- `ENGINE_DESERIALIZE_FAILED`
- `PROFILE_REJECTED`
- `EXECUTION_FAILED`
- `OUTPUT_SCHEMA_CHANGED`
- `NONFINITE_OUTPUT`
- `NUMERICAL_REGRESSION`
- `NONDETERMINISM_REGRESSION`
- `PERFORMANCE_REGRESSION`
- `MEMORY_REGRESSION`
- `SANITIZER_FAILURE`
- `INFRASTRUCTURE_INVALID`
- `INCONCLUSIVE`

Every failure includes phase, environment, model, precision, shape, input, gate, observed value, threshold, evidence paths, and a stable signature hash.
The reducer preserves the same failure code and relevant predicate fields.
A process crash without adequate evidence is `EXECUTION_FAILED` or `INFRASTRUCTURE_INVALID`, not a numerical regression.

## 12. Public CLI contract

```text
upgrade-guard doctor [--json]
upgrade-guard matrix lock MATRIX.yaml [--out FILE] [--json]
upgrade-guard corpus materialize CORPUS.yaml [--out DIR] [--json]
upgrade-guard qualify QUALIFICATION.yaml --out DIR [--json]
upgrade-guard compare RUN_DIR [--json]
upgrade-guard reduce FAILURE_DIR --out DIR [--json]
upgrade-guard reproduce verify BUNDLE [--json]
upgrade-guard reproduce run BUNDLE --out DIR [--trust-included-engine] [--trust-source-code] [--json]
upgrade-guard report RUN_DIR --format text|json|html
```

`qualify` is the normal end-to-end entry point.
Lower-level developer commands may exist under `upgrade-guard dev` but are not the public README path.
No command may silently rewrite a lock or mutate a frozen corpus.

Exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | All required gates passed. |
| 1 | A qualification gate failed. |
| 2 | Specification, lock, or corpus was invalid. |
| 3 | The requested environment or case was unsupported. |
| 4 | Infrastructure was invalid or the result was inconclusive. |
| 5 | Internal tool failure. |

## 13. Environment matrix workflow

An authored matrix contains human-readable image references and intended comparison order.
A lock operation performs these steps:

1. Verify the NVIDIA container runtime can expose the selected GPU.
2. Resolve each image reference to an immutable digest for Linux x86-64.
3. Start a minimal probe container with one GPU.
4. Capture actual tool and library versions.
5. Verify the host driver supports the container CUDA runtime.
6. Verify the GPU compute capability is supported by the observed TensorRT version.
7. Inventory `trtexec` options from that exact executable.
8. Verify compiler, CMake, CUDA headers, and TensorRT headers for plugin builds.
9. Verify Compute Sanitizer and Nsight tool availability or mark optional capabilities precisely.
10. Write a complete immutable environment lock.

A qualification refuses mutable tags without a matching lock.
It also refuses when the observed digest, tool version, GPU UUID, or driver differs from the locked policy.
The report shows all baseline-to-candidate changes and states that any regression belongs to the compared stack unless a smaller controlled experiment isolates a component.

## 14. Model and input corpus

The V1 core corpus contains three intentionally different families.

### 14.1 Project-owned dynamic mini transformer

Architecture:

- Four transformer blocks.
- Hidden width 256.
- Eight attention heads.
- Feed-forward width 1024.
- GELU activation.
- Pre-normalization.
- No dropout.
- Fixed generated weights.
- Input tensor `tokens` with shape `[batch, sequence, 256]`.
- Mask input with shape compatible with the chosen attention export.
- Output tensor with shape `[batch, sequence, 256]`.

Optimization profile:

```text
min = batch 1, sequence 8
opt = batch 4, sequence 128
max = batch 8, sequence 512
```

Required concrete cases include `1x8`, `1x127`, `1x128`, `1x129`, `4x64`, `4x128`, `8x256`, and `8x512`.
The 127, 128, and 129 cases exercise boundary behavior.
Every shape must execute with real tensors and cannot be represented by ONNX metadata editing alone.

### 14.2 Dynamic MobileNetV3 Small 0.75

Use a properly attributed ONNX Models source snapshot and derive one frozen dynamic-spatial artifact in the exporter environment.
The planning snapshot was ONNX Models commit `4f43949841cb55a0b98dc8fcd045431ccafd9f96` and its Git LFS pointer named object SHA-256 `ef7b5191b3e2586c409ddcfbfef42a6434a9ac885608b8d16e9c767c518f1c31`.
Materialization must verify the downloaded object against that complete hash and must record any intentionally updated source commit separately.

Optimization profile:

```text
min = 1x3x160x160
opt = 8x3x224x224
max = 16x3x320x320
```

Required cases include minimum, optimum, maximum, odd spatial dimensions inside the profile, and at least one batch boundary.
Inputs include deterministic numeric tensors and a very small redistributable image set with source license and SHA-256.

### 14.3 Plugin micrograph

The graph invokes the project-owned `ResidualRMSNorm` custom operator.
Inputs are `x`, `residual`, and FP32 `gamma`.
`x` and `residual` have the same dynamic shape.
The last dimension is hidden width.
The output is:

```text
z = x + residual
rms = sqrt(mean(z * z, axis=-1) + epsilon)
y = z * gamma / rms
```

The reference implementation uses NumPy or PyTorch with FP32 accumulation.
Cases cover FP32, FP16, aligned hidden sizes, non-vector-divisible tails, minimum tokens, maximum tokens, zero inputs, large finite inputs, and noncontiguous host fixture generation where relevant.

### 14.4 Optional generated cases

NNSmith may generate a small frozen ONNX seed corpus in a dedicated generation environment.
The generated files must be materialized, reviewed, hashed, and then treated like ordinary static cases.
NNSmith's TensorRT runtime path must not be used because inspected code relied on legacy binding APIs and would couple UpgradeGuard to an unrelated compatibility layer.

## 15. Reference and numerical policy

For standard ONNX cases, the reference is a pinned ONNX Runtime CPU environment outside both TensorRT workers.
For the plugin case, the reference is the project-owned formula implementation with FP32 accumulation.
Reference execution must reject nonfinite outputs, output-name drift, dtype drift, shape drift, and nondeterminism before TensorRT comparison.

Each candidate output is compared to the reference and to the baseline TensorRT output.
The qualification locks three separate numerical policies:

- Baseline-to-reference validity.
- Candidate-to-reference validity.
- Candidate-to-baseline upgrade drift.

Each elementwise policy uses:

```text
absolute_error <= atol + rtol * abs(reference)
```

Researched starting tolerances are:

| Precision artifact | Starting `atol` | Starting `rtol` |
| --- | --- | --- |
| FP32 | 1e-5 | 1e-4 |
| Explicit FP16 or Q/DQ | 5e-3 | 5e-3 |

These are starting policies rather than post-hoc truths.
Baseline characterization may tighten them but may never widen FP32 beyond `atol = 1e-4, rtol = 1e-3` or explicit FP16 or Q/DQ beyond `atol = 1e-2, rtol = 1e-2`.
Those immutable ceilings are code-level safety limits for V1.
The implementation must justify locked thresholds from task semantics and stable baseline variation and freeze them before candidate evaluation.

The report records maximum, mean, median, and p99 absolute error, maximum and p99 relative error under a small-reference guard, cosine similarity, L2 error, nonfinite counts, and failed element indexes up to a bounded sample.
Task-aware semantics include top-1 and top-5 agreement for MobileNet and exact output schema for every case.
One aggregate allclose boolean is insufficient evidence.

Decision precedence within a case is output schema, nonfinite values, elementwise ceiling, locked task-semantic gate, then aggregate diagnostic metrics.
The reference must first be deterministic, finite, and schema-valid or the case is `CORPUS_INVALID`.

The three-way numerical decision table is:

| Baseline vs reference | Candidate vs reference | Candidate vs baseline drift | Classification |
| --- | --- | --- | --- |
| Fail | Any | Any | `CORPUS_INVALID`, because the baseline does not establish a valid upgrade case. |
| Pass | Fail | Any | `NUMERICAL_REGRESSION`. |
| Pass | Pass | Pass | Numerical gate passes. |
| Pass | Pass | Fail | `NUMERICAL_REGRESSION` for upgrade behavior drift, even when the candidate happens to be closer to the reference. |

An execution, profile, schema, or nonfinite failure uses its more specific failure code before this table.
The report identifies which of the three gates failed and never widens a threshold to rescue a baseline.

## 16. Determinism policy

Each correctness case runs at least 20 repeated executions with identical engine, context policy, shapes, and inputs.
The result records bitwise output hashes separately from tolerance-based equality.
The qualification configuration states whether bitwise stability is required for each case.
An output that is not bitwise identical may still pass a documented tolerance policy, but any change from baseline repetition behavior is visible.
Engine builds are repeated only during confirmation when tactic-selection variability may explain a failure.

Nondeterminism gates must not confuse uninitialized output, nonfinite values, input mutation, or infrastructure instability with normal floating-point variation.
Those causes receive their own evidence and failure code.

## 17. Performance qualification

### 17.1 Primary measurement

`trtexec` is the primary runtime benchmark because it is distributed with each TensorRT environment.
A version adapter inventories supported flags from `trtexec --help` and records the exact command.
The core path uses one inference stream, explicit warmup and measurement duration or iteration count, exact input shapes and files, CUDA Graph disabled, and host-to-device transfer policy declared.
Compute-only and end-to-end timings are separate result series.
No benchmark relies on an undocumented default.

Each concrete shape is measured separately.
An aggregate is computed only from locked workload weights such as:

```yaml
shape_weights:
  b1_s8: 0.20
  b1_s128: 0.35
  b4_s128: 0.25
  b8_s512: 0.20
```

Weights must sum to one and have a workload provenance note.
Arbitrary shapes are never averaged and called an SLO.

### 17.2 Paired block design

For each shape, create at least 20 accepted pairs.
A seeded bit chooses `baseline then candidate` or `candidate then baseline` within each pair, and the two adjacent blocks form that pair.
Every block reaches the locked warmup and thermal condition before recording samples.
The controller records temperature, graphics and memory clocks, power, utilization, GPU UUID, host load, and competing GPU processes where APIs permit.
A block is rejected rather than smoothed when the locked validity policy is violated.

Raw per-iteration timings and all rejection reasons are retained.
The report shows accepted and rejected block counts.

### 17.3 Statistical gate

For accepted pair `j` and shape `s`, compute `r_j,s = log(candidate_block_median / baseline_block_median)`.
The point estimate is `R_s = exp(mean_j(r_j,s))`.
Use 5,000 seeded paired-bootstrap replicates over complete pairs and transform the mean log ratio back to ratio space.
The workload aggregate is `R_weighted = exp(sum_s(weight_s * mean_j(r_j,s)))`, with each shape resampled within its paired blocks on every replicate.
Report the two-sided 95 percent interval for description and one-sided 95 percent lower and upper bounds for the gate.

For locked practical regression allowance `delta_s`, a shape passes only when the one-sided upper bound is at most `1 + delta_s`.
It is a confirmed regression when the one-sided lower bound is greater than `1 + delta_s`.
It is `INCONCLUSIVE` between those conditions.
The same rule applies to the locked workload aggregate allowance.
Any required shape regression makes the overall performance gate fail even when the aggregate passes.
Overall pass requires every required shape and the aggregate to pass.
Any inconclusive required result makes the overall result inconclusive unless another required result is already a confirmed regression.
Fewer than 20 accepted pairs after locked rejection rules produces `INFRASTRUCTURE_INVALID` rather than a wider interval.

Report every ratio, interval, coefficient of variation, accepted and rejected pair count, iteration count, and pilot-estimated minimum detectable effect.
The initial seeded acceptance test injects a 10 percent slowdown and requires its interval to exceed the allowed gate.
No smaller sensitivity claim is made until a real baseline pilot establishes variance.

An A/A comparison with identical environments must be part of protocol validation.
The chosen false-positive tolerance is locked before an A/B candidate result is viewed.

### 17.4 Build performance

Measure cold build with an empty timing cache and warm build with an environment-local timing cache.
Never share timing caches across environment versions.
Record build duration, engine size, parser and builder warnings, tactic evidence, builder memory observations, and cache hash.
The main runtime gate uses one built engine per environment.
Repeated builds are diagnostic when confirmation suggests tactic-selection variance.

### 17.5 Memory evidence

Keep engine-reported device memory, execution-context allocation, builder memory monitoring, coarse process GPU memory, and host memory as separate fields.
An `nvidia-smi` sample is not described as an exact peak allocation.
V1 gateable measurements are serialized engine bytes and the worker's engine-reported device-memory requirement for the exact profile and shape through a version adapter.
Builder monitoring, coarse process samples, and host memory are diagnostic only.
Run three A/A builds before locking the memory protocol.
The default practical allowances are `max(1 MiB, 5 percent)` for serialized engine size and `max(8 MiB, 5 percent)` for engine-reported device memory.
A candidate is `MEMORY_REGRESSION` only when all three confirmation builds exceed a locked allowance and A/A variation stays within it.
Mixed confirmations are `INCONCLUSIVE`.
A seeded result-preserving plugin fixture requests an extra 64 MiB of workspace and must trigger the device-memory gate.

## 18. `ResidualRMSNorm` `IPluginV3`

Plugin identity is:

```text
Name: ResidualRMSNorm
Version: 1
Namespace: com.udayarora.upgradeguard
```

Required interfaces are `IPluginV3`, `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`, and `IPluginCreatorV3One`.
The implementation must support capability routing through `getCapabilityInterface`.
It must implement output dtype negotiation, output shape expressions, position-aware `supportsFormatCombination`, build-time shape checks, runtime validation in `onShapeChange`, workspace calculation, enqueue on TensorRT's provided CUDA stream, context-specific cloning through `attachToContext`, versioned `PluginFieldCollection` serialization, and registry creation for build and runtime phases.

Supported V1 inputs are FP32 or FP16 `x` and `residual` with equal shape and contiguous TensorRT formats, plus FP32 `gamma` matching the hidden dimension.
Accumulation is FP32.
`epsilon` is a validated positive serialized field.
The output dtype matches `x`.
Unsupported rank, shape, dtype, format, gamma length, nonpositive epsilon, or profile transition fails explicitly.

### 18.1 Tactics

`kSCALAR_REFERENCE` uses straightforward scalar loads, FP32 accumulation, a simple block reduction, correct tail behavior, and readability-first code.
`kVECTORIZED_WARP` uses vectorized loads only when alignment and hidden-size divisibility permit, a scalar tail path, warp-shuffle reduction, limited shared memory, FP32 accumulation for FP16 input, and no hidden synchronization.

Expose both tactics through `getNbTactics`, `getValidTactics`, and `setTactic`.
The timing-cache ID contains only creation state not already represented by shape and tensor format.
The selected tactic must be recoverable from engine-inspector or plugin diagnostic evidence.
The optimized tactic remains only if it measurably improves at least one declared workload without violating required correctness or materially harming required cases beyond policy.

### 18.2 Build contract

Use CMake 3.26 or newer, C++17, a supported CUDA language standard, position-independent code, `find_package(CUDAToolkit REQUIRED)`, a narrow `FindTensorRT.cmake`, release-with-line-info builds, and architecture values supplied by the environment lock.
Do not vendor TensorRT or CUDA headers.
Do not use `FetchContent` for CUDA or TensorRT.

Targets are:

```text
upgrade_guard_plugin
upgrade_guard_plugin_smoke
upgrade_guard_kernel_tests
upgrade_guard_fault_plugin
```

Fault fixtures compile only with `UG_ENABLE_FAULT_FIXTURES=ON`.
The plugin shared object is compiled inside each exact worker environment and stored with source hash, command, compiler versions, and binary hash.
The Python wheel remains architecture independent.

## 19. Compute Sanitizer and profiling workflow

### 19.1 Compute Sanitizer

Run `memcheck`, `racecheck`, `initcheck`, and `synccheck` on the focused plugin smoke executable where each tool applies.
Use nonzero error exit codes, line information, bounded deterministic shapes, and kernel-name filters where available.
The clean plugin must pass every applicable tool.
A quarantined tail-handling defect must be detected by `memcheck`.
Sanitizer runs are correctness gates for plugin work and not performance measurements.

### 19.2 Nsight Systems

Run Nsight Systems only after an unprofiled performance regression reproduces.
Add a narrow NVTX capture range around the benchmark body.
Capture a bounded number of CUDA and NVTX events.
GPU metrics are optional because permissions and hardware support vary.
The workflow must continue while explicitly recording their absence.

Current Nsight Systems versions no longer support every historical text or JSON export path.
Preserve the `.nsys-rep` and generate supported SQLite, Arrow, or CSV summaries through version-probed commands.
Do not write a parser that assumes one old export format.

### 19.3 Nsight Compute

Profile only the selected `ResidualRMSNorm` optimized kernel after timeline evidence identifies it.
Start with a basic set and add focused Speed of Light, Memory Workload Analysis, Launch Statistics, and Occupancy sections.
Filter by demangled kernel name and selected invocation.
Do not collect the full metric set across the entire engine.
Preserve `.ncu-rep`, exact command, tool version, source hash, kernel identity, and a small supported summary export.

## 20. Failure confirmation and reduction

A reduction session stores a machine-readable predicate containing failure code, original environment pair, model and output, concrete shape, input hashes, threshold, confirmation count, and allowed infrastructure state.
A failure must reproduce with the locked confirmation count before reduction begins.
Infrastructure-invalid trials return `INCONCLUSIVE` rather than satisfying the predicate.

Reduction order is:

1. Remove unrelated outputs.
2. Reduce the concrete input shape.
3. Narrow the optimization profile around the failing shape.
4. Replace input regions with zeros, ones, constants, or simpler finite values.
5. Remove nonessential builder options.
6. Freeze a failing dynamic shape.
7. Fold shape operations.
8. Invoke `polygraphy debug reduce` through an UpgradeGuard predicate command.
9. Use bisect mode first.
10. Use linear mode on the smaller result when tractable.
11. Reduce an ordered environment history to the first passing and first failing pair.
12. Re-run the final reproduction from an empty directory.

A numerical predicate preserves failure code, output, shape, and threshold relationship.
A performance predicate requires repeated paired blocks and a confidence interval.
A single slow execution cannot satisfy the reducer.
The reducer stops under both trial and wall-clock budgets and returns the smallest confirmed artifact found.

API Capture and Replay is deferred until a post-V1 C++ builder exists.
A future integration must state that capture covers build calls rather than inference execution.

## 21. Reproduction bundle

```text
README.md
bundle.json
SHA256SUMS
model.onnx
model.data, when ONNX external data is required
inputs/
baseline.environment.json
candidate.environment.json
qualification.yaml
expected.json
commands/
logs/
plugin-source/, when relevant
reproduce.sh
```

`upgrade-guard reproduce verify` validates schema, every hash, path safety, file count, expanded size, allowed file types, and environment references before execution.
`upgrade-guard reproduce run` rebuilds engines by default.
Verification proves artifact identity and does not make source code trustworthy.
The CLI never executes the bundled `reproduce.sh` and uses only the typed bundle manifest.
A bundle containing plugin, custom builder, or other compiled source requires `--trust-source-code` in addition to ordinary hash verification.
Before accepting that flag, the CLI displays every source path and hash, worker image digest, requested GPU, and build command.
Third-party source-bearing bundles should run on an ephemeral GPU host.
The script and CLI never upload the bundle or create an external issue.
The README should include the environment, exact failure, one-command reproduction, expected result, and trust warning.

## 22. Seeded defect corpus

CPU-only stored result fixtures cover every failure code and make classifier and report testing possible on macOS.

Real GPU seeds are:

| Seed | Mechanism | Required result |
| --- | --- | --- |
| G1 | Unsupported custom-domain ONNX node | `ONNX_PARSE_FAILED` with parser evidence. |
| G2 | Plugin omits residual under one hidden-size condition | `NUMERICAL_REGRESSION`. |
| G3 | Zero epsilon with zero input in a fault fixture | `NONFINITE_OUTPUT`. |
| G4 | Vectorized tail without a bound in quarantined fault target | `SANITIZER_FAILURE` under memcheck. |
| G5 | Result-preserving extra arithmetic or controlled device delay | `PERFORMANCE_REGRESSION` under the 10 percent seed gate. |
| G6 | Creator fails to restore epsilon | Serialization or numerical failure with field evidence. |
| G7 | Concrete input exceeds the optimization profile | `PROFILE_REJECTED`. |

An intentionally racy production kernel should not be created merely to advertise nondeterminism.
Stored result fixtures test nondeterminism classification until a safe, reproducible GPU seed exists.
Every defect requires a nearby clean control.

## 23. Repository layout

This is the current high-level ownership map.
Use `rg --files` for the exact tracked inventory instead of treating this summary as an exhaustive tree.

```text
tensorrt-upgrade-guard/
  BUILD_PLAN.md
  README.md
  Makefile
  pyproject.toml
  uv.lock
  CMakeLists.txt
  src/upgrade_guard/
    cli.py                     # public command surface
    orchestrator.py            # checked-in qualification entry point
    qualification.py           # host qualification control plane
    contracts/                 # strict authored and machine contracts
    matrix/                    # image, host, and compatibility locking
    corpus/                    # frozen corpus materialization
    containers/                # isolated Docker execution
    worker/                    # in-container build and execution commands
    compare/                   # correctness, determinism, performance, and memory gates
    reduce/                    # bounded failure reduction
    reproduce/                 # bundle creation, verification, and replay
    report/                    # text, JSON, and HTML reports
  scripts/
    run_gpu_qualification.sh   # resumable hardware qualification runner
    qualification_state.py     # source-bound completion markers
    generate_remote_evidence.py
    generate_schemas.py
    check_repository_docs.py
  schemas/
    *.schema.json
  cpp/
    plugin/                    # `IPluginV3` implementation
    kernels/                   # scalar and optimized CUDA tactics
    tests/                     # plugin and kernel checks
    faults/                    # bounded seeded GPU defects
  containers/
    Dockerfile.worker
    Dockerfile.reference
    requirements-*.txt
  matrices/
    examples/
      controlled-minor.yaml
  qualification/
    full.yaml
  models/
    generators/
    locks/
  corpus/
    registry.yaml
    attribution.yaml
  tests/
    unit/
    integration_cpu/
    fixtures/
  docs/
    remote-qualification.md     # exact resume procedure and external prerequisites
  reports/
    published/
```

Generated engines, timing caches, raw input expansions, run directories, large Nsight reports, and worker build products are ignored.
Selected aggregate tables, small raw benchmark tables, locks, manifests, and reports may be committed.

## 24. Implementation stack

The host control plane should use Python 3.12, Pydantic 2, Typer, PyYAML, Jinja2 for static reports if needed, NumPy, SciPy for verified statistical utilities where justified, and uv for locking.
The host Docker integration should use the Docker CLI through a narrow command runner unless a library materially improves digest or stream handling.
The control plane must remain testable without Docker by injecting command and filesystem interfaces.

Worker dependencies are inherited from the locked NVIDIA TensorRT container where possible.
Extra Python packages are installed through a locked worker requirements artifact or a derived immutable worker image.
The final result records observed versions rather than trusting package constraints.

The plugin uses C++17, CUDA, CMake, and Ninja where available.
The implementation agent must use current official TensorRT headers from each worker and must not create a legacy `IPluginV2` fallback.

## 25. Verification strategy

### 25.1 CPU unit tests

- Strict configuration and unknown-field rejection.
- Environment and corpus lock hashing.
- Command construction and shell-argument safety.
- Version-probe parsing across stored outputs.
- Every failure classification and exit code.
- Numerical metrics on hand-calculated arrays.
- Nonfinite, schema, dtype, and shape failure precedence.
- Every row of the three-way numerical decision table.
- Reference capability rejection for unsupported reduced-precision graphs.
- Paired bootstrap on analytical and simulated cases.
- Workload-weight validation.
- Benchmark block rejection policy.
- Determinism hash accounting.
- Reduction state transitions and predicates.
- Report generation from stored fixtures.
- Bundle hash, traversal, symlink, duplicate, and expansion-limit rejection.
- Source-bearing bundle trust acknowledgement and refusal to execute bundled scripts.

### 25.2 CPU integration tests

- Materialize the project-owned transformer twice and obtain identical hashes.
- Validate every ONNX artifact with the ONNX checker.
- Run standard cases through pinned ONNX Runtime CPU.
- Prove each reduced-precision artifact executes in its locked reference provider or is excluded before qualification.
- Generate plugin reference outputs without TensorRT.
- Exercise the complete CLI against stored worker results.
- Verify the macOS package imports without any TensorRT or CUDA module.
- Reproduce a synthetic failure bundle without a GPU up to the documented GPU boundary.

### 25.3 GPU smoke tests

- Lock one current environment on the real GPU.
- Compile and register the plugin.
- Build one standard engine and one plugin engine.
- Run two transformer shapes and one plugin tail shape.
- Compare outputs to references.
- Serialize and reload each trusted engine in the same environment.
- Preserve complete manifests and clean exact run resources.

### 25.4 GPU qualification tests

- Build the same ONNX SHA in both environments.
- Execute every required model, precision, shape, and input case.
- Run 20 determinism repetitions.
- Execute A/A and A/B paired benchmark blocks.
- Retain at least 20 accepted adjacent randomized-order pairs per required shape.
- Detect the seeded 10 percent slowdown.
- Detect the seeded 64 MiB device-memory increase and keep coarse process memory diagnostic.
- Reject an unstable or occupied-GPU block.
- Reduce at least one numerical and one profile failure.
- Reproduce each reduced bundle from a clean directory.

### 25.5 Plugin tests

- Compare scalar and optimized kernels with the reference across aligned and tail shapes.
- Cover FP32 and FP16 inputs with FP32 accumulation.
- Cover zero, random, large finite, and boundary values.
- Verify invalid rank, dtype, gamma, epsilon, and profile transition fail.
- Verify serialization and deserialization preserve fields.
- Verify tactic selection and cloning remain context safe.
- Run all applicable Compute Sanitizer tools on clean targets.
- Confirm the quarantined tail defect is caught by memcheck.

## 26. Ordered implementation milestones

### Milestone 0 - Hardware and compatibility lock

Implement the host package skeleton, schemas, `doctor`, digest resolution, environment probing, compatibility checks, and `matrix lock`.
Run it on the actual NVIDIA machine before choosing the final environment pair.

Gate:

- The real GPU, UUID, compute capability, driver, VRAM, runtime, and tools are captured.
- Two candidate containers launch on the same GPU.
- Every observed version comes from a probe rather than a tag assumption.
- Unsupported combinations fail before model work.

### Milestone 1 - CPU contracts and classifier

Implement every typed contract, result directory, stable failure taxonomy, JSON output, CLI golden fixture, report skeleton, and bundle safety layer.

Gate:

- The complete local CPU gate passes on macOS and Linux.
- Every failure code has a stored result fixture.
- Unknown fields and unsafe bundles fail closed.
- Importing the package does not require CUDA or TensorRT.

### Milestone 2 - Single-environment vertical slice

Use one frozen transformer ONNX artifact, one profile, one shape, one input, and one locked worker.
Implement engine build, ORT reference, TensorRT run, output comparison, build and run manifests, and bundle export.

Gate:

- A fresh GPU directory builds and runs without a host TensorRT installation.
- All inputs, outputs, commands, environments, and binaries have hashes.
- Engine serialization and reload work only inside the same locked environment.

### Milestone 3 - Two-environment qualification

Add sequential baseline and candidate orchestration, stack-difference reporting, failure confirmation, and cross-environment numerical comparison.

Gate:

- Both workers build the exact same ONNX SHA.
- The report separates reference, baseline, and candidate comparisons.
- The result does not attribute stack-level changes to TensorRT alone.

### Milestone 4 - Dynamic corpus

Add the full mini transformer shapes, dynamic MobileNet, input attribution, exporter locks, FP32 and explicit FP16 or Q/DQ artifacts, boundary cases, and profile rejection fixture.

Gate:

- Every declared shape actually executes or receives an explicit failure.
- No worker performs its own precision transform.
- Output names, dtypes, and shapes match the case contract.
- External model source and input licenses are documented.

### Milestone 5 - Determinism and performance

Add repeated execution, `trtexec` adapters, raw timing ingestion, alternating blocks, environment validity, workload weights, paired bootstrap, memory fields, cold and warm build metrics, A/A validation, and seeded slowdown.

Gate:

- Baseline pilot reports variance and minimum detectable effect.
- A/A behavior meets the locked false-positive policy.
- Every required shape retains at least 20 valid paired blocks or the run is infrastructure-invalid.
- The seeded 10 percent slowdown is detected.
- The seeded 64 MiB workspace increase is detected by the engine-reported device-memory gate.
- Profiled measurements are absent from the primary gate.

### Milestone 6 - Failure reduction and reproduction

Add output, shape, profile, input, option, graph, and environment reduction plus hash-verified replay.
Integrate Polygraphy reduction while keeping API Capture and Replay deferred.

Gate:

- Seeded failures produce smaller artifacts with the same predicate.
- A performance reducer uses repeated confidence evidence.
- Final reduced cases reproduce from an empty directory.
- Source-bearing bundles require explicit code trust and never execute their bundled shell script automatically.

### Milestone 7 - Correctness-first `IPluginV3`

Implement creator, interfaces, dynamic shapes, serialization, registration, scalar CUDA tactic, worker compilation, plugin micrograph, and reference comparison.

Gate:

- All FP32 and FP16 plugin cases pass locked numerical policies.
- Engine reload preserves epsilon and output.
- Invalid configurations fail explicitly.
- Scalar CUDA tests and applicable sanitizer tools pass.

### Milestone 8 - Optimized CUDA tactic

Implement vectorized loads, warp reduction, alignment selection, scalar tail, tactic exposure, timing-cache identity, and focused benchmark cases.

Gate:

- Optimized and scalar outputs agree with the reference.
- All applicable sanitizer tools are clean.
- The optimization has a measured benefit on at least one declared workload.
- Required shapes do not regress beyond policy.

### Milestone 9 - Nsight evidence and seeded corpus

Add narrow NVTX ranges, version-aware Nsight Systems capture, focused Nsight Compute analysis, all real GPU fault fixtures, controls, and classifier integration.

Gate:

- The report shows why the optimized tactic behaves as measured.
- Every fault receives its expected failure code.
- Every clean control passes.
- Diagnostic tooling does not alter primary qualification data.

### Milestone 10 - Portfolio publication

Publish architecture, compatibility contract, model and input attribution, numerical policy, benchmark methodology, raw small tables, one reduced numerical failure, one performance regression, one CUDA optimization narrative, limitations, and a short demo.

Gate:

- A trusted user with a compatible GPU can run `doctor`, lock a matrix, and reproduce the smoke qualification.
- Every numerical resume claim points to checked-in aggregate evidence.
- Large generated binaries and profiler files are excluded from Git.
- Security warnings for engines, containers, and GPU runners are prominent.

## 27. Portfolio-ready core and extended NVIDIA track

The portfolio-ready core is complete after Milestone 5 when the real environment lock exists, one standard dynamic corpus runs in two environments, the three-way numerical policy works, repeated determinism evidence is retained, A/A is valid, and the paired performance gate detects its seeded slowdown.
That core is independently suitable for an NVIDIA inference-performance resume bullet and demonstration.
Milestones 6 through 10 form the extended NVIDIA track with reduction, reproduction, `IPluginV3`, CUDA optimization, Compute Sanitizer, and Nsight evidence.
The README and resume must state which gate the repository has actually reached.
The extended definition remains the full V1 target, but failure to finish it does not erase a completed and documented core qualifier.

## 28. Local and target verification

`make verify` is the complete local control-plane gate.
It regenerates schemas, rejects schema drift, validates internal documentation links, runs Ruff formatting and lint checks, runs strict mypy, and executes the configured branch-coverage suite.
Run it before every direct commit to `main` and again on the Linux qualification host before hardware work.

The public `upgrade-guard qualify` command is the complete target-hardware gate.
It invokes the checked-in resumable runner, selects one GPU UUID, rejects competing processes and invalid hardware observations, cleans only project-owned resources, and retains generated evidence outside Git.
The lower-level `UG_SMOKE_ONLY=1 bash scripts/run_gpu_qualification.sh` and `UG_SANITIZER_ONLY=1 bash scripts/run_gpu_qualification.sh` modes remain available for bounded diagnosis.

Run GPU qualification only from a reviewed, clean commit on a trusted host.
Performance results from unverified shared cloud hardware are exploratory and cannot update published claims.

## 29. Security and supply-chain requirements

The repository documents container and engine trust boundaries in `SECURITY.md`.
OCI image digests, Python and system dependency locks, model hashes, input hashes, plugin source hashes, compiler commands, and report checksums are required.
GPU qualification must not expose long-lived registry credentials to untrusted code.
Worker network access is disabled after image and corpus materialization.
Subprocess arguments are passed as arrays and never through an interpolated shell command.
Logs redact registry credentials and environment secrets.
Reproduction extraction is path and size safe.

The release should generate an SBOM for the host package and derived worker image.
Dependency vulnerability findings are triaged before release.
No tool automatically uploads models, outputs, engine files, profiles, or system information.

## 30. Reporting deliverables

The final static report contains:

- Host, GPU, driver, and complete baseline and candidate environment locks.
- Frozen corpus and input provenance.
- Exact builder and runner commands.
- Buildability and warning matrix.
- Reference, baseline, and candidate numerical tables.
- Repetition-level determinism evidence.
- Per-shape and workload-weighted unprofiled performance intervals.
- Accepted and rejected benchmark blocks.
- Memory fields by measurement source.
- Engine-inspector and selected tactic evidence.
- One reduced failure walkthrough.
- One sanitizer finding from a quarantined defect and clean control.
- Nsight evidence explaining the optimized CUDA tactic.
- Threats to validity and non-goals.
- Reproduction commands and bundle hashes.

Every chart must have an underlying small machine-readable table.
The report must distinguish observed fact, policy decision, and inference.

## 31. Risks and mitigations

### Unknown GPU and driver

This is the only external blocker to choosing the final matrix.
Milestone 0 probes the real system and resolves the pair before qualification code makes compatibility assumptions.

### Mutable container tags

Matrix locking resolves OCI digests and records observed tool versions.
Publication mode refuses unlocked tags or version drift.

### Tactic-selection variation

Timing caches remain environment local.
Inspector and tactic artifacts are retained.
Repeated builds happen during confirmation and are analyzed separately from runtime repetitions.

### Thermal and shared-GPU noise

The protocol uses one GPU UUID, warmup, alternating blocks, environmental observations, competing-process checks, paired intervals, and explicit block rejection.

### Polygraphy interface differences

Use the version bundled with each worker through a small command adapter.
Capture version and exact commands.
Do not expose Polygraphy's internal Python classes in UpgradeGuard contracts.

### TensorRT plugin API churn

Implement `IPluginV3` only, compile inside each environment, and keep version shims narrow and compile-tested.
Do not add a legacy V2 fallback that doubles scope.

### API Capture and Replay limitations

API Capture and Replay is a documented post-V1 extension because V1 has no C++ builder adapter.
A future implementation may use capture only for supported C++ build paths after ordinary reduction.
V1 preserves ONNX and plugin source as its portable build evidence.

### Model and profiler artifact size

Keep the project-owned corpus small.
Commit hashes, configs, aggregate tables, and selected reports rather than every engine or profiler artifact.
Use release assets only after license and trust review.

## 32. Definition of done

TensorRT UpgradeGuard V1 is complete only when all conditions below are true.

- The real GPU and driver produce a complete environment lock.
- Both compared workers rebuild the same frozen ONNX hashes.
- FP32 and explicit reduced-precision artifacts remain distinct and strongly typed.
- Every declared dynamic shape actually executes or is explicitly classified.
- Numerical, nonfinite, schema, determinism, performance, and memory evidence remain separate.
- Baseline-to-reference, candidate-to-reference, and candidate-to-baseline gates follow the locked decision table.
- Primary performance gates use unprofiled paired blocks and a validated A/A policy.
- A seeded 10 percent slowdown is detected.
- Failure reduction preserves a typed stable predicate.
- Reproduction rebuilds engines and verifies every source hash.
- The clean `IPluginV3` passes reference and sanitizer gates.
- The optimized CUDA tactic has measured and profiled evidence.
- The complete local gate works on macOS without NVIDIA software.
- GPU qualification runs only from a reviewed, clean commit on a trusted host.
- Public claims are scoped to the tested stack and hardware.

## 33. Implementation-agent rules

The implementation agent should complete `doctor` and the real environment lock before selecting the final container pair.
It should never infer library versions from an image tag.
It should freeze model transformations outside the compared workers.
It should use official TensorRT, CUDA, Polygraphy, Compute Sanitizer, and Nsight documentation for current APIs.
It should benchmark before profiling and profile only a confirmed narrow problem.
It should add a clean control beside every seeded defect.
It should not deserialize untrusted engines or run untrusted source on the GPU host.
It should not add Triton Server, Dynamo, Kubernetes, quantization variants, or a learned planner before V1 is complete.
It should keep claims narrower than the evidence and preserve raw small tables behind charts.

## 34. Authoritative references

- [TensorRT 11.2.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html)
- [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [TensorRT performance benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
- [TensorRT API Capture and Replay](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capture-replay.html)
- [TensorRT custom-layer overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)
- [TensorRT C++ plugin guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html)
- [TensorRT 10.x to 11.x migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html)
- [TensorRT security guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/security.html)
- [TensorRT issue-reporting guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-reporting.html)
- [Polygraphy API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/polygraphy/index.html)
- [TensorRT container installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-container.html)
- [NVIDIA framework container releases](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Nsight Systems CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [ONNX specification](https://onnx.ai/onnx/)
- [NVIDIA TensorRT source](https://github.com/NVIDIA/TensorRT)
- [NNSmith](https://github.com/ise-uiuc/nnsmith)
- [ONNX Models](https://github.com/onnx/models)

Source snapshots inspected during planning include TensorRT commit `1dade062a4e796c14ab6b3f32461ad694ec58951` and NNSmith commit `bc0af42c7d5fc4fd201efb76e5313f6298c2d573`.
These identifiers document the research basis and do not replace executable environment and corpus locks.
