# Reproduction bundle format

A reproduction bundle contains a strict manifest plus hash-addressed payload files.
Verification establishes identity and path safety, not trustworthiness.

## Required layout

```text
README.md
bundle.json
SHA256SUMS
model.onnx
inputs/
baseline.environment.json
candidate.environment.json
qualification.yaml
expected.json
commands/replay.json
reproduce.sh
```

Source-bearing cases can also contain `plugin-source/`, command records, and logs.
Public numerical-failure bundles retain a hash-bound three-way predicate under `reduction/`, the typed source failure and source result under `logs/`, and every evidence artifact referenced by that failure record.
An included serialized engine is optional and requires separate explicit trust.

## Verification

```bash
upgrade-guard reproduce verify BUNDLE --json
```

The verifier checks:

- normalized relative paths;
- regular files only;
- allowed file types;
- duplicate and traversal rejection;
- file-count and expanded-size limits;
- exact byte counts and SHA-256 digests;
- complete manifest inventory;
- the manifest self-hash.

Directory, ZIP, and tar bundles use the same checks.
Clean-directory materialization reads verified members individually and never calls a general archive extraction function.

## Trust gates

A source-bearing bundle requires `--trust-source-code` after the operator reviews every source path, source hash, worker image digest, selected GPU UUID, and build command.
An included engine independently requires `--trust-included-engine`.

The CLI never executes `reproduce.sh` from the bundle.
That file exists only as a convenience for a trusted operator who chooses to invoke it directly.

## Typed execution

```bash
upgrade-guard reproduce run BUNDLE --out EMPTY_DIR \
  --gpu GPU-11111111-1111-1111-1111-111111111111 \
  --trust-source-code --json
```

The output directory must not exist and must be outside a directory-form bundle.
The CLI verifies and materializes the bundle before opening the candidate environment lock or replay recipe.
The original worker manifest and GPU UUID remain immutable provenance.
The selected replay GPU can differ from the original GPU when its directly observed platform, compute capability, driver, and VRAM satisfy the bundle's locked replay requirements.
When exactly one GPU is visible, `--gpu` can be omitted.
When multiple GPUs are visible, the operator must select one UUID explicitly.
A Docker Registry v2 endpoint must already be listening at `--local-registry`, which defaults to `127.0.0.1:5500`.
The full qualification runner provisions that project-owned registry automatically.
For a standalone replay, start an operator-owned local registry before the command and remove that exact container afterward:

```bash
docker run --detach --name upgrade-guard-replay-registry \
  --publish 127.0.0.1:5500:5000 \
  registry@sha256:46faa9a1ae6813194b53921a370f2f4f8c5e1aae228a89bceafef5847a6a3278
upgrade-guard reproduce run BUNDLE --out EMPTY_DIR --trust-source-code --json
docker container rm --force upgrade-guard-replay-registry
```

It then runs only the argument arrays in `commands/replay.json` through the isolated GPU worker boundary.
The first recipe step must exactly match the build command in the bundle manifest.
Every step declares accepted return codes and any required result-file status or JSON predicate.
The replay passes only when the clean control succeeds and the seeded case fails for the authored reason.
The CLI writes per-step records, a bounded `logs/worker-build.log`, and `replay-result.json` into the new output directory.
The replay result binds the exact rebuild-log artifact, rebuilt worker manifest, selected replay GPU, bundle manifest, observed failure code, and observed predicate evidence.

## Reduction order

The bounded reducer removes unrelated outputs, narrows shapes and profiles, simplifies finite inputs, removes nonessential options, delegates graph reduction through argument-array Polygraphy commands, and isolates the first adjacent pass-to-fail environment boundary.

Every behavioral reducer requires repeated confirmation.
Infrastructure-invalid trials are inconclusive.
Performance reduction retains at least 20 paired blocks and the original confidence-based regression decision.
The G5 seeded performance path starts with 24 balanced hardware-valid pairs, retains a smaller hash-addressed subset of at least 20 pairs, and re-evaluates that subset for the locked confirmation count from a new empty directory.

## Public failure reduction coverage

Genuine `NUMERICAL_REGRESSION` decisions from the core, plugin, or MobileNet domains use the full confirmation, candidate reduction, source-bearing bundle, and empty-directory replay path.
The G5 seeded `PERFORMANCE_REGRESSION` has its own paired-data reduction and clean replay gate.

Other genuine domain failure classes currently receive a typed `not_applicable` disposition with a specific reason.
This includes build, parse, deserialization, profile, execution, output-schema, nonfinite, nondeterminism, memory, and generic performance failures for which V1 does not provide a faithful portable reducer and replay boundary.
The publication reports that limitation and never substitutes a seeded failure or unrelated control as evidence for the genuine failure.
