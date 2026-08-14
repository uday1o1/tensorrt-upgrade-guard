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
upgrade-guard reproduce run BUNDLE --out EMPTY_DIR --trust-source-code --json
```

The output directory must not exist and must be outside a directory-form bundle.
The CLI verifies and materializes the bundle before opening the candidate environment lock or replay recipe.
It requires the locked worker manifest and GPU UUID to match the reviewed source-build request.
It then runs only the argument arrays in `commands/replay.json` through the isolated GPU worker boundary.
The first recipe step must exactly match the build command in the bundle manifest.
Every step declares accepted return codes and any required result-file status or JSON predicate.
The replay passes only when the clean control succeeds and the seeded case fails for the authored reason.
The CLI writes per-step records and `replay-result.json` into the new output directory.

## Reduction order

The bounded reducer removes unrelated outputs, narrows shapes and profiles, simplifies finite inputs, removes nonessential options, delegates graph reduction through argument-array Polygraphy commands, and isolates the first adjacent pass-to-fail environment boundary.

Every behavioral reducer requires repeated confirmation.
Infrastructure-invalid trials are inconclusive.
Performance reduction retains at least 20 paired blocks and the original confidence-based regression decision.
