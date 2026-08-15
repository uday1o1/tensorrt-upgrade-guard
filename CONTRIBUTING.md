# Contributing

Contributions should preserve strict failure semantics, reproducibility, and the host-to-worker security boundary.

## Set up the repository

```bash
uv sync --frozen
```

Python 3.12 is required.

## Run the local gate

```bash
make verify
```

This command regenerates schemas, checks for drift, validates internal documentation links, runs Ruff, runs strict mypy, and executes the coverage suite.

Run the closest real public CLI path when changing user-facing behavior.
Add a clean control beside every new fault seed.

## C++ and CUDA changes

Compile the plugin in every locked worker environment.
Run CTest and all applicable Compute Sanitizer tools before claiming the GPU gate passes.

Measure optimized and scalar tactics without a profiler first.
Capture focused profiler evidence only after the unprofiled result reproduces.

## Contracts and generated files

Edit Pydantic contracts rather than generated JSON Schema files.
Run `make schema-check` after every contract change.

Do not manually modify generated schemas, generated corpora, engines, timing caches, profiler reports, or evidence directories.

## Commits

Use a focused Conventional Commit message.
Do not add automated agent attribution or co-author lines.

Never weaken an acceptance gate to make a change pass.
Keep public claims narrower than the checked-in evidence.
