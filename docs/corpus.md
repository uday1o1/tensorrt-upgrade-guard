# Corpus and attribution

The corpus freezes model transformations and input generation outside the compared workers.
Both workers receive identical hashes.

## Project-owned mini transformer

The primary model uses dynamic batch and sequence dimensions with hidden size 256.
The repository generates separate FP32 and explicit FP16 ONNX artifacts with deterministic weights.

Concrete cases cover the declared minimum, optimum, intermediate, and maximum shapes.
Every input fixture uses a stable generator seed and records its hash.

## MobileNetV3 Small

The extended standard-model case derives a dynamic input graph from the ONNX Model Zoo object pinned in `models/locks/mobilenetv3.lock.json`.
The lock preserves the source repository revision, source object hash, derived graph hash, and license metadata.

Publication of redistributed model bytes requires the review recorded in `corpus/attribution.yaml`.

## Plugin micrograph

The project-owned custom-domain graph contains `ResidualRMSNorm` with dynamic activation shapes and a one-dimensional FP32 gamma input.
Separate FP32 and explicit FP16 graphs preserve strong typing.

Cases include aligned, unaligned tail, zero, large finite, noncontiguous-generation, minimum, and maximum shapes.

## Artifact locks

Corpus materialization writes through a staging directory and publishes an immutable content-addressed destination only after the complete artifact inventory and materializer identity verify.
An existing identity is reused only when every byte and the producer identity match.
Every entry records a relative path, SHA-256 digest, byte count, and media type.

Run the public materializer with:

```bash
uv run --frozen upgrade-guard corpus materialize corpus/registry.yaml --out DIR --json
```

The command also runs the locked CPU reference for each supported reduced-precision artifact.
