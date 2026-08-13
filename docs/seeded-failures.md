# Seeded failure walkthroughs

The fault corpus proves that each gate detects a known mechanism and that a nearby clean control still passes.
This page describes the expected evidence shape before publication of real hardware results.

## Reduced numerical failure

G2 launches a quarantined CUDA kernel that omits the residual input for hidden size 259.
The same buffers then run through the clean scalar `ResidualRMSNorm` tactic.

The expected decision is `NUMERICAL_REGRESSION` for the seed and a passing clean control.
The numerical reducer selects the strongest finite elementwise threshold violation and writes one-element reference and candidate arrays with new hashes.

The reduced predicate retains the failure code, output name, original shape, selected multidimensional index, absolute error, threshold, confirmation count, and original failure signature.

## Repeated performance regression

G5 runs a result-preserving identity kernel with and without one controlled device delay.
The runner gathers 20 adjacent control and delayed observations.

The performance reducer cannot use a single slow observation.
It retains at least 20 pairs and requires the one-sided lower bootstrap bound to exceed the 10 percent seed allowance.

The clean control verifies that the delay does not alter output values.

## Other real GPU seeds

| Seed | Mechanism | Expected result |
| --- | --- | --- |
| G1 | Unsupported custom-domain ONNX node without plugin | `ONNX_PARSE_FAILED` |
| G3 | Zero epsilon with zero input | `NONFINITE_OUTPUT` |
| G4 | Quarantined vector tail without a bound | `SANITIZER_FAILURE` |
| G6 | Creator does not restore epsilon | Serialization or numerical failure |
| G7 | Input exceeds optimization profile | `PROFILE_REJECTED` |

The final evidence generator refuses to pass when any seed is absent, has the wrong decision, or lacks its clean control.
