# Numerical policy

UpgradeGuard makes three independent numerical decisions for every output.

| Comparison | Purpose |
| --- | --- |
| Baseline to reference | Establish whether the existing stack is already outside policy. |
| Candidate to reference | Establish whether the proposed stack is independently correct. |
| Candidate to baseline | Detect upgrade drift even when both stacks remain close to the reference. |

The comparison uses `absolute_error <= atol + rtol * abs(reference)` element by element.
The report preserves maximum absolute error, maximum relative error, mismatch counts, nonfinite counts, and optional classification agreement.

## V1 ceilings

| Precision | Maximum `atol` | Maximum `rtol` |
| --- | ---: | ---: |
| FP32 | `1e-4` | `1e-3` |
| Explicit FP16 | `1e-2` | `1e-2` |
| Q/DQ | `1e-2` | `1e-2` |

An authored policy can be stricter than these ceilings but cannot be looser.
The checked-in full qualification uses `0.005` absolute and relative tolerances for explicit FP16.

## Nonfinite values

NaN or infinity receives `NONFINITE_OUTPUT` instead of an ordinary tolerance failure.
The reducer also keeps nonfinite evidence separate from finite numerical reduction.

## Reference capability

A reduced-precision artifact must execute in the locked CPU reference provider before baseline execution.
Unsupported graphs are excluded or classified before qualification rather than silently compared with a different artifact.
