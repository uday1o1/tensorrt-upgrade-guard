# Determinism policy

Every declared correctness case executes 20 times in each worker.
UpgradeGuard stores repetition-level output hashes and input hashes before it summarizes determinism.

## Decisions

`NONDETERMINISTIC_OUTPUT` applies when repeated outputs differ beyond the authored tolerance.
Bitwise equality can be required for cases that declare it.

Input hashes must remain identical across repetitions.
An input mutation is evidence corruption, not output nondeterminism.

## Stored fixtures

CPU fixtures exercise nondeterminism classification without creating an intentionally racy production kernel.
The GPU fault corpus therefore has no race seed designed only for demonstration.

## Relationship to performance

Determinism repetitions are correctness evidence.
They do not replace paired timing blocks and do not enter the performance confidence interval.
