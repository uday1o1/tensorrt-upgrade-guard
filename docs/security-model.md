# Security model

UpgradeGuard treats containers, compilers, plugin source, and serialized TensorRT engines as executable inputs.
Hash verification prevents silent byte changes but does not make those inputs safe.

## Trust boundaries

The host control plane parses authored YAML and JSON with strict schemas.
It does not import TensorRT or CUDA libraries.

GPU workers run with no network, a read-only root filesystem, no added Linux capabilities, `no-new-privileges`, a private IPC namespace, a process limit, and no Docker socket.
Only the source, corpus, and output paths mount into the worker.

## Supply chain

The environment lock records exact OCI manifest and configuration digests.
The Python environment uses `uv.lock` and exact direct dependency versions.
Model and input locks preserve SHA-256 identities.
Local verification runs from the locked Python environment and an exact clean source commit.

The remote runner generates SPDX package inventories for both exact derived workers and binds each document to its immutable image identity.
Publication audits the exact host Python lock and hash-locked Python packages added by the worker and independent reference Dockerfiles.
Preinstalled NGC Python packages, Debian packages, and proprietary NVIDIA packages remain inventory-only scopes with an explicit limitation rather than a vulnerability-free claim.

## Credentials

Registry credentials enter through explicitly scoped environment variables and never enter locks or reports.
Worker execution disables network access after image and corpus materialization.

Logs must not contain tokens, passwords, private keys, or unredacted authorization headers.

## Reproduction bundles

The verifier rejects traversal paths, symlinks, non-regular files, duplicate members, unsupported file types, oversized manifests, excessive file counts, and excessive expanded size.

Source and engine trust acknowledgements are independent.
Run untrusted source only on a disposable machine with no valuable credentials or data.
