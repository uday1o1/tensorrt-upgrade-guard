# Compatibility contract

This reference defines the observations required before UpgradeGuard accepts an environment pair.

## Platform boundary

The complete qualification requires Linux x86-64 and one selected NVIDIA GPU UUID.
Both workers must launch on that exact UUID.

## Image identity

For a multi-platform image, the lock records:

- the authored reference;
- the index digest;
- the selected `linux/amd64` child-manifest digest;
- the child configuration digest;
- the manifest and configuration media types.

For a single-platform image, the index digest remains absent.
The resolver verifies response bytes against the requested or returned content digest and rejects ambiguous platform entries.

## Worker provenance

The final worker must differ from its base image.
Its `com.udayarora.upgradeguard.base.manifest.digest` label must equal the selected base manifest.

The lock probes the final worker by manifest digest, not by mutable tag.

## Required observations

Each worker probe records:

| Category | Required evidence |
| --- | --- |
| GPU | name, UUID, compute capability, VRAM, VBIOS, and power limit |
| Stack | driver, CUDA runtime, CUDA toolkit, TensorRT, Python, Polygraphy, ONNX, and ONNX Runtime |
| Execution | `trtexec` path, hash, help hash, and option inventory |
| Diagnostics | Compute Sanitizer, Nsight Systems, and Nsight Compute paths and versions |
| Compilation | C, C++, CUDA compilers, CMake, Ninja, and required header paths |
| Host | kernel, architecture, Docker client and server, literal runtime inventory, exact GPU injection result, and NVIDIA Container Toolkit version provenance when observable |

## Acceptance

The pair passes only when current compatibility policy accepts both exact workers and both workers observe the selected GPU.
A missing tool, unsupported option, mismatched UUID, label mismatch, registry failure, or inconclusive probe cannot pass.
An unavailable host-side NVIDIA Container Toolkit version does not by itself make the pair inconclusive because the two exact immutable GPU-container launches are the capability gate.
The unavailable state is disclosed in the lock and cannot be replaced by an inferred runtime mode or fabricated version.

Refer to [Environment locking](environment-locking.md) for the operator procedure.
