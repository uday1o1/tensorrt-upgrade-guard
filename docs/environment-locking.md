# Environment locking

An authored matrix names a base TensorRT image and a separately derived final worker image for each side of the comparison.
The lock preserves the authored reference, an optional multi-platform index digest, the selected `linux/amd64` manifest digest, and its image-configuration digest.
UpgradeGuard pulls and runs the selected child manifest by digest.
It never runs the mutable tag after resolution.

The final worker must carry the label `com.udayarora.upgradeguard.base.manifest.digest`.
The label value must equal the selected base-image manifest digest.
The supplied worker Dockerfile adds this provenance label and does not embed the changing UpgradeGuard source tree.
Worker source is mounted read-only and hashed by later build and run manifests.

Build the final worker from an already resolved base manifest, publish it only to an authorized registry, and use the resulting immutable worker reference in the matrix.
Matrix locking refuses a base image reused as the worker identity.
It also refuses a missing or mismatched provenance label.

Use the exact selected base manifest in both build arguments.

```text
uv run --frozen upgrade-guard dev resolve-image BASE_IMAGE --json
docker build --platform linux/amd64 --build-arg BASE_IMAGE=REGISTRY/REPOSITORY@sha256:BASE_MANIFEST --build-arg BASE_MANIFEST_DIGEST=sha256:BASE_MANIFEST --tag AUTHORIZED_REGISTRY/upgrade-guard/worker:ENVIRONMENT --file containers/Dockerfile.worker .
docker push AUTHORIZED_REGISTRY/upgrade-guard/worker:ENVIRONMENT
```

The hidden `dev` command is a bootstrap utility and is not the documented qualification entry point.
The push is an explicit operator action because immutable registry manifest identity does not exist until the derived image is stored in the authorized registry.
UpgradeGuard does not push worker images.
Use the pushed reference as `worker_image` and let `matrix lock` independently resolve its manifest and configuration bytes.

Registry credentials are optional and explicitly scoped through `UPGRADE_GUARD_REGISTRY_HOST`, `UPGRADE_GUARD_REGISTRY_USERNAME`, and `UPGRADE_GUARD_REGISTRY_TOKEN`.
All three variables must be present together.
`UPGRADE_GUARD_REGISTRY_AUTH_HOST` must also be set when the registry delegates token authentication to a different trusted host.
Secrets remain in memory and are not written to the environment lock.
Docker must also be authenticated separately when the selected worker manifest requires a private pull.

The lock is written atomically only after both exact workers launch on the selected GPU and all compatibility checks pass.
Existing lock files are never overwritten.
Host-side NVIDIA Container Toolkit version discovery is provenance, not a substitute for the exact worker launch.
When no unprivileged toolkit binary or package record exposes a version, the lock records that version provenance as unavailable together with every attempted source.
It does not infer that the toolkit is absent, invent a version, or infer a CDI runtime from Docker's named runtime inventory.
The completed host record preserves Docker's literal runtime inventory, ordered CDI specification directories, and normalized discovered-device inventory.
Those inventories are provenance only and do not substitute for a capability check.
The lock records `docker --gpus` injection as verified only after both immutable workers succeed on the selected UUID.

Run the hardware gate from a Linux x86-64 host with the selected GPU idle and visible.

```text
uv sync --frozen
uv run --frozen upgrade-guard doctor --json
uv run --frozen upgrade-guard matrix lock MATRIX.yaml --out MATRIX.lock.json --json
```

A successful doctor is only preflight evidence.
The matrix lock is the Milestone 0 gate artifact because it proves both final workers launched by selected manifest digest on the same GPU.
