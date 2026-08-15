# Security policy

## Supported versions

The `main` branch is the only supported development line before the first tagged release.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability or exposed credential.
Use GitHub private vulnerability reporting for this repository when available.
Otherwise, contact the repository owner privately through the profile address associated with the repository.

Include the affected commit, a minimal reproduction, impact, and any known mitigation.
Do not include real credentials, restricted models, or private infrastructure details.

## Trust warnings

TensorRT engines are executable, device-specific artifacts.
Load an engine only when you trust its complete provenance.

Container images and plugin source can execute arbitrary code.
Inspect exact digests and source hashes before use.

The `--trust-source-code` and `--trust-included-engine` flags are explicit acknowledgements, not security scans.
Use an ephemeral GPU host for third-party source-bearing bundles.

## Runner isolation

Qualification workers receive no network, Docker socket, added capability, or writable source mount.
The host still trusts Docker, the NVIDIA runtime, the selected images, compilers, and kernel driver.

Run GPU qualification only from a reviewed, clean commit whose source and dependency changes are trusted.
Never execute unreviewed third-party code on a credentialed or persistent GPU host.

## Secrets and artifacts

Never commit registry tokens, SSH keys, credentials, restricted datasets, engines, timing caches, profiler reports, or generated qualification directories.
The `.gitignore` excludes their standard paths and extensions.

Review logs before sharing because host and GPU observations can disclose system details.
UpgradeGuard never uploads evidence automatically.

Refer to [docs/security-model.md](docs/security-model.md) for the complete design.
