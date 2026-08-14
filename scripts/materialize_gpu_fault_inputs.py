"""Materialize bounded generated inputs used only by real GPU fault gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.corpus.generators import generate_plugin_micrograph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    root = arguments.destination
    root.mkdir(parents=True, exist_ok=True)
    tokens = np.zeros((9, 8, 256), dtype=np.float32)
    mask = np.zeros((9, 1, 1, 8), dtype=np.float32)
    g7 = root / "g7"
    g7.mkdir(exist_ok=True)
    np.save(g7 / "tokens.npy", tokens, allow_pickle=False)
    np.save(g7 / "mask.npy", mask, allow_pickle=False)
    workspace = root / "residual-rmsnorm-workspace-seed.onnx"
    generate_plugin_micrograph(workspace, extra_workspace_bytes=64 * 1024 * 1024)
    manifest = {
        "schema_version": "upgradeguard.dev/gpu-fault-inputs/v1",
        "artifacts": {
            path.relative_to(root).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (g7 / "tokens.npy", g7 / "mask.npy", workspace)
        },
    }
    (root / "inputs.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
