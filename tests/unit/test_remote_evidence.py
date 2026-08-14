"""Fail-closed validation for generated remote publication evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_remote_evidence import (
    _inventory,
    _validate_clean_replays,
    _validate_cuda_benchmark,
    _validate_pip_audit,
    _validate_profile_summary,
    _validate_spdx,
)


def test_semantic_publication_inputs_are_validated(tmp_path: Path) -> None:
    image = "registry.example/worker@sha256:" + "1" * 64
    sbom = tmp_path / "worker.spdx.json"
    sbom.write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "documentComment": f"Observed package inventory for {image}",
                "packages": [{"name": "package"}],
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps([{"name": "package", "vulns": []}]), encoding="utf-8")
    profile = tmp_path / "profile.csv"
    profile.write_text("Name,Time\nresidualRmsNormFloat4,1\n", encoding="utf-8")

    _validate_spdx(sbom, expected_image=image)
    _validate_pip_audit(audit)
    _validate_profile_summary(profile)

    audit.write_text(
        json.dumps([{"name": "package", "vulns": [{"id": "CVE-TEST"}]}]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="require triage"):
        _validate_pip_audit(audit)
    profile.write_text("TensorRT Release banner only\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="selected kernel"):
        _validate_profile_summary(profile)


def test_inventory_excludes_active_log_and_generated_outputs(tmp_path: Path) -> None:
    state = tmp_path / ("1" * 40)
    (state / "logs").mkdir(parents=True)
    retained = state / "retained.json"
    retained.write_text("{}\n", encoding="utf-8")
    (state / "logs" / "final-evidence.log").write_text("still changing", encoding="utf-8")
    output = state / "evidence.json"
    output.write_text("stale", encoding="utf-8")

    inventory = _inventory(state, output)

    assert set(inventory) == {"retained.json"}


def test_cuda_benchmark_requires_twenty_alternating_pairs() -> None:
    pairs = [
        {"order": ("scalar_then_optimized" if index % 2 == 0 else "optimized_then_scalar")}
        for index in range(20)
    ]
    _validate_cuda_benchmark({"cases": [{"pairs": pairs}]})
    pairs[-1]["order"] = "scalar_then_optimized"
    with pytest.raises(RuntimeError, match="alternate"):
        _validate_cuda_benchmark({"cases": [{"pairs": pairs}]})


def test_reduction_requires_both_clean_typed_cli_replays() -> None:
    digest = "sha256:" + "1" * 64
    value = {
        "clean_bundles": {seed: {"bundle_manifest_sha256": digest} for seed in ("G2", "G7")},
        "clean_replays": {
            "G2": {
                "status": "passed",
                "expected_failure_code": "NUMERICAL_REGRESSION",
                "bundle_manifest_sha256": digest,
                "step_results": ["build", "seed"],
            },
            "G7": {
                "status": "passed",
                "expected_failure_code": "PROFILE_REJECTED",
                "bundle_manifest_sha256": digest,
                "step_results": ["build", "seed"],
            },
        },
    }
    _validate_clean_replays(value)
    value["clean_replays"].pop("G7")
    with pytest.raises(RuntimeError, match="exactly G2 and G7"):
        _validate_clean_replays(value)
