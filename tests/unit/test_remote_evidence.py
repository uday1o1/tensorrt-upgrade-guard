"""Fail-closed validation for generated remote publication evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_remote_evidence import (
    BENCHMARK_ORDER_SCHEDULE,
    _build_manifest_table,
    _inventory,
    _validate_clean_replays,
    _validate_cuda_benchmark,
    _validate_pip_audit,
    _validate_sanitizer_evidence,
    _validate_spdx,
)
from scripts.validate_profiler_outputs import validate_summary
from upgrade_guard.containers.commands import command_sha256


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
    validate_summary(profile, summary_kind="nsys")

    audit.write_text(
        json.dumps([{"name": "package", "vulns": [{"id": "CVE-TEST"}]}]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="require triage"):
        _validate_pip_audit(audit)
    profile.write_text("TensorRT Release banner only\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="selected kernel"):
        validate_summary(profile, summary_kind="nsys")


def test_inventory_excludes_active_log_and_generated_outputs(tmp_path: Path) -> None:
    state = tmp_path / ("1" * 40)
    (state / "logs").mkdir(parents=True)
    retained = state / "retained.json"
    retained.write_text("{}\n", encoding="utf-8")
    (state / "stale").mkdir()
    (state / "stale" / "old.bin").write_bytes(b"old")
    (state / "diagnostics").mkdir()
    (state / "diagnostics" / "failure.json").write_text("{}\n", encoding="utf-8")
    (state / "logs" / "final-evidence.log").write_text("still changing", encoding="utf-8")
    output = state / "evidence.json"
    output.write_text("stale", encoding="utf-8")

    inventory = _inventory(state, output)

    assert set(inventory) == {"retained.json"}


def test_cuda_benchmark_requires_twenty_seeded_balanced_pairs() -> None:
    pairs = [{"order": order} for order in BENCHMARK_ORDER_SCHEDULE]
    _validate_cuda_benchmark({"order_seed": 20260813, "cases": [{"pairs": pairs}]})
    pairs[-1]["order"] = "scalar_then_optimized"
    with pytest.raises(RuntimeError, match="balanced"):
        _validate_cuda_benchmark({"order_seed": 20260813, "cases": [{"pairs": pairs}]})


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


def test_sanitizer_controls_and_diagnostic_are_hash_bound(tmp_path: Path) -> None:
    root = tmp_path / "sanitizers"
    root.mkdir()
    diagnostic = root / "sanitizer-tail-oob.log"
    diagnostic.write_text("Invalid __global__ read\nERROR SUMMARY: 1 error\n", encoding="utf-8")
    from upgrade_guard.contracts.base import sha256_file

    (root / "sanitizer-seed.json").write_text(
        json.dumps(
            {
                "expected": "SANITIZER_FAILURE",
                "control": "passed",
                "observed_exit_code": 86,
                "diagnostic": "out_of_bounds_global_access",
                "diagnostic_log_sha256": sha256_file(diagnostic),
            }
        ),
        encoding="utf-8",
    )
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        (root / f"sanitizer-{tool}-control.log").write_text(
            "ERROR SUMMARY: 0 errors\n", encoding="utf-8"
        )
    _validate_sanitizer_evidence(tmp_path)
    diagnostic.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact diagnostic"):
        _validate_sanitizer_evidence(tmp_path)


def test_build_table_requires_exact_strongly_typed_commands(tmp_path: Path) -> None:
    path = tmp_path / "core-run" / "baseline" / "fp32" / "build-0.json"
    path.parent.mkdir(parents=True)
    value = {
        "status": "passed",
        "command": ["python3", "-m", "upgrade_guard.worker.build_engine"],
        "command_sha256": command_sha256(["python3", "-m", "upgrade_guard.worker.build_engine"]),
        "strongly_typed": True,
        "engine": {"sha256": "sha256:" + "2" * 64},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert len(_build_manifest_table(tmp_path)) == 1
    value["strongly_typed"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="strong-typed"):
        _build_manifest_table(tmp_path)
