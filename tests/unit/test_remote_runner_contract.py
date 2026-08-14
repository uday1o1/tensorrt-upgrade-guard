"""Static contract checks for the trusted one-command GPU runner."""

from __future__ import annotations

from pathlib import Path

RUNNER = Path("scripts/run_cuda_pm_qualification.sh")


def test_full_runner_orders_early_and_candidate_gates_fail_closed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ordered = (
        "run_always_step registry-bootstrap",
        "run_step capacity-preflight",
        "run_step worker-images",
        "run_step matrix-lock",
        "run_step corpus-materialization",
        "run_step plugin-compile-test",
        "run_step profiler-preflight",
        "run_step aa-pilot",
        "run_step core-qualification",
        "run_step plugin-benchmark",
        "run_step reduction-prepare",
        "run_step replay-G2",
        "run_step replay-G7",
        "run_step reduction-validation",
        "run_step final-evidence",
    )
    positions = [text.rindex(item) for item in ordered]
    assert positions == sorted(positions)


def test_runner_retains_registry_identity_and_uses_content_addressed_inputs() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "docker volume rm" not in text
    assert 'volume_retained":True' in text
    assert 'CORPUS_STORE="${PROJECT_ROOT}/.upgrade-guard/corpora/by-id"' in text
    assert "scripts/corpus_store.py publish" in text
    assert "--core-corpus" in text
    assert "--plugin-corpus" in text
    assert "scripts/qualification_state.py reconcile" in text


def test_runner_has_single_process_lock_live_gpu_probe_and_exact_seed_schedule() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"${LOCK_ROOT}/runner.lock"' in text
    assert 'nvidia-smi --id="${UG_EXPECTED_GPU_UUID}"' in text
    assert "NSIGHT_COMPUTE_COUNTER_PERMISSION_UNAVAILABLE" in text
    assert "ERR_NVGPUCTRPERM" in text
    assert '--pair-index "${accepted}"' in text
    assert '--entrypoint ""' in text
    assert "--step terminal-cleanup" in text
