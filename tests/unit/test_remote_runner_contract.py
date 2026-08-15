"""Static contract checks for the trusted one-command GPU runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RUNNER = Path("scripts/run_gpu_qualification.sh")
KERNEL_BENCHMARK = Path("cpp/tests/kernel_benchmark.cu")
GPU_FAULTS = Path("cpp/faults/gpu_faults.cu")


def test_full_runner_orders_early_and_candidate_gates_fail_closed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ordered = (
        "run_always_step gpu-runtime-preflight",
        "run_step dependency-audit",
        "run_always_step registry-bootstrap",
        "run_step capacity-preflight",
        "run_step corpus-materialization",
        "run_step worker-images",
        "run_step matrix-lock",
        "run_step plugin-compile-test",
        "run_step profiler-preflight",
        "run_step target-readiness",
        "run_step sanitizers",
        "run_step sboms",
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
    assert text.count('--qualification-spec "${QUALIFICATION_TEMPLATE}"') == 6


def test_runner_has_single_process_lock_live_gpu_probe_and_exact_seed_schedule() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"${LOCK_ROOT}/runner.lock"' in text
    assert 'nvidia-smi --id="${UG_EXPECTED_GPU_UUID}"' in text
    assert "scripts/check_docker_gpu_runtime.py" in text
    assert text.index("CURRENT_STEP=reconcile") < text.rindex("run_always_step registry-bootstrap")
    assert "NSIGHT_COMPUTE_COUNTER_PERMISSION_UNAVAILABLE" in text
    assert "ERR_NVGPUCTRPERM" in text
    assert '--pair-index "${accepted}"' in text
    assert '--entrypoint ""' in text
    assert "--step terminal-cleanup" in text


def test_g5_source_schedule_range_and_retry_budget_are_consistent() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    source = GPU_FAULTS.read_text(encoding="utf-8")
    from scripts.validate_seeded_gpu_faults import ORDER_SCHEDULE

    assert len(ORDER_SCHEDULE) == 24
    assert ORDER_SCHEDULE.count("baseline_then_candidate") == 12
    assert "std::array<bool, 24> PERFORMANCE_ORDER_SCHEDULE" in source
    assert "pairIndex) >= PERFORMANCE_ORDER_SCHEDULE.size()" in source
    gpu_faults = runner[runner.index("run_gpu_faults() {") : runner.index("prepare_reductions() {")]
    assert "${accepted} -lt 24 && ${attempt} -lt 72" in gpu_faults
    assert "${accepted} -eq 24" in gpu_faults
    assert gpu_faults.count('> "${attempt_root}/precondition.json" || pair_valid=0') == 1
    assert gpu_faults.count('> "${attempt_root}/sample.json" || pair_valid=0') == 1


def test_terminal_messages_follow_verified_cleanup_and_distinguish_failure() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    finalize = text[text.index("finalize() {") : text.index("finish_failed_qualification() {")]
    failed = text[
        text.index("finish_failed_qualification() {") : text.index("resolve_public_failure() {")
    ]
    assert "COMPLETE evidence=" not in finalize
    assert failed.index("run_step terminal-cleanup") < failed.index("QUALIFICATION_FAILED")
    assert text.rindex("run_step terminal-cleanup") < text.rindex("COMPLETE evidence=")


def test_replay_observes_target_facts_instead_of_accepting_authored_values() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    replay = text[text.index("run_replay_seed() {") : text.index("validate_reduction_replays() {")]
    assert '--gpu "${GPU_UUID}"' in replay
    assert '--local-registry "${REGISTRY_ADDRESS}"' in replay
    assert "--compute-capability" not in replay
    assert "--vram-mib" not in replay
    assert "--driver-version" not in replay
    assert text.count('["observed_failure_code"]==') == 3


def test_runner_bounds_gpu_work_and_cleans_only_its_exact_container() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    gpu_run = text[text.index("gpu_run() {") : text.index("run_core_qualification() {")]
    assert 'bounded_run gpu docker run --rm --name "${container_name}"' in gpu_run
    assert 'bounded_run cleanup docker container rm --force "${container_name}"' in gpu_run
    assert "if docker run" not in gpu_run
    assert "docker container prune" not in text
    assert "docker system prune" not in text


def test_runner_bounds_docker_network_and_dependency_commands() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'source "${PROJECT_ROOT}/scripts/bounded_executor.sh"' in text
    assert "initialize_bounded_executor" in text
    assert "bounded_run build docker build" in text
    assert "bounded_run network docker push" in text
    assert "bounded_run quick docker container start" in text
    assert "bounded_run quick docker exec" in text
    assert "--connect-timeout 3 --max-time 5" in text
    assert "bounded_run preflight nvidia-smi" in text
    assert 'bounded_run network "${bootstrap}/bin/python" -m pip install' in text
    assert "tool run --from pip-audit" not in text
    assert 'bounded_run audit "${UV[@]}" run --frozen pip-audit' in text
    assert "--requirement containers/requirements-worker.txt" in text
    assert "--requirement containers/requirements-reference.txt" in text
    assert '--output "${output}/reference-pip-audit.json"' in text
    assert text.rindex("run_step dependency-audit") < text.rindex("run_step worker-images")
    assert 'if [[ "${RUN_MODE}" == "full" ]]; then\n  run_step dependency-audit' in text


def test_invocation_guard_returns_typed_codes_before_expensive_work(tmp_path: Path) -> None:
    text = RUNNER.read_text(encoding="utf-8")
    functions = text[text.index("invocation_guard_failure() {") : text.index("select_uv() {")]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text(
        '#!/usr/bin/env bash\n[[ "$1" == "rev-parse" ]] && printf "fixture\\n"\n',
        encoding="utf-8",
    )
    uname = fake_bin / "uname"
    uname.write_text(
        '#!/usr/bin/env bash\n[[ "$1" == "-s" ]] && printf "%s\\n" "${FAKE_SYSTEM}" '
        '|| printf "%s\\n" "${FAKE_ARCH}"\n',
        encoding="utf-8",
    )
    python = fake_bin / "python3"
    python.symlink_to(sys.executable)
    git.chmod(0o755)
    uname.chmod(0o755)
    script = tmp_path / "guard.sh"
    state = tmp_path / "state"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f"PROJECT_ROOT={tmp_path!s}\n"
        f"STATE_ROOT={state!s}\n"
        "SOURCE_ID=fixture\n"
        "GPU_INDEX=0\n"
        "GPU_UUID=\n"
        'bounded_run() { shift; "$@"; }\n'
        f"{functions}\n"
        "invocation_guard\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "FAKE_SYSTEM": "Darwin",
        "FAKE_ARCH": "arm64",
    }

    unsupported = subprocess.run(  # noqa: S603 - executes a generated fixed test script
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert unsupported.returncode == 3
    diagnostic = json.loads(
        (state / "diagnostics" / "invocation-guard.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "unsupported"
    assert diagnostic["reason"] == "HOST_OS_UNSUPPORTED"

    environment.update(FAKE_SYSTEM="Linux", FAKE_ARCH="x86_64")
    unavailable = subprocess.run(  # noqa: S603 - executes a generated fixed test script
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert unavailable.returncode == 4
    diagnostic = json.loads(
        (state / "diagnostics" / "invocation-guard.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "infrastructure_invalid"
    assert diagnostic["reason"] == "HOST_TOOL_UNAVAILABLE"
    assert "docker" in diagnostic["detail"]


def test_runner_reuses_pinned_registry_image_and_captures_filesystem_id() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    registry = text[text.index("start_local_registry() {") : text.index("capacity_preflight() {")]
    assert 'ensure_exact_docker_image "${REGISTRY_IMAGE}"' in registry
    assert 'docker pull "${REGISTRY_IMAGE}"' not in registry
    capacity = text[text.index("capacity_preflight() {") : text.index("build_workers() {")]
    assert "stat -c %d /var/lib/registry" in capacity
    assert '--docker-filesystem-id "${docker_filesystem_id}"' in capacity


def test_runner_uses_bounded_preflight_and_evidence_timeouts() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert (
        'bounded_run preflight "${UV[@]}" run --frozen python scripts/check_docker_gpu_runtime.py'
        in text
    )
    assert (
        'bounded_run evidence "${UV[@]}" run --frozen python scripts/generate_remote_evidence.py'
        in text
    )
    assert "bounded_run quick python3 scripts/check_capacity.py" in text
    assert (
        'bounded_run evidence "${UV[@]}" run --frozen python scripts/qualification_state.py verify'
        in text
    )
    assert (
        'bounded_run evidence "${UV[@]}" run --frozen python '
        "scripts/qualification_state.py reconcile" in text
    )


def test_bounded_modes_do_not_materialize_mobilenet_or_run_dependency_audit() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "materialize_bounded_corpora()" in text
    assert 'if [[ "${RUN_MODE}" == "full" ]]; then\n    MOBILENET_CORPUS=' in text
    materialization = (
        'if [[ "${RUN_MODE}" == "full" ]]; then\n'
        "  run_step corpus-materialization materialize_corpora"
    )
    assert materialization in text


def test_profiler_benchmark_uses_registered_nvtx_domain_message() -> None:
    text = KERNEL_BENCHMARK.read_text(encoding="utf-8")
    profile_only = text[text.index("bool profileOnly()") : text.index("} // namespace")]
    assert 'nvtxDomainCreateA("upgrade_guard")' in profile_only
    assert 'nvtxDomainRegisterStringA(domain, "residual_rmsnorm_optimized")' in profile_only
    assert "event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;" in profile_only
    assert "event.message.registered = rangeName;" in profile_only
    assert "NVTX_MESSAGE_TYPE_ASCII" not in profile_only
    assert "event.message.ascii" not in profile_only


def test_profiler_permission_is_early_but_diagnostics_stay_after_benchmark() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    preflight = text[
        text.index("run_profiler_preflight() {") : text.index("run_target_readiness() {")
    ]
    profiles = text[text.index("run_profiles() {") : text.index("generate_sboms() {")]
    assert "--profile-only" in preflight
    assert "--section SpeedOfLight" in preflight
    assert "scripts/validate_ncu_capability.py" in preflight
    assert "NSIGHT_COMPUTE_COUNTER_PERMISSION_UNAVAILABLE" in preflight
    assert "--list-sections" in preflight
    assert "--wait=true" in profiles
    assert "ERR_NVGPUCTRPERM" in profiles
    assert text.rindex("run_step profiler-preflight") < text.rindex("run_step target-readiness")
    assert text.rindex("run_step plugin-benchmark") < text.rindex("run_step profiles")


def test_full_readiness_exercises_both_workers_and_all_three_workloads() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    readiness = text[
        text.index("run_target_readiness() {") : text.index("run_plugin_benchmark() {")
    ]
    assert "local names=(baseline candidate)" in readiness
    assert "tiny-transformer-fp32.onnx" in readiness
    assert "tail-random-h259" in readiness
    assert "mobilenetv3-small-075-dynamic.onnx" in readiness
    assert "scripts/validate_target_readiness.py" in readiness
    assert "--repetitions 2" in readiness
