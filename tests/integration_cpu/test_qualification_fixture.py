"""Host-control qualification test with a deterministic simulated GPU worker."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tests.factories import digest, resolved_image, worker_probe
from upgrade_guard.containers.commands import CommandResult
from upgrade_guard.contracts.base import sha256_file
from upgrade_guard.contracts.environment import (
    CompatibilityEvidence,
    EnvironmentLock,
    HostObservation,
    MatrixLock,
)
from upgrade_guard.corpus.materialize import materialize_corpus
from upgrade_guard.corpus.reference import run_onnx_reference
from upgrade_guard.qualification import QualificationRunner, compare_stored_run

GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"
TRTEXEC_OPTIONS = (
    "--loadEngine",
    "--shapes",
    "--exportTimes",
    "--warmUp",
    "--duration",
    "--noDataTransfers",
    "--infStreams",
)


class SimulatedWorkerRunner:
    """Materialize worker outputs at the same bind-mounted paths Docker would use."""

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        del timeout_seconds, cwd, env
        command = tuple(args)
        if command[0] == "nvidia-smi":
            if any(item.startswith("--query-gpu=") for item in command):
                stdout = f"{GPU_UUID}, 40, 2100, 9000, 100, 300, 0\n"
            else:
                stdout = ""
            return CommandResult(command, 0, stdout, "", 0.01)
        output = self._mount(command, "/output")
        corpus = self._mount(command, "/corpus")
        inner = command[command.index("PYTHONPATH=/opt/upgrade-guard/src") + 2 :]
        if "upgrade_guard.worker.build_engine" in inner:
            self._build(inner, output)
        elif "upgrade_guard.worker.run_correctness" in inner:
            self._correctness(inner, output, corpus)
        else:
            self._timings(inner, output)
        return CommandResult(command, 0, "", "", 0.01)

    @staticmethod
    def _mount(command: tuple[str, ...], destination: str) -> Path:
        prefix = "type=bind,src="
        suffix = f",dst={destination}"
        for item in command:
            if item.startswith(prefix) and suffix in item:
                source = item.removeprefix(prefix).split(suffix, maxsplit=1)[0]
                return Path(source)
        raise AssertionError(f"missing simulated mount {destination}")

    @staticmethod
    def _host_path(container_path: str, root: Path, prefix: str = "/output/") -> Path:
        return root / container_path.removeprefix(prefix)

    def _build(self, command: tuple[str, ...], output: Path) -> None:
        engine = self._host_path(command[command.index("--engine") + 1], output)
        inspector = self._host_path(command[command.index("--inspector") + 1], output)
        timing_cache = self._host_path(command[command.index("--timing-cache") + 1], output)
        result = self._host_path(command[command.index("--result") + 1], output)
        for path, contents in (
            (engine, b"simulated-engine"),
            (inspector, b"{}\n"),
            (timing_cache, b"simulated-cache"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        result.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "engine": {"bytes": engine.stat().st_size, "device_memory_bytes": 4096},
                }
            ),
            encoding="utf-8",
        )

    def _correctness(self, command: tuple[str, ...], output: Path, corpus: Path) -> None:
        result_container = command[command.index("--result") + 1]
        output_container = command[command.index("--output") + 1]
        result = self._host_path(result_container, output)
        output_directory = self._host_path(output_container, output)
        input_arguments = [
            command[index + 1] for index, item in enumerate(command) if item == "--input"
        ]
        inputs = {
            item.split("=", maxsplit=1)[0]: np.load(
                self._host_path(item.split("=", maxsplit=1)[1], corpus, "/corpus/"),
                allow_pickle=False,
            )
            for item in input_arguments
        }
        model = corpus / "models" / "tiny-transformer-fp32.onnx"
        expected = run_onnx_reference(model, inputs)[0].values
        repetitions = []
        for index in range(20):
            path = output_directory / f"output-{index:02d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, expected, allow_pickle=False)
            repetitions.append(
                {
                    "outputs": [
                        {
                            "name": "output",
                            "path": f"{output_container}/output-{index:02d}.npy",
                            "sha256": sha256_file(path),
                        }
                    ]
                }
            )
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "input_sha256": {
                        name: digest(str(index + 1)) for index, name in enumerate(sorted(inputs))
                    },
                    "repetitions": repetitions,
                }
            ),
            encoding="utf-8",
        )

    def _timings(self, command: tuple[str, ...], output: Path) -> None:
        export = next(
            item.split("=", maxsplit=1)[1] for item in command if item.startswith("--exportTimes=")
        )
        path = self._host_path(export, output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"times": [{"computeMs": 1.0} for _ in range(8)]}),
            encoding="utf-8",
        )


def _environment(identifier: str, image_character: str) -> EnvironmentLock:
    base = resolved_image(
        reference=f"registry.example/upgrade/base:{identifier}",
        manifest_character=image_character,
        config_character="3",
    )
    worker = resolved_image(
        reference=f"registry.example/upgrade/worker:{identifier}",
        manifest_character=image_character,
        config_character="4",
    )
    probe = worker_probe(manifest_digest=worker.manifest_digest, gpu_uuid=GPU_UUID)
    probe = probe.model_copy(
        update={"trtexec": probe.trtexec.model_copy(update={"options": TRTEXEC_OPTIONS})}
    )
    return EnvironmentLock(
        id=identifier,
        base_image=base,
        worker_image=worker,
        declared_base_manifest_digest=base.manifest_digest,
        probe=probe,
        host=HostObservation(
            operating_system="Ubuntu 24.04",
            kernel="6.8.0",
            architecture="x86_64",
            docker_client_version="29.0.0",
            docker_server_version="29.0.0",
            docker_runtime="nvidia",
            nvidia_container_toolkit_version="1.17.8",
        ),
        compatibility=CompatibilityEvidence(
            policy_version="fixture-v1",
            source_urls=("https://docs.nvidia.com/",),
            checked_at=datetime(2026, 8, 13, tzinfo=UTC),
            minimum_driver="580.0",
            minimum_compute_capability="8.0",
            compatible=True,
            reasons=(),
        ),
        probe_command_sha256=digest("5"),
        probe_output_sha256=digest("6"),
        probed_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


def test_complete_qualification_passes_with_simulated_worker(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """api_version: upgradeguard.dev/v1alpha1
kind: CorpusRecipe
id: fixture
generator_version: tiny-transformer-v1
precisions: [fp32]
expected_model_sha256:
  fp32: sha256:16dd39f7df92632a0d9268b0b669ee8e110d0bce6b2da189fd046e3b4d2e71b4
transformer_shapes:
  - {id: b1_s8, batch: 1, sequence: 8, weight: 1.0}
""",
        encoding="utf-8",
    )
    corpus = tmp_path / ".upgrade-guard" / "corpora" / "fixture"
    materialize_corpus(recipe, corpus)

    matrix = MatrixLock(
        api_version="upgradeguard.dev/v1alpha1",
        kind="EnvironmentLock",
        source_matrix_sha256=digest("7"),
        gpu_uuid=GPU_UUID,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        environments=(_environment("baseline", "1"), _environment("candidate", "2")),
        lock_sha256=digest("8"),
    )
    matrix = matrix.model_copy(update={"lock_sha256": matrix.computed_sha256()})
    matrix_path = tmp_path / "matrix.lock.json"
    matrix_path.write_text(matrix.model_dump_json(indent=2) + "\n", encoding="utf-8")
    specification = tmp_path / "qualification.yaml"
    specification.write_text(
        f"""api_version: upgradeguard.dev/v1alpha1
kind: Qualification
baseline_environment_id: baseline
candidate_environment_id: candidate
environment_lock: {matrix_path}
corpus_lock_id: fixture
required_cases: [tiny-transformer]
precision_modes: [fp32]
optimization_profiles:
  - id: transformer
    inputs:
      tokens: {{minimum: [1, 8, 256], optimum: [1, 8, 256], maximum: [1, 8, 256]}}
      mask: {{minimum: [1, 1, 1, 8], optimum: [1, 1, 1, 8], maximum: [1, 1, 1, 8]}}
concrete_shapes:
  - id: b1_s8
    inputs: {{tokens: [1, 8, 256], mask: [1, 1, 1, 8]}}
input_fixture_ids: [deterministic-numeric]
builder:
  strongly_typed: true
  timing_cache: environment_local
  workspace_limit_bytes: 1048576
  optimization_level: 3
numerical:
  baseline_to_reference: {{atol: 0.00001, rtol: 0.0001}}
  candidate_to_reference: {{atol: 0.00001, rtol: 0.0001}}
  candidate_to_baseline: {{atol: 0.00001, rtol: 0.0001}}
determinism:
  repetitions: 20
  require_bitwise: true
  tolerance: {{atol: 0, rtol: 0}}
performance:
  warmup_milliseconds: 0
  measurement_milliseconds: 1
  minimum_accepted_pairs: 20
  bootstrap_replicates: 1000
  bootstrap_seed: 20260813
  practical_allowance: 0.05
  shape_allowances: {{b1_s8: 0.05}}
  shape_weights: {{b1_s8: 1.0}}
  workload_provenance: deterministic fixture
  one_inference_stream: true
  cuda_graph: false
memory:
  confirmation_builds: 3
hardware_validity:
  selected_gpu_uuid: {GPU_UUID}
  maximum_temperature_celsius: 85
  maximum_clock_variation_ratio: 0.1
  maximum_power_variation_ratio: 0.1
  maximum_gpu_utilization_before_block: 5
  reject_competing_compute_processes: true
  require_stable_power_limit: true
required_confirmations: 2
reduction_budget:
  maximum_trials: 10
  maximum_seconds: 60
  confirmation_count: 2
retention: {{}}
""",
        encoding="utf-8",
    )

    destination = tmp_path / "run"
    outcome = QualificationRunner(runner=SimulatedWorkerRunner(), source_root=tmp_path).run(
        specification, destination
    )
    assert outcome.status == "passed"
    assert outcome.failure_codes == ()
    summary = compare_stored_run(destination)
    assert summary["status"] == "passed"
    assert summary["cases"][0]["candidate_determinism"]["bitwise_stable"] is True
