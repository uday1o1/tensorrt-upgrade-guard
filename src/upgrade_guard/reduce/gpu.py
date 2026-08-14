"""Real candidate-aware GPU predicates for the seeded G2 and G7 failures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from upgrade_guard.containers.runtime import DockerGpuWorker, WorkerMounts
from upgrade_guard.contracts.base import canonical_json_bytes, sha256_bytes, sha256_file
from upgrade_guard.contracts.environment import MatrixLock
from upgrade_guard.errors import FailureCode, InfrastructureError, UpgradeGuardError
from upgrade_guard.reduce.candidate import G2ReductionCandidate, G7ReductionCandidate
from upgrade_guard.reduce.workflow import PredicateObservation, PredicateOutcome


class CandidateGpuPredicate:
    """Execute candidate fields inside the exact locked candidate worker."""

    def __init__(
        self,
        *,
        project: Path,
        state: Path,
        matrix: MatrixLock,
        signature_sha256: str,
        evidence_root: Path,
        worker: DockerGpuWorker | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.project = project.resolve(strict=True)
        self.state = state.resolve(strict=True)
        self.matrix = matrix
        self.signature = signature_sha256
        self.evidence_root = evidence_root
        self.worker = worker or DockerGpuWorker()
        self.timeout_seconds = timeout_seconds
        self._sequence = 0

    def evaluate_g2(
        self, candidate: G2ReductionCandidate, output: Path | None = None
    ) -> PredicateObservation:
        """Run the parameterized C++ seed and require its clean control."""

        trial = self._trial(output, "G2")
        corpus = trial / "corpus"
        corpus.mkdir()
        image = self._image(candidate.environment_id)
        command = [
            f"/state/plugin-build/{candidate.environment_id}/build/upgrade_guard_gpu_faults",
            "--pair-index",
            "0",
            "--rows",
            str(candidate.rows),
            "--hidden",
            str(candidate.hidden),
            "--x-value",
            format(candidate.x_value, ".9g"),
            "--residual-value",
            format(candidate.residual_value, ".9g"),
            "--gamma-value",
            format(candidate.gamma_value, ".9g"),
        ]
        if candidate.outputs == ("G2",):
            command.append("--only-g2")
        try:
            result = self.worker.run(
                image=image,
                gpu_uuid=self.matrix.gpu_uuid,
                mounts=WorkerMounts(self.project, corpus, trial / "output", self.state),
                command=tuple(command),
                timeout_seconds=self.timeout_seconds,
                accepted_returncodes=(0, 1, 2),
            )
        except (InfrastructureError, UpgradeGuardError) as error:
            return self._infrastructure(error)
        raw = trial / "g2-stdout.json"
        raw.write_text(result.stdout.strip() + "\n", encoding="utf-8")
        evidence = (sha256_file(raw),)
        if result.returncode == 2:
            return self._infrastructure(
                InfrastructureError("G2 worker rejected its generated candidate arguments")
            )
        try:
            payload = json.loads(result.stdout)
            observed = payload["G2"]
            reproduced = (
                result.returncode == 0
                and observed["detected"] is True
                and observed["control"] == "passed"
                and int(observed.get("rows", candidate.rows)) == candidate.rows
                and int(observed.get("hidden", candidate.hidden)) == candidate.hidden
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return self._infrastructure(
                InfrastructureError(f"G2 worker output is malformed: {type(error).__name__}")
            )
        return self._observation(
            reproduced,
            FailureCode.NUMERICAL_REGRESSION,
            evidence,
            "G2 numerical relation or clean control did not reproduce",
        )

    def evaluate_g7(
        self, candidate: G7ReductionCandidate, output: Path | None = None
    ) -> PredicateObservation:
        """Rebuild a candidate engine, pass an in-profile control, then reject its shape."""

        try:
            candidate.verify_artifacts()
            trial = self._trial(output, "G7")
            corpus = trial / "corpus"
            worker_output = trial / "output"
            corpus.mkdir()
            shutil.copyfile(candidate.model_path, corpus / "model.onnx")
            self._write_g7_inputs(candidate, corpus)
            (corpus / "profile.json").write_text(
                json.dumps(self._profile(candidate), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            image = self._image(candidate.environment_id)
            mounts = WorkerMounts(self.project, corpus, worker_output, self.state)
            build = self.worker.run(
                image=image,
                gpu_uuid=self.matrix.gpu_uuid,
                mounts=mounts,
                command=(
                    "python3",
                    "-m",
                    "upgrade_guard.worker.build_engine",
                    "--model",
                    "/corpus/model.onnx",
                    "--profile",
                    "/corpus/profile.json",
                    "--engine",
                    "/output/engine.plan",
                    "--inspector",
                    "/output/inspector.json",
                    "--timing-cache",
                    "/output/timing.cache",
                    "--result",
                    "/output/build.json",
                    "--workspace-bytes",
                    str(candidate.workspace_bytes),
                    "--optimization-level",
                    str(candidate.optimization_level),
                ),
                timeout_seconds=self.timeout_seconds,
                accepted_returncodes=(0, 1),
            )
            build_result = self._load_worker_result(worker_output / "build.json")
            if (build.returncode == 0) != (build_result.get("status") == "passed"):
                raise InfrastructureError("G7 build exit code and result status disagree")
            if build.returncode != 0:
                return self._observation(
                    False,
                    FailureCode.PROFILE_REJECTED,
                    self._evidence(worker_output / "build.json"),
                    "candidate engine did not build",
                )
            control = self.worker.run(
                image=image,
                gpu_uuid=self.matrix.gpu_uuid,
                mounts=mounts,
                command=self._correctness_command("control", repetitions=2),
                timeout_seconds=self.timeout_seconds,
                accepted_returncodes=(0, 1),
            )
            control_result = self._load_worker_result(worker_output / "control.json")
            if (control.returncode == 0) != (control_result.get("status") == "passed"):
                raise InfrastructureError("G7 control exit code and result status disagree")
            if control.returncode != 0:
                return self._observation(
                    False,
                    FailureCode.PROFILE_REJECTED,
                    self._evidence(worker_output / "control.json"),
                    "in-profile control did not pass",
                )
            failure = self.worker.run(
                image=image,
                gpu_uuid=self.matrix.gpu_uuid,
                mounts=mounts,
                command=self._correctness_command("failure", repetitions=2),
                timeout_seconds=self.timeout_seconds,
                accepted_returncodes=(0, 1),
            )
            evidence = self._evidence(
                worker_output / "build.json",
                worker_output / "control.json",
                worker_output / "failure.json",
            )
            result = self._load_worker_result(worker_output / "failure.json")
            if (failure.returncode == 0) != (result.get("status") == "passed"):
                raise InfrastructureError("G7 failure exit code and result status disagree")
            reproduced = (
                failure.returncode == 1
                and result.get("status") == "failed"
                and result.get("failure_code") == FailureCode.PROFILE_REJECTED.value
                and "input shape was rejected" in str(result.get("message", ""))
            )
            return self._observation(
                reproduced,
                FailureCode.PROFILE_REJECTED,
                evidence,
                "G7 profile rejection did not reproduce",
            )
        except (InfrastructureError, UpgradeGuardError, OSError, json.JSONDecodeError) as error:
            return self._infrastructure(error)

    def evaluate_g2_environment(
        self, candidate: G2ReductionCandidate, environment_id: str
    ) -> PredicateObservation:
        """Execute the same reduced G2 predicate under one locked environment."""

        return self.evaluate_g2(candidate.model_copy(update={"environment_id": environment_id}))

    def evaluate_g7_environment(
        self, candidate: G7ReductionCandidate, environment_id: str
    ) -> PredicateObservation:
        """Execute the same reduced G7 predicate under one locked environment."""

        return self.evaluate_g7(candidate.model_copy(update={"environment_id": environment_id}))

    def transform_g7(
        self,
        candidate: G7ReductionCandidate,
        operation: str,
        *,
        maximum_seconds: float,
    ) -> G7ReductionCandidate:
        """Run one exact-worker graph transform whose output becomes the next candidate."""

        candidate.verify_artifacts()
        trial = self._trial(None, f"G7-{operation}")
        corpus = trial / "corpus"
        worker_output = trial / "output"
        corpus.mkdir()
        shutil.copyfile(candidate.model_path, corpus / "model.onnx")
        self._write_g7_inputs(candidate, corpus)
        (corpus / "profile.json").write_text(
            json.dumps(self._profile(candidate), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_name = f"{operation}.onnx"
        command = (
            "python3",
            "-m",
            "upgrade_guard.reduce.profile_graph",
            "--operation",
            operation,
            "--model",
            "/corpus/model.onnx",
            "--output",
            f"/output/{output_name}",
            "--profile",
            "/corpus/profile.json",
            "--control-tokens",
            "/corpus/control-tokens.npy",
            "--control-mask",
            "/corpus/control-mask.npy",
            "--failure-tokens",
            "/corpus/failure-tokens.npy",
            "--failure-mask",
            "/corpus/failure-mask.npy",
            "--workspace-bytes",
            str(candidate.workspace_bytes),
            "--optimization-level",
            str(candidate.optimization_level),
            "--signature",
            self.signature,
            "--maximum-seconds",
            format(maximum_seconds, ".9g"),
        )
        result = self.worker.run(
            image=self._image(candidate.environment_id),
            gpu_uuid=self.matrix.gpu_uuid,
            mounts=WorkerMounts(self.project, corpus, worker_output, self.state),
            command=command,
            timeout_seconds=maximum_seconds + 30,
            accepted_returncodes=(0,),
        )
        del result
        model = worker_output / output_name
        if not model.is_file() or model.is_symlink():
            raise InfrastructureError(f"{operation} did not produce a candidate ONNX model")
        history = tuple(
            sha256_file(path)
            for path in sorted(worker_output.glob("*.json*"))
            if path.is_file() and not path.is_symlink()
        )
        if operation in {"bisect", "linear"} and len(history) < 2:
            raise InfrastructureError("Polygraphy reduction did not retain check history")
        return candidate.model_copy(
            update={
                "model_path": model,
                "model_sha256": sha256_file(model),
                "graph_history_sha256": (*candidate.graph_history_sha256, *history),
            }
        )

    def _trial(self, supplied: Path | None, seed: str) -> Path:
        if supplied is not None:
            if any(supplied.iterdir()):
                raise InfrastructureError("clean reduction replay directory is not empty")
            return supplied
        self._sequence += 1
        trial = self.evidence_root / f"{self._sequence:04d}-{seed}"
        if trial.exists() or trial.is_symlink():
            raise InfrastructureError("reduction evidence trial path already exists")
        trial.mkdir(parents=True)
        return trial

    def _image(self, environment_id: str) -> str:
        for environment in self.matrix.environments:
            if environment.id == environment_id:
                return environment.worker_image.canonical_reference
        raise InfrastructureError(f"reduction environment is not locked: {environment_id}")

    def _write_g7_inputs(self, candidate: G7ReductionCandidate, corpus: Path) -> None:
        shape = (candidate.batch, candidate.sequence, candidate.hidden)
        mask_shape = (candidate.batch, 1, 1, candidate.sequence)
        if candidate.input_mode == "original":
            assert candidate.tokens_path is not None and candidate.mask_path is not None
            tokens = np.load(candidate.tokens_path, allow_pickle=False)
            mask = np.load(candidate.mask_path, allow_pickle=False)
        else:
            value = 0.0 if candidate.input_mode == "zeros" else 1.0
            tokens = np.full(shape, value, dtype=np.float32)
            mask = np.full(mask_shape, value, dtype=np.float32)
        if tokens.shape != shape or mask.shape != mask_shape:
            raise InfrastructureError("candidate input arrays do not match the contracted shape")
        np.save(corpus / "failure-tokens.npy", tokens, allow_pickle=False)
        np.save(corpus / "failure-mask.npy", mask, allow_pickle=False)
        control_shape = (
            candidate.profile_min_batch,
            candidate.profile_min_sequence,
            candidate.hidden,
        )
        control_mask_shape = (
            candidate.profile_min_batch,
            1,
            1,
            candidate.profile_min_sequence,
        )
        np.save(corpus / "control-tokens.npy", np.zeros(control_shape, np.float32))
        np.save(corpus / "control-mask.npy", np.zeros(control_mask_shape, np.float32))

    @staticmethod
    def _profile(candidate: G7ReductionCandidate) -> dict[str, object]:
        return {
            "tokens": {
                "min": [
                    candidate.profile_min_batch,
                    candidate.profile_min_sequence,
                    candidate.hidden,
                ],
                "opt": [
                    candidate.profile_opt_batch,
                    candidate.profile_opt_sequence,
                    candidate.hidden,
                ],
                "max": [
                    candidate.profile_max_batch,
                    candidate.profile_max_sequence,
                    candidate.hidden,
                ],
            },
            "mask": {
                "min": [candidate.profile_min_batch, 1, 1, candidate.profile_min_sequence],
                "opt": [candidate.profile_opt_batch, 1, 1, candidate.profile_opt_sequence],
                "max": [candidate.profile_max_batch, 1, 1, candidate.profile_max_sequence],
            },
        }

    @staticmethod
    def _correctness_command(kind: str, *, repetitions: int) -> tuple[str, ...]:
        return (
            "python3",
            "-m",
            "upgrade_guard.worker.run_correctness",
            "--engine",
            "/output/engine.plan",
            "--input",
            f"tokens=/corpus/{kind}-tokens.npy",
            "--input",
            f"mask=/corpus/{kind}-mask.npy",
            "--output",
            f"/output/{kind}-outputs",
            "--result",
            f"/output/{kind}.json",
            "--repetitions",
            str(repetitions),
        )

    @staticmethod
    def _evidence(*paths: Path) -> tuple[str, ...]:
        return tuple(
            sha256_file(path) for path in paths if path.is_file() and not path.is_symlink()
        )

    @staticmethod
    def _load_worker_result(path: Path) -> dict[str, object]:
        if not path.is_file() or path.is_symlink():
            raise InfrastructureError("G7 worker did not retain its typed result")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InfrastructureError("G7 worker result is malformed") from error
        if not isinstance(value, dict) or value.get("status") not in {"passed", "failed"}:
            raise InfrastructureError("G7 worker result has no valid status")
        return value

    def _observation(
        self,
        reproduced: bool,
        failure_code: FailureCode,
        evidence: tuple[str, ...],
        detail: str,
    ) -> PredicateObservation:
        return PredicateObservation(
            outcome=(
                PredicateOutcome.REPRODUCED if reproduced else PredicateOutcome.NOT_REPRODUCED
            ),
            failure_code=failure_code if reproduced else None,
            predicate_signature_sha256=self.signature if reproduced else None,
            evidence_sha256=evidence,
            detail=None if reproduced else detail,
        )

    @staticmethod
    def _infrastructure(error: BaseException) -> PredicateObservation:
        detail = f"{type(error).__name__}: {error}"
        return PredicateObservation(
            outcome=PredicateOutcome.INFRASTRUCTURE_INVALID,
            detail=detail[:4096],
            evidence_sha256=(sha256_bytes(canonical_json_bytes({"detail": detail})),),
        )
