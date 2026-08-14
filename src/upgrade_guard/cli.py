"""Public command-line interface."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from upgrade_guard.classify import exit_code_for_failure
from upgrade_guard.doctor import doctor_exit_code, doctor_json, run_doctor
from upgrade_guard.errors import ExitCode, FailureCode, InvalidInputError, UpgradeGuardError
from upgrade_guard.matrix.digest import (
    RegistryClient,
    credentials_from_environment,
    parse_image_reference,
)
from upgrade_guard.matrix.lock import MatrixLocker
from upgrade_guard.report.html_report import render_html
from upgrade_guard.report.json_report import render_json
from upgrade_guard.report.model import ReportModel
from upgrade_guard.report.text import render_text
from upgrade_guard.reproduce.run import execute_replay, observe_replay_target
from upgrade_guard.reproduce.verify import verify_bundle

app = typer.Typer(
    name="upgrade-guard",
    help="Qualify TensorRT stack upgrades against frozen artifacts.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
matrix_app = typer.Typer(help="Resolve and lock worker environment matrices.", no_args_is_help=True)
dev_app = typer.Typer(help="Lower-level development utilities.", no_args_is_help=True)
reproduce_app = typer.Typer(
    help="Verify and replay reduced failure bundles.",
    no_args_is_help=True,
)
corpus_app = typer.Typer(help="Materialize and verify frozen model corpora.", no_args_is_help=True)
app.add_typer(matrix_app, name="matrix")
app.add_typer(corpus_app, name="corpus")
app.add_typer(dev_app, name="dev", hidden=True)
app.add_typer(reproduce_app, name="reproduce")


def _guard_cli_command[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Map every unexpected public-command failure to the stable exit contract."""

    @wraps(function)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except typer.Exit:
            raise
        except UpgradeGuardError as error:
            bound = signature(function).bind_partial(*args, **kwargs)
            _fail(error, json_output=bool(bound.arguments.get("json_output", False)))
        except Exception:
            bound = signature(function).bind_partial(*args, **kwargs)
            _fail(
                UpgradeGuardError("unexpected internal tool failure"),
                json_output=bool(bound.arguments.get("json_output", False)),
            )

    return guarded


@corpus_app.command("materialize")
@_guard_cli_command
def corpus_materialize_command(
    recipe: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--out", help="New immutable corpus directory")],
    reference_lock: Annotated[
        Path,
        typer.Option(
            "--reference-lock",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Independent immutable reference-environment lock",
        ),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Emit the corpus lock")] = False,
) -> None:
    """Generate frozen models and inputs, then execute their CPU references."""

    from upgrade_guard.corpus.materialize import materialize_corpus

    try:
        from upgrade_guard.contracts.reference_environment import ReferenceEnvironmentLock

        reference = ReferenceEnvironmentLock.model_validate_json(
            reference_lock.read_text(encoding="utf-8")
        )
        if reference.computed_sha256() != reference.lock_sha256:
            raise InvalidInputError("reference environment lock self-hash differs")
        lock = materialize_corpus(
            recipe,
            output,
            reference_environment_sha256=reference.lock_sha256,
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        _fail(
            InvalidInputError(
                "reference environment lock is invalid",
                details={"reason": str(error)},
            ),
            json_output=json_output,
        )
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    if json_output:
        typer.echo(lock.model_dump_json(indent=2))
    else:
        typer.echo(f"Materialized immutable corpus: {output}")
        typer.echo(f"Artifacts: {len(lock.artifacts)}")


@app.command("qualify")
@_guard_cli_command
def qualify_command(
    qualification: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--out", help="New qualification result directory")],
    project_root: Annotated[
        Path | None,
        typer.Option("--project-root", help="Repository root containing BUILD_PLAN.md"),
    ] = None,
    matrix: Annotated[
        Path | None,
        typer.Option("--matrix", help="Authored baseline/candidate environment matrix"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit the result identity")] = False,
) -> None:
    """Run or resume every portfolio-core and extended-V1 qualification gate."""

    from upgrade_guard.orchestrator import FullQualificationRunner

    try:
        outcome = FullQualificationRunner().run(
            qualification,
            output,
            project_root=project_root,
            matrix_path=matrix,
            capture_output=json_output,
        )
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    payload = {
        "schema_version": "upgradeguard.dev/qualification-command/v1",
        "directory": str(outcome.directory),
        "status": outcome.status,
        "failure_codes": [item.value for item in outcome.failure_codes],
    }
    typer.echo(
        json.dumps(payload, indent=2, sort_keys=True)
        if json_output
        else f"Qualification {outcome.status}: {outcome.directory}"
    )
    if outcome.status == "failed":
        raise typer.Exit(code=1)
    if outcome.status == "unsupported":
        raise typer.Exit(code=3)
    if outcome.status in {"inconclusive", "infrastructure_invalid"}:
        raise typer.Exit(code=4)


@app.command("compare")
@_guard_cli_command
def compare_command(
    run_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine JSON")] = False,
) -> None:
    """Validate and display a completed qualification decision."""

    from upgrade_guard.qualification import compare_stored_run

    try:
        summary = compare_stored_run(run_directory)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Qualification status: {summary['status']}")
        typer.echo(f"Failure codes: {', '.join(summary['failure_codes']) or 'none'}")
    decision_exit = _decision_exit_code(summary["status"], summary["failure_codes"])
    if decision_exit:
        raise typer.Exit(code=decision_exit)


@app.command("reduce")
@_guard_cli_command
def reduce_command(
    failure_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--out", help="New reduced evidence directory")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit reduction result JSON")] = False,
) -> None:
    """Reduce a stored stable numerical or profile failure predicate."""

    from upgrade_guard.reduce.session import reduce_failure_directory

    try:
        result = reduce_failure_directory(failure_directory, output)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Reduced {result['failure_code']} evidence: {output}")


@app.command("doctor")
@_guard_cli_command
def doctor_command(
    json_output: bool = typer.Option(False, "--json", help="Emit versioned machine JSON."),
) -> None:
    """Inspect the real host without importing TensorRT or CUDA on the control plane."""

    result = run_doctor()
    if json_output:
        typer.echo(doctor_json(result))
    else:
        typer.echo(f"Host preflight: {result.outcome}")
        typer.echo(
            f"Host: {result.host_os} {result.host_architecture}; "
            f"Docker: {'available' if result.docker.available else 'unavailable'}; "
            f"NVIDIA GPUs: {len(result.gpus)}"
        )
        for issue in result.issues:
            typer.echo(f"[{issue.code}] {issue.message}")
    exit_code = doctor_exit_code(result)
    if exit_code:
        raise typer.Exit(code=exit_code)


@matrix_app.command("lock")
@_guard_cli_command
def matrix_lock_command(
    matrix: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--out", help="Destination immutable JSON lock."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete lock as JSON."),
    ] = False,
) -> None:
    """Resolve exact images and probe both workers on one selected GPU."""

    destination = output or matrix.with_suffix(".lock.json")
    try:
        lock = MatrixLocker().lock(matrix, destination)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    except Exception:
        _fail(
            UpgradeGuardError("unexpected internal tool failure"),
            json_output=json_output,
        )
    if json_output:
        typer.echo(lock.model_dump_json(indent=2))
    else:
        typer.echo(f"Wrote immutable environment lock: {destination}")
        typer.echo(f"Lock SHA-256: {lock.lock_sha256}")


@matrix_app.command("verify")
@_guard_cli_command
def matrix_verify_command(
    lock_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the verified lock identity."),
    ] = False,
) -> None:
    """Re-probe the current host and exact workers and reject lock drift."""

    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
        from upgrade_guard.contracts.environment import MatrixLock

        lock = MatrixLock.model_validate(value)
        MatrixLocker().verify(lock)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        _fail(
            InvalidInputError("environment lock is invalid", details={"reason": str(error)}),
            json_output=json_output,
        )
    payload = {
        "schema_version": "upgradeguard.dev/matrix-verification/v1",
        "status": "passed",
        "lock_sha256": lock.lock_sha256,
        "gpu_uuid": lock.gpu_uuid,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Verified current environment against lock: {lock.lock_sha256}")


@dev_app.command("resolve-image")
@_guard_cli_command
def resolve_image_command(
    image_reference: Annotated[str, typer.Argument(help="Authored OCI image reference.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete resolved identity as JSON."),
    ] = False,
) -> None:
    """Resolve one image to its exact linux/amd64 manifest and config."""

    try:
        parts = parse_image_reference(image_reference)
        credentials = credentials_from_environment(parts.registry)
        image = RegistryClient(credentials=credentials).resolve_linux_amd64(image_reference).image
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    except Exception:
        _fail(
            UpgradeGuardError("unexpected internal tool failure"),
            json_output=json_output,
        )
    if json_output:
        typer.echo(image.model_dump_json(indent=2))
    else:
        typer.echo(f"Selected manifest: {image.manifest_digest}")
        typer.echo(f"Image configuration: {image.config_digest}")
        typer.echo(f"Canonical reference: {image.canonical_reference}")


@dev_app.command("qualify-core")
@_guard_cli_command
def qualify_core_command(
    qualification: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--out")],
    project_root: Annotated[Path, typer.Option("--project-root")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the internal tiny-transformer core used by the full orchestrator."""

    from upgrade_guard.qualification import QualificationRunner

    outcome = QualificationRunner(source_root=project_root).run(qualification, output)
    payload = {
        "schema_version": "upgradeguard.dev/qualification-core-command/v1",
        "directory": str(outcome.directory),
        "status": outcome.status,
        "failure_codes": [item.value for item in outcome.failure_codes],
    }
    typer.echo(
        json.dumps(payload, indent=2, sort_keys=True)
        if json_output
        else f"Core qualification {outcome.status}: {outcome.directory}"
    )
    decision_exit = _decision_exit_code(
        outcome.status,
        [item.value for item in outcome.failure_codes],
    )
    if decision_exit:
        raise typer.Exit(code=decision_exit)


@dev_app.command("lock-reference")
@_guard_cli_command
def lock_reference_command(
    image_reference: Annotated[str, typer.Argument()],
    output: Annotated[Path, typer.Option("--out")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resolve and probe the independent immutable CPU reference image."""

    from upgrade_guard.reference_environment import ReferenceEnvironmentLocker

    lock = ReferenceEnvironmentLocker().lock(image_reference, output)
    if json_output:
        typer.echo(lock.model_dump_json(indent=2))
    else:
        typer.echo(f"Wrote independent reference lock: {output}")


@reproduce_app.command("verify")
@_guard_cli_command
def reproduce_verify_command(
    bundle: Annotated[Path, typer.Argument(exists=True, readable=True)],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit verified identity as JSON."),
    ] = False,
) -> None:
    """Verify every bundle path, size, type, hash, and manifest field."""

    try:
        verified = verify_bundle(bundle)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    payload = {
        "schema_version": "upgradeguard.dev/bundle-verification/v1",
        "bundle_id": verified.manifest.id,
        "manifest_sha256": verified.manifest.manifest_sha256,
        "source_code_present": verified.source_code_present,
        "engine_present": verified.engine_present,
        "file_count": len(verified.observed_files),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Verified reproduction bundle: {verified.manifest.id}")
        typer.echo(f"Manifest SHA-256: {verified.manifest.manifest_sha256}")


@reproduce_app.command("run")
@_guard_cli_command
def reproduce_run_command(
    bundle: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--out", help="Replay output directory.")],
    gpu_uuid: Annotated[
        str | None,
        typer.Option("--gpu", help="Selected replay GPU UUID; required when multiple are visible."),
    ] = None,
    local_registry: Annotated[
        str,
        typer.Option(
            "--local-registry",
            help="Operator-owned localhost registry for the rebuilt worker.",
        ),
    ] = "127.0.0.1:5500",
    trust_included_engine: Annotated[
        bool,
        typer.Option(
            "--trust-included-engine",
            help="Acknowledge trusted executable engine content.",
        ),
    ] = False,
    trust_source_code: Annotated[
        bool,
        typer.Option(
            "--trust-source-code",
            help="Acknowledge reviewed source before compilation.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify and execute a typed replay without invoking bundled reproduce.sh."""

    try:
        from upgrade_guard.reproduce.builder import LocalDockerReplayImageBuilder

        result = execute_replay(
            bundle,
            output,
            trust_source_code=trust_source_code,
            trust_included_engine=trust_included_engine,
            replay_target=observe_replay_target(gpu_uuid),
            image_builder=LocalDockerReplayImageBuilder(local_registry),
        )
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    payload = asdict(result)
    payload["worker_build_log"] = result.worker_build_log.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Reproduced {result.expected_failure_code}: {result.bundle_id}")
        typer.echo(f"Evidence: {output / 'replay-result.json'}")


@app.command("report")
@_guard_cli_command
def report_command(
    run_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    report_format: Annotated[
        str,
        typer.Option("--format", help="One of text, json, or html."),
    ] = "text",
) -> None:
    """Render a static report from a stored typed report model."""

    source = run_directory / "report-model.json"
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        report = ReportModel.model_validate(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        _fail(
            InvalidInputError(
                "run directory does not contain a valid report-model.json",
                details={"reason": str(error)},
            ),
            json_output=False,
        )
    renderers = {"text": render_text, "json": render_json, "html": render_html}
    renderer = renderers.get(report_format)
    if renderer is None:
        _fail(
            InvalidInputError("report format must be text, json, or html"),
            json_output=False,
        )
    typer.echo(renderer(report), nl=False)


def _decision_exit_code(status: str, raw_codes: list[str]) -> int:
    """Preserve the stable exit contract for one stored qualification decision."""

    try:
        failure_codes = tuple(FailureCode(item) for item in raw_codes)
    except ValueError as error:
        raise InvalidInputError("qualification decision has an unknown failure code") from error
    if any(exit_code_for_failure(code) is ExitCode.INVALID_INPUT for code in failure_codes):
        return int(ExitCode.INVALID_INPUT)
    mapping = {
        "passed": ExitCode.SUCCESS,
        "failed": ExitCode.QUALIFICATION_FAILED,
        "unsupported": ExitCode.UNSUPPORTED,
        "inconclusive": ExitCode.INFRASTRUCTURE_INVALID,
        "infrastructure_invalid": ExitCode.INFRASTRUCTURE_INVALID,
    }
    try:
        return int(mapping[status])
    except KeyError as error:
        raise InvalidInputError("qualification decision status is invalid") from error


def _fail(error: UpgradeGuardError, *, json_output: bool) -> NoReturn:
    payload = {
        "schema_version": "upgradeguard.dev/error/v1",
        "error_code": error.error_code,
        "message": error.message,
        "details": error.details,
    }
    if json_output:
        typer.echo(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"{error.error_code}: {error.message}", err=True)
        review_inventory = error.details.get("review_inventory")
        if isinstance(review_inventory, dict):
            typer.echo("Source review inventory:", err=True)
            typer.echo(
                json.dumps(
                    review_inventory,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ),
                err=True,
            )
    raise typer.Exit(code=int(error.exit_code))


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":
    main()
