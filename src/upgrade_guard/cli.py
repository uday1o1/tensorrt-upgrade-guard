"""Public command-line interface."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from upgrade_guard.corpus.materialize import materialize_corpus
from upgrade_guard.doctor import doctor_exit_code, doctor_json, run_doctor
from upgrade_guard.errors import InvalidInputError, UpgradeGuardError
from upgrade_guard.matrix.digest import (
    RegistryClient,
    credentials_from_environment,
    parse_image_reference,
)
from upgrade_guard.matrix.lock import MatrixLocker
from upgrade_guard.qualification import QualificationRunner, compare_stored_run
from upgrade_guard.reduce.session import reduce_failure_directory
from upgrade_guard.report.html_report import render_html
from upgrade_guard.report.json_report import render_json
from upgrade_guard.report.model import ReportModel
from upgrade_guard.report.text import render_text
from upgrade_guard.reproduce.run import prepare_replay, require_gpu_for_replay
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


@corpus_app.command("materialize")
def corpus_materialize_command(
    recipe: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--out", help="New immutable corpus directory")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit the corpus lock")] = False,
) -> None:
    """Generate frozen models and inputs, then execute their CPU references."""

    try:
        lock = materialize_corpus(recipe, output)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    if json_output:
        typer.echo(lock.model_dump_json(indent=2))
    else:
        typer.echo(f"Materialized immutable corpus: {output}")
        typer.echo(f"Artifacts: {len(lock.artifacts)}")


@app.command("qualify")
def qualify_command(
    qualification: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--out", help="New qualification result directory")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit the result identity")] = False,
) -> None:
    """Run the frozen correctness, determinism, memory, and performance matrix."""

    try:
        outcome = QualificationRunner().run(qualification, output)
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
    if outcome.status in {"inconclusive", "infrastructure_invalid"}:
        raise typer.Exit(code=4)


@app.command("compare")
def compare_command(
    run_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine JSON")] = False,
) -> None:
    """Validate and display a completed qualification decision."""

    try:
        summary = compare_stored_run(run_directory)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Qualification status: {summary['status']}")
        typer.echo(f"Failure codes: {', '.join(summary['failure_codes']) or 'none'}")


@app.command("reduce")
def reduce_command(
    failure_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--out", help="New reduced evidence directory")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit reduction result JSON")] = False,
) -> None:
    """Reduce a stored stable numerical or profile failure predicate."""

    try:
        result = reduce_failure_directory(failure_directory, output)
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Reduced {result['failure_code']} evidence: {output}")


@app.command("doctor")
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


@dev_app.command("resolve-image")
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


@reproduce_app.command("verify")
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
def reproduce_run_command(
    bundle: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--out", help="Replay output directory.")],
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
    """Prepare a typed replay without ever invoking bundled reproduce.sh."""

    del output
    try:
        plan = prepare_replay(
            bundle,
            trust_source_code=trust_source_code,
            trust_included_engine=trust_included_engine,
        )
        require_gpu_for_replay()
    except UpgradeGuardError as error:
        _fail(error, json_output=json_output)
    typer.echo(json.dumps(asdict(plan), indent=2))


@app.command("report")
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
    raise typer.Exit(code=int(error.exit_code))


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":
    main()
