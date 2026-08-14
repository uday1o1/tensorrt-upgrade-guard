"""Classify one reduced G2 executable result for typed clean replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from upgrade_guard.errors import FailureCode


def main() -> None:
    """Run the reviewed executable and emit a code only after observing the predicate."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--x-value", type=float, required=True)
    parser.add_argument("--residual-value", type=float, required=True)
    parser.add_argument("--gamma-value", type=float, required=True)
    arguments = parser.parse_args()
    command = (
        str(arguments.executable),
        "--pair-index",
        "0",
        "--rows",
        str(arguments.rows),
        "--hidden",
        str(arguments.hidden),
        "--x-value",
        format(arguments.x_value, ".9g"),
        "--residual-value",
        format(arguments.residual_value, ".9g"),
        "--gamma-value",
        format(arguments.gamma_value, ".9g"),
        "--only-g2",
    )
    completed = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        shell=False,
    )
    try:
        payload = json.loads(completed.stdout)
        observed = payload["G2"]
        reproduced = (
            completed.returncode == 0
            and observed["detected"] is True
            and observed["control"] == "passed"
            and int(observed.get("rows", arguments.rows)) == arguments.rows
            and int(observed.get("hidden", arguments.hidden)) == arguments.hidden
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        reproduced = False
        payload = {"unclassified_stdout": completed.stdout}
    result = {
        "schema_version": "upgradeguard.dev/g2-replay-observation/v1",
        "status": "failed" if reproduced else "passed",
        "failure_code": FailureCode.NUMERICAL_REGRESSION.value if reproduced else None,
        "returncode": completed.returncode,
        "observation": payload,
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
