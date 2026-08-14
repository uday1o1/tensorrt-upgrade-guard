"""Resolve one retained domain failure before failed evidence publication."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from upgrade_guard.reduce.public_failure import DomainStep, process_public_failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--source-step",
        required=True,
        choices=("core-qualification", "plugin-matrix", "mobilenet-matrix"),
    )
    parser.add_argument("--core-corpus", required=True, type=Path)
    parser.add_argument("--plugin-corpus", required=True, type=Path)
    parser.add_argument("--mobilenet-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-registry", default="127.0.0.1:5500")
    arguments = parser.parse_args()
    disposition = process_public_failure(
        state=arguments.state,
        project=arguments.project,
        source_step=cast(DomainStep, arguments.source_step),
        core_corpus=arguments.core_corpus,
        plugin_corpus=arguments.plugin_corpus,
        mobilenet_corpus=arguments.mobilenet_corpus,
        output=arguments.output,
        registry=arguments.local_registry,
    )
    print(disposition.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
