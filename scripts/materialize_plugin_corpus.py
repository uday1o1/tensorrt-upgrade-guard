"""Materialize the deterministic plugin micrograph corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from upgrade_guard.corpus.plugin import materialize_plugin_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = materialize_plugin_corpus(arguments.output)
    print(f"materialized {len(result['artifacts'])} plugin artifacts")


if __name__ == "__main__":
    main()
