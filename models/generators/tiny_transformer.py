#!/usr/bin/env python3
"""Generate one project-owned frozen mini-transformer artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from upgrade_guard.corpus.generators import generate_tiny_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    arguments = parser.parse_args()
    print(generate_tiny_transformer(arguments.output, precision=arguments.precision))


if __name__ == "__main__":
    main()
