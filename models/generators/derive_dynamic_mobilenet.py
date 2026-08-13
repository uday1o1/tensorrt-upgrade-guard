#!/usr/bin/env python3
"""Download, verify, and derive the pinned dynamic MobileNet artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from upgrade_guard.corpus.mobilenet import derive_dynamic_mobilenet, download_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    arguments = parser.parse_args()
    source = arguments.source or arguments.output.with_name("mobilenetv3-small-075-source.onnx")
    if not source.exists():
        download_source(source)
    print(derive_dynamic_mobilenet(source, arguments.output))


if __name__ == "__main__":
    main()
