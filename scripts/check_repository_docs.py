"""Check internal links in public Markdown documentation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _check_markdown() -> list[str]:
    failures: list[str] = []
    for document in sorted(
        (*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md"), *ROOT.glob("reports/**/*.md"))
    ):
        text = document.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            value = target.strip("<>").split("#", maxsplit=1)[0]
            if not value or "://" in value or value.startswith("mailto:"):
                continue
            resolved = (document.parent / value).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)}: broken link {target}")
    return failures


def main() -> None:
    failures = _check_markdown()
    if failures:
        raise SystemExit("\n".join(failures))
    print("Documentation links are valid.")


if __name__ == "__main__":
    main()
