"""Check public Markdown links and immutable GitHub Action references."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
USES = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s*#.*)?$")


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


def _check_actions() -> list[str]:
    failures: list[str] = []
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), start=1):
            if "uses:" in line and not USES.fullmatch(line):
                failures.append(
                    f"{workflow.relative_to(ROOT)}:{number}: action must use a 40-character SHA"
                )
    return failures


def main() -> None:
    failures = [*_check_markdown(), *_check_actions()]
    if failures:
        raise SystemExit("\n".join(failures))
    print("Documentation links and action pins are valid.")


if __name__ == "__main__":
    main()
