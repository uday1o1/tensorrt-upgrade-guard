.PHONY: sync schemas schema-check docs-check format lint typecheck test coverage verify

sync:
	uv sync --frozen

schemas:
	uv run --frozen python scripts/generate_schemas.py

schema-check: schemas
	git diff --exit-code -- schemas

docs-check:
	uv run --frozen python scripts/check_repository_docs.py

format:
	uv run --frozen ruff format .

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

typecheck:
	uv run --frozen mypy

test:
	uv run --frozen pytest

coverage:
	uv run --frozen pytest --cov=upgrade_guard --cov-report=term-missing

verify: schema-check docs-check lint typecheck coverage
