.PHONY: test lint typecheck

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run pyright
