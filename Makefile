# Makefile for tidydraws development workflow

.PHONY: help install test lint type-check docs docs-preview cleandocs build clean

# Help target to show available commands
help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies with uv"
	@echo "  test          - Run all tests"
	@echo "  lint          - Run code linting with ruff"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  docs          - Build the documentation site (great-docs)"
	@echo "  docs-preview  - Build and serve the docs locally with live reload"
	@echo "  cleandocs     - Remove the ephemeral great-docs/ build directory"
	@echo "  clean         - Clean build artifacts"

# Install dependencies
install:
	uv sync

# Run tests
test:
	uv run pytest

# Lint code with ruff
lint:
	uv run ruff check .

# Type check with mypy
type-check:
	uv run mypy .

# Build the documentation site into the ephemeral great-docs/_site/ directory
docs:
	uv run great-docs build

# Build and serve the docs locally with live reload (http://localhost:3000)
docs-preview:
	uv run great-docs preview

# Remove the ephemeral great-docs build directory (regenerated on every build)
cleandocs:
	rm -rf great-docs

# Clean build artifacts
clean: cleandocs
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +