# Makefile for tidydraws development workflow

.PHONY: help install test lint type-check precommit docs docs-preview cleandocs build clean release release-patch release-minor release-major

# Help target to show available commands
help:
	@echo "Available commands:"
	@echo "  install       - Install all dependencies (incl. dev) with uv"
	@echo "  test          - Run all tests"
	@echo "  lint          - Run code linting with ruff"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  precommit     - Run prek (pre-commit) hooks on all files"
	@echo "  docs          - Build the documentation site (great-docs)"
	@echo "  docs-preview  - Build and serve the docs locally with live reload"
	@echo "  cleandocs     - Remove the ephemeral great-docs/ build directory"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Releasing (admin only; see CONTRIBUTING.md > Releasing):"
	@echo "  release-patch  - Bump patch (0.4.0 -> 0.4.1) and cut a release"
	@echo "  release-minor  - Bump minor (0.4.0 -> 0.5.0) and cut a release"
	@echo "  release-major  - Bump major (0.4.0 -> 1.0.0) and cut a release"

# Install dependencies
install:
	uv sync --all-extras

# Run tests
test:
	uv run pytest

# Lint code with ruff
lint:
	uv run ruff check .

# Type check with mypy
type-check:
	uv run mypy tidydraws

# Run pre-commit hooks via prek
precommit:
	uv run prek run --all-files

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

# ── Release ─────────────────────────────────────────────────────────────────
# Admin only. Bumps the version (single source: tidydraws/__init__.py),
# re-derives uv.lock, commits, tags, and pushes. The tag push triggers both
# release.yml (→ GitHub Release) and publish.yml (→ TestPyPI → PyPI) in
# parallel. Note: GITHUB_TOKEN cannot trigger downstream workflows, so
# publish.yml listens directly for the tag push.
# Requires admin push rights to main (enforce_admins is off) and tag-push
# rights (tag protection rule on v*).
release-patch:
	uv run bumpver update --patch

release-minor:
	uv run bumpver update --minor

release-major:
	uv run bumpver update --major
