# Contributing to tidydraws

Welcome to the tidydraws project! We're excited that you're interested in contributing.
This document explains how to set up your development environment and contribute to this project.

## Development Environment Setup

tidydraws uses `uv` for managing dependencies. Make sure you have it installed before proceeding.

### Prerequisites
- Python 3.12 or higher (as specified in `pyproject.toml`)
- [`uv`](https://github.com/astral-sh/uv) package manager
- [`great-docs`](https://posit-dev.github.io/great-docs/) for documentation building

### Getting Started

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tidydraws.git
   cd tidydraws
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks for code quality checks (optional but recommended):
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

2. Install pre-commit hooks for code quality checks (optional but recommended):
   ```bash
   uv run pre-commit install
   ```

3. Run tests with:
   ```bash
   uv run pytest
   ```

4. Lint code with ruff:
   ```bash
   uv run ruff check .
   ```

5. Type check with mypy:
   ```bash
   uv run mypy .
   ```

## Testing 
Run tests with:
```bash
uv run pytest
```

## Code Linting and Type Checking
```bash
uv run ruff check .
uv run mypy .
```

## Documentation

We use [Great Docs](https://posit-dev.github.io/great-docs/) (which wraps Quarto) for the documentation site. A single `great-docs.yml` at the repo root controls the build: it wires up the API reference, the narrative tutorials under `docs/user_guide/`, and the worked examples under `docs/examples/`.

### Building Documentation Locally

1. Build the site (output goes to the ephemeral `great-docs/_site/` directory):
   ```bash
   uv run great-docs build      # or: make docs
   ```

2. Preview locally with live reload at http://localhost:3000:
   ```bash
   uv run great-docs preview    # or: make docs-preview
   ```

3. See what API symbols Great Docs can discover:
   ```bash
   uv run great-docs scan --verbose
   ```

The `great-docs/` directory is **ephemeral** — it is regenerated on every build and is git-ignored. Never edit files inside it directly; change `great-docs.yml` or the source `.qmd` files under `docs/` instead. To clear it, run `make cleandocs`.

### Agent skills for the docs

This repo ships the [Great Docs Agent Skills](https://posit-dev.github.io/great-docs/) under `.agents/skills/` (`great-docs`, `configure-site`, `write-user-guide`, `revise-docstrings`, `author-skills`), pinned via `skills-lock.json`. AI coding agents working on the docs pick these up automatically; you do not need to install anything. To refresh them against upstream, run `npx skills add https://posit-dev.github.io/great-docs/` from the repo root and commit the result.

## Making Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the project's coding style and maintain compatibility.

3. Add tests for your changes where appropriate.

4. Run all checks before committing:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy .
   ```

5. Commit your changes with a descriptive message.

6. Push to your fork and create a pull request.

## Pull Request Guidelines

- Reference relevant issues in your PR description 
- Ensure all tests pass
- Add or update documentation as needed
- Keep changes focused and atomic