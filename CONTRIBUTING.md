# Contributing to tidydraws

## Set up a checkout

Install [uv](https://docs.astral.sh/uv/) (CI uses 0.13.7) and Python 3.12. Clone your fork and run:

```bash
uv sync --locked --group docs
uv run prek install
```

The default `dev` group includes the `test`, `lint`, and `build` groups plus bumpver. The optional `docs` group adds Great Docs, ipykernel, PyMC and plotting libraries. Consumer extras remain `letsplot`, `plotnine`, `altair`, and `all`; development tools are dependency groups rather than a published `dev` extra. For core tests only, use `uv sync --locked --no-default-groups --group test` and `uv run --no-sync pytest`.

For each new session, run `uv sync --locked --group docs`. A stale lock must fail rather than be silently regenerated. When changing dependencies intentionally, use `uv lock --upgrade-package PACKAGE` (or `uv lock --upgrade`), inspect the lock diff, and sync again. Keep ArviZ unpinned as agreed in #26; runtime dependency reductions belong to #38.

## Make a change

Create a branch from `main`, add a reproducer or meaningful regression test, and run:

```bash
make precommit
make test
make type-check
make build
```

`make precommit` is required before every commit. The local Ruff and mypy hooks run the locked project tools with real dependencies; mypy checks the whole package even for configuration-only changes. Workflow changes also run zizmor, whose exit status blocks findings. The remaining remote hooks cover whitespace, YAML/TOML, merge markers, file size and license headers. To refresh those revisions, run `uv run prek autoupdate`, review the diff and rerun `make precommit`; Dependabot updates Actions and uv dependencies, not remote hook revisions.

CI explicitly tests Linux on Python 3.11–3.14 and macOS/Windows on Python 3.12. The locked 3.11 leg asserts real legacy ArviZ `InferenceData`; modern legs assert ArviZ 1+ and DataTree. Each leg checks dependency consistency and the suite's 80% coverage floor, and uploads coverage XML. The stable `Tests` aggregate fails if any matrix leg fails. `Lock freshness`, `Lint and types`, `Workflow security`, `Build Docs`, and `Artifact validation` are the other intended required statuses.

`make build` builds both distributions, checks metadata with twine, then installs each into a separate clean uv environment and runs outside the checkout with isolated Python. It checks the installed module path, distribution/package version, four public exports, extraction/summary row counts and each plotting extra. Only validated files are copied to `.scratch/dist/`. No PyPI credentials or release are needed.

## Documentation

Install the [Quarto CLI](https://quarto.org/docs/get-started/) separately (CI uses 1.9.38), then run:

```bash
make docs
make docs-preview
```

Both commands register `python3` in a checkout-owned `.scratch/jupyter/` kernelspec, verify its `argv[0]` matches the project interpreter, and set `JUPYTER_PATH` and `QUARTO_PYTHON` for the build. They do not overwrite a user kernel. Re-running after an environment replacement refreshes the registration. `make docs` verifies nonempty `llms.txt`, `llms-full.txt` and `skill.md` outputs.

Great Docs builds from `great-docs.yml`, `index.qmd`, docstrings and `docs/examples/` into the disposable `great-docs/_site/`. Edit those sources, never generated files. CI executes all examples from a fresh checkout with no restored execution cache or frozen output; local execution caches can be cleared with `make cleandocs`. A manual docs run deploys only on `main` with `deploy: true`; trusted pushes to `main` deploy automatically. A manual `deploy: false` run builds only.

The `.agents/skills/` Great Docs skills are pinned by `skills-lock.json`. Refresh them with `npx skills add https://posit-dev.github.io/great-docs/`, review the changes, and validate the build.

## Dependency maintenance

Dependabot opens weekly grouped Actions updates and at most three uv dependency PRs, with development minor/patch updates grouped. The three-day cooldown applies to version updates; security updates remain immediate. Review scientific-stack upgrades individually when their compatibility is uncertain. The maintainer (@drbenvincent) owns dependency alerts and weekly maintenance failures.

The weekly/manual `Weekly maintenance` workflow resolves fresh dependencies in disposable uv environments without rewriting `uv.lock`, tests the declared runtime lower bounds from `.github/constraints/minimum.txt`, and crawls the published docs with bounded retries. The only fragment exemptions are four API fragments emitted without matching IDs by Great Docs 0.15; their destination pages are still checked. Unbounded dependencies resolve normally; the lower-bound constraints must track declared floors. Latest-resolution checks also import representative docs/plotting tools. Reproduce locally with:

```bash
uv run python scripts/check_compatibility.py
uv run python scripts/check_compatibility.py --lowest
uv run python scripts/check_links.py
```

A failed run creates or updates one open `Weekly maintenance failure` issue with the run link and failed jobs. The maintainer investigates, links a repair PR, reruns the workflow, and closes the tracking issue after success. Successful unchanged runs stay quiet. This notification uses the workflow token, including for manually triggered runs on `main`.

## Releasing

An authorized maintainer releases from a clean, up-to-date `main` checkout with one command:

```bash
make release-patch
# or: make release-minor / make release-major
```

The command verifies branch, cleanliness, remote alignment and lock freshness, then bumpver updates the package and configured version. Its pre-commit hook refreshes the lock and runs hooks, tests, artifact validation and docs before the bump commit, tag and push. Failure stops the release with an error; inspect any local version edits before retrying. The published 0.4.3 tag is retained; `main`'s version fields are reconciled to 0.4.3 without retagging.

The numeric tag push triggers `publish.yml`. It verifies tag/version agreement and ancestry on `main`, reruns release validation on that exact commit, and builds/checks the distributions. OIDC trusted publishing uploads them to TestPyPI. Verification retrieves the exact version's wheel from TestPyPI, requires its hash to equal the validated wheel, and tests its installation; dependencies come separately from PyPI. Only successful verification allows automatic production PyPI upload, followed by the GitHub Release with generated notes. No approval click, release PR, manual version edit or extra release command is part of a successful run.

Per-tag concurrency serializes publication, with cancellation disabled. `skip-existing` supports retries and TestPyPI verification rejects a conflicting existing wheel. A manual run on the same tag retries the pipeline; PRs and manual branch runs only validate artifacts. There is no downstream release-event trigger to race the tag workflow.

### Repository settings and activation

GitHub settings are separate from YAML. The audit on 4 November 2026 found no required status checks on `main`, zero required approvals, administrator bypass enabled, and a `release` environment with no manual reviewers or branch/tag restrictions. The existing `protect-release-tags` ruleset protects numeric tags with a narrowly configured maintainer user bypass; preserve it. Do not claim merge enforcement until the new statuses have run and are configured.

After a representative PR passes, the maintainer must require the stable statuses listed above, with an up-to-date branch requirement. Keep zero mandatory approvals for the solo-maintainer workflow and the deliberate administrator bypass needed to push the validated release bump directly to `main`. Ordinary contributors must use PRs and pass required checks. Verify a deliberately failing check prevents an ordinary merge. Restrict the `release` environment to numeric version tags with no human reviewers, preserve the trusted publisher's `publish.yml`/`release` identity, and inspect the live API/UI after activation. Test the automatic flow with the release harness and a non-production rehearsal before relying on it. Never reuse/move an existing version tag.

Dependency alerts, automated security fixes and private vulnerability reporting were enabled and verified through the live API during this change. Required-check activation, release-environment tag restrictions, representative remote CI, successful manual maintenance runs and actual Dependabot PRs remain rollout evidence to collect after these workflows are published. A passing local build does not establish those settings.
