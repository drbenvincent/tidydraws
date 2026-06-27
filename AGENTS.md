# Agent Instructions for tidydraws

**Style:** No hard line breaks in prose.

tidydraws is a tidybayes-inspired data layer for Bayesian visualisation in Python. It extracts MCMC draws from ArviZ 1.0 DataTrees into tidy Polars DataFrames. This file holds the hard rules, signatures, and design rationale for agents working in the repo.

---

## Hard Rules (never violate)

1. **Always return `pl.DataFrame`** from `parameter_draws()`, `prediction_draws()`, and `compare_draws()` — eager, never `pl.LazyFrame`. The data is already materialised during the xarray→pandas conversion, so wrapping it lazy would be dishonest about cost and force every plotting call through a needless `.collect()`.
2. **String var specs only:** `"beta[groups, time]"` → `("beta", ["groups", "time"])`. No kwargs API.
3. **Fail loudly on missing data.** If `prediction_draws(newdata=None)` can't find the constant data group, raise a clear, actionable error.
4. **Support any group** (posterior, prior, custom) from day 1.
5. **Row counts are the primary test.** `parameter_draws(dt, "beta[groups]")` must return exactly `chains × draws × groups` rows, no duplication.
6. **No star imports in docs or examples.** Import the exact plotting functions used (for example, `from lets_plot import ggplot, aes, geom_line`) so rendered examples keep an explicit namespace contract.

---

## Environment: use `uv` (not venv or base)

This project is managed with `uv`. Do not use `python -m venv`, `pip install` into base, or conda. Always run code and tests through `uv`.

```bash
# One-time: create the project venv and install everything (incl. dev deps)
uv sync

# Run a command inside the project environment
uv run python script.py
uv run pytest
uv run pytest tests/test_parameter_draws.py

# Add a dependency (updates pyproject.toml + uv.lock)
uv add polars

# Add a dev-only dependency
uv add --dev pytest
```

Key point: prefix Python/pytest commands with `uv run` so they use the project's locked environment, not the system Python.

---

## Design Decisions (rationale)

- **String specs** mirror R's tidybayes and handle nested dims and mixed scalar/array vars compactly.
- **Eager `pl.DataFrame`, not LazyFrame.** The extraction path is `xarray.DataArray.to_dataframe()` (pandas, eager) → `pl.from_pandas()`; the data is fully in memory before any Polars object exists. A `LazyFrame` wrapper would buy nothing — predicate pushdown and streaming only help when the *source* is lazy (e.g. `scan_parquet`), which it isn't here — and would tax every plotting call with a `.collect()` step plus a `.to_pandas()` hop the backends require anyway. Eager frames are honest about the cost and a one-hop `.to_pandas()` from lets-plot/plotnine.
- **Fail loudly** on missing data to avoid silent misalignment bugs; give the user explicit guidance.
- **Cross-dim joins auto-join with a logged warning** (e.g. scalar `sigma` broadcast across `beta[groups]`) — common case "just works", transparently.
- **`compare_draws()` ships in v0.1** because prior vs. posterior comparison is fundamental.
- **ArviZ 1.0 DataTree only** — greenfield, no legacy `InferenceData` debt. Access groups via `.children[group].to_dataset()`.
- **Polars over pandas** for faster joins and a consistent tidy-frame type across the API surface.

---

### Key signatures

```python
def parameter_draws(
    dt, *var_specs, group="posterior", chain_dim="chain", draw_dim="draw"
) -> pl.DataFrame: ...


def prediction_draws(
    dt,
    newdata,
    var_name,
    idata_group="predictions",
    constant_data_group="predictions_constant_data",
    join_on="obs_ind",
) -> pl.DataFrame: ...


def compare_draws(
    dt, *var_specs, groups=["posterior", "prior"], group_name="source"
) -> pl.DataFrame: ...
```

### Core helpers (in `_extract.py`)

- `_parse_var_spec(spec)` → `("beta", ["groups"])`; raise on malformed specs (`"beta["`, `"beta]"`, `"beta[]"`).
- `_datatree_group_to_df(dt, group)` → `pl.DataFrame` with chain, draw, and all coord columns.
- `_align_dims(frames)` → inner-join same-dim frames; cross-join different-dim frames with a logged warning.
- `_coerce_to_dataframe(newdata)` → `pl.DataFrame` from `pl.DataFrame` / `pd.DataFrame`.

---

## Common Pitfalls

- Returning `pl.LazyFrame` or leaving a `.lazy()` / `.collect()` round-trip in the extraction path — the data layer is eager by design.
- Parser not splitting nested dims on `,` inside brackets.
- Confusing dimension names with coordinate names in xarray.
- Forgetting groups are accessed via `.children[group].to_dataset()`.
- Running `pytest`/`python` without `uv run` (wrong environment).

---
