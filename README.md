# tidydraws

> A tidybayes-inspired data layer for declarative Bayesian visualisation in Python

---

## Quick Example

```python
import arviz as az
import tidydraws as td
from lets_plot import *

# Load an ArviZ 1.0+ xarray.DataTree (e.g., from PyMC sampling)
dt = az.from_netcdf("model.nc")

# ── Parameter space: tidy posterior draws ────────────────────────
beta_draws = td.spread_draws(dt, "beta[groups]", "intercept[groups]")
# → LazyFrame with columns: chain, draw, groups, beta, intercept
# → 4 chains × 1000 draws × 4 groups = 16,000 tidy rows

# ── Prediction space: join predictive draws to covariates ────────
pred_draws = td.add_epred_draws(dt, newdata=None, var_name="mu")
# → LazyFrame with columns: chain, draw, obs_ind, x, group, mu
# → Parameters NOT included — clean separation of semantic levels

# ── Plot with lets-plot ──────────────────────────────────────────
(
    ggplot()
    # Posterior parameter forest plot
    + td.stat_pointinterval(beta_draws, x="groups", y="beta", prob=0.89)
    
    # Posterior predictive fit — lazy filtering before collection
    + td.stat_hdi(pred_draws.filter(pl.col("group") == 0).head(100), 
                   x="x", y="mu", group="group", alpha=0.2)
    + theme_classic()
)
```

## Why tidydraws?

Plotting MCMC output in Python requires manual wrangling of `arviz.InferenceData` objects — slicing xarray dimensions, iterating over groups, managing coordinate alignment. This is imperative, boilerplate-heavy, and error-prone.

R's [`tidybayes`](https://github.com/TuringLang/tidybayes) solved this elegantly with a data transformation layer that respects the semantics of parameter space vs. prediction space. `tidydraws` brings the same philosophy to Python — using **Polars LazyFrames** so you only pay the materialisation cost when you're ready.

| Problem | tidydraws Solution |
| --- | --- |
| Manually flatten xarray Datasets | `spread_draws()` returns a clean, per-dimension frame |
| Denormalised tables duplicating coefficients across observations | Parameter space (`beta`) and prediction space (`mu`) kept separate |
| Eager evaluation loads everything into memory | LazyFrames let you filter before `.collect()` — only load what you need |

## Core Functions

- **`spread_draws(dt, "var[dim1, dim2]", ...)`** — Extract posterior draws into a tidy Polars LazyFrame
- **`add_epred_draws(dt, newdata=..., var_name="mu")`** — Join prediction draws to a covariate grid
- **`spread_draws_compare(dt, "var[groups]", groups=["posterior", "prior"])`** — Stack draws from multiple groups (e.g., prior vs. posterior)

All functions always return `pl.LazyFrame`. Call `.collect()` when you're ready to plot or inspect.

## Getting Started

```bash
uv init tidydraws && cd tidydraws
uv add polars arviz lets-plot
# …then copy this package into the project
```

Or install from source:

```bash
git clone https://github.com/YOURUSER/tidydraws.git
cd tidydraws
pip install -e .
```

## Documentation

See the [user guide](./docs/user_guide/) for tutorials, worked examples, and migration help.

| Guide | Description |
| --- | --- |
| [01-quickstart](./docs/user_guide/01-quickstart.qmd) | 5-minute intro to `spread_draws()` and `add_epred_draws()` |
| [02-parameter-space](./docs/user_guide/02-parameter-space.qmd) | Forest plots, ridge plots with `stat_pointinterval()` |
| [03-prediction-space](./docs/user_guide/03-prediction-space.qmd) | Posterior predictive fits, ribbons, credible intervals |
| [04-migration-from-arviz](./docs/user_guide/04-migration-from-arviz.qmd) | Replacing ArviZ's imperative approach with tidydraws |

## Installation Options

```bash
# Core data layer only (data functions work without any plotting library)
uv add tidydraws

# With lets-plot backend
uv add "tidydraws[letsplot]"

# With plotnine backend (ggplot2-familiar syntax)
uv add "tidydraws[plotnine]"

# Both backends
uv add "tidydraws[all]"
```

## Status

**v0.1.0a0** — Data layer in development. `spread_draws()`, `add_epred_draws()`, and core utilities are underway. Documentation is the primary deliverable.

---

*Inspired by [tidybayes](https://github.com/TuringLang/tidybayes) for R.*
