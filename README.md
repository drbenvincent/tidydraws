# tidydraws

> A tidybayes-inspired data layer for declarative Bayesian visualisation in Python

---

## Quick Example

```python
import numpy as np
import arviz as az
import polars as pl
import tidydraws as td
from xarray import DataArray
from lets_plot import *
LetsPlot.setup_html()

# A small synthetic posterior: 4 chains × 500 draws × 4 groups
chains, draws, n_groups = 4, 500, 4
beta = np.random.normal(1.0, 0.2, (chains, draws, n_groups))
dt = az.from_dict(
    {"posterior": {
        "beta": DataArray(beta, dims=["chain", "draw", "groups"]),
    }},
    coords={"groups": [f"g{i}" for i in range(n_groups)]},
    dims={"beta": ["groups"]},
)

# ── Parameter space: tidy posterior draws ────────────────────────
beta_draws = td.spread_draws(dt, "beta[groups]")
# → LazyFrame with columns: chain, draw, groups, beta

# ── Summarise and plot with lets-plot ───────────────────────────
summary = (
    beta_draws.group_by("groups")
    .agg(
        pl.col("beta").quantile(0.055).alias("lower"),
        pl.col("beta").median().alias("median"),
        pl.col("beta").quantile(0.945).alias("upper"),
    )
    .sort("groups")
    .collect()
)

(
    ggplot(summary.to_pandas(), aes(x="groups", y="median"))
    + geom_pointrange(aes(ymin="lower", ymax="upper"), size=0.8)
    + geom_hline(yintercept=0, linetype="dashed", color="#888888")
    + labs(x="group", y="beta", title="Posterior beta by group (89% CrI)")
    + theme_minimal()
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
| [01-quickstart](./docs/user_guide/01-quickstart.qmd) | 5-minute intro: extract tidy draws and plot with lets-plot |
| [02-parameter-space](./docs/user_guide/02-parameter-space.qmd) | Forest plots, density plots, prior vs. posterior comparison |
| [03-prediction-space](./docs/user_guide/03-prediction-space.qmd) | Posterior predictive fits, ribbons, credible intervals |
| [04-migration-from-arviz](./docs/user_guide/04-migration-from-arviz.qmd) | Replacing ArviZ's imperative approach with tidydraws |
| [05-backends](./docs/user_guide/05-backends.qmd) | The same plots rendered with lets-plot and plotnine |

> The rendered docs site is at <https://drbenvincent.github.io/tidy-draws/>.

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
