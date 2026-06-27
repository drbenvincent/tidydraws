# tidydraws

> A tidybayes-inspired data layer for declarative Bayesian visualisation in Python

`tidydraws` turns MCMC output (ArviZ `DataTree`) into tidy Polars frames that are ready to plot — one `.to_pandas()` away from any ggplot-like backend. It does no plotting itself. Three functions, three spaces:

| Function | Space | Plot archetype |
| --- | --- | --- |
| [`parameter_draws()`](https://drbenvincent.github.io/tidy-draws/docs/examples/parameter_draws.html) | parameter | density, forest, scatter |
| [`prediction_draws()`](https://drbenvincent.github.io/tidy-draws/docs/examples/prediction_draws.html) | prediction | ribbon + line, fit + data |
| [`compare_draws()`](https://drbenvincent.github.io/tidy-draws/docs/examples/compare_draws.html) | comparison | prior vs posterior, intervals |

![tidydraws example](index-plot.png)

## Install

With uv:
```bash
uv add tidydraws
```

With pip:
```bash
pip install tidydraws
```

## Why tidydraws?

Plotting MCMC output in Python means manually slicing xarray dimensions, iterating groups, and aligning coordinates — imperative, verbose, error-prone. R's [`tidybayes`](https://github.com/TuringLang/tidybayes) solved this with a data layer that respects parameter space vs prediction space. `tidydraws` brings that to Python on Polars.

| Task | ArviZ (manual) | tidydraws |
| --- | --- | --- |
| Extract `beta[groups]` | `dt.posterior["beta"].values` + reshape | `parameter_draws(dt, "beta[groups]")` |
| Compare prior / posterior | concat two arrays by hand | `compare_draws(dt, "beta[groups]")` |
| Join predictions to covariates | manual merge on obs index | `prediction_draws(dt, var_name="mu")` |
| Subset before plotting | slice after materialising | `.filter(...)` on the returned frame |

> **Backend-agnostic:** `tidydraws` returns Polars `DataFrame`s. Call `.to_pandas()` to bridge to lets-plot, plotnine, or any library that takes pandas. See the [examples](https://drbenvincent.github.io/tidy-draws/docs/examples/parameter_draws.html) for both lets-plot and plotnine versions.

---

*Inspired by [tidybayes](https://github.com/TuringLang/tidybayes) for R.*