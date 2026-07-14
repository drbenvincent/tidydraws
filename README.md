<div align="center">
  <a href="https://github.com/drbenvincent/tidydraws"><img width="40%" src="https://raw.githubusercontent.com/drbenvincent/tidydraws/main/docs/assets/logo.jpg"></a>
</div>

----

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/tidydraws)](https://pypi.org/project/tidydraws/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://pypi.org/project/tidydraws/)

[![CI](https://github.com/drbenvincent/tidydraws/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/drbenvincent/tidydraws/actions/workflows/ci.yml) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[![GitHub Stars](https://img.shields.io/github/stars/drbenvincent/tidydraws?style=flat)](https://github.com/drbenvincent/tidydraws) [![Downloads](https://static.pepy.tech/badge/tidydraws)](https://pepy.tech/project/tidydraws) [![PyPI Downloads](https://img.shields.io/pypi/dm/tidydraws)](https://pypi.org/project/tidydraws/)
</div>

<!-- docs-start -->

# tidydraws

> A tidybayes-inspired data layer for declarative Bayesian visualisation in Python

`tidydraws` turns MCMC output (ArviZ) into tidy Polars frames that are ready to plot — one `.to_pandas()` away from any ggplot-like backend. It does no plotting itself. Three functions, three spaces:

| Function | Space | Plot archetype |
| --- | --- | --- |
| [`parameter_draws()`](https://drbenvincent.github.io/tidydraws/docs/examples/parameter_draws.html) | parameter | density, forest, scatter |
| [`prediction_draws()`](https://drbenvincent.github.io/tidydraws/docs/examples/prediction_draws.html) | prediction | ribbon + line, fit + data |
| [`compare_draws()`](https://drbenvincent.github.io/tidydraws/docs/examples/compare_draws.html) | comparison | prior vs posterior, intervals |

![tidydraws example](https://raw.githubusercontent.com/drbenvincent/tidydraws/main/docs/assets/index-plot.png)

## Install

With uv:
```bash
uv add tidydraws
```

With pip:
```bash
pip install tidydraws
```

If you want the latest functionality merged into main but not yet released, install directly from GitHub:

```bash
pip install git+https://github.com/drbenvincent/tidydraws.git
```

Or with uv:

```bash
uv add git+https://github.com/drbenvincent/tidydraws.git
```

## Why tidydraws?

Plotting MCMC output in Python means manually slicing xarray dimensions, iterating groups, and aligning coordinates — imperative, verbose, error-prone. R's [`tidybayes`](https://github.com/mjskay/tidybayes) solved this with a data layer that respects parameter space vs prediction space. `tidydraws` brings that to Python on Polars.

> **Backend-agnostic:** `tidydraws` returns Polars `DataFrame`s. Call `.to_pandas()` to bridge to lets-plot, plotnine, or any library that takes pandas. See the [examples](https://drbenvincent.github.io/tidydraws/docs/examples/parameter_draws.html) for both lets-plot and plotnine versions.

---

*Inspired by [tidybayes](https://github.com/mjskay/tidybayes) for R.*
