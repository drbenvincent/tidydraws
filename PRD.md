# PRD: `tidydraws`

**A tidybayes-inspired data layer for declarative Bayesian visualisation in Python**

---

## 1. Problem Statement

Plotting MCMC output in Python today requires manual wrangling of `arviz.InferenceData`
objects: slicing xarray dimensions, iterating over groups, managing coordinate alignment.
This is imperative, boilerplate-heavy, and error-prone.

R's `tidybayes` package solved this problem elegantly by providing:

1. Functions that extract MCMC draws into properly-shaped tidy frames (`spread_draws`,
  `add_epred_draws`)
2. Custom ggplot2 stats and geoms for Bayesian-specific summaries (`stat_halfeye`,
  `stat_pointinterval`)

No equivalent exists in Python. The closest attempt — converting `InferenceData` to a flat
pandas DataFrame via a single `make_tidy()` function — produces a massively denormalised
table that conflates two semantically different levels (parameter space vs. data/prediction
space), duplicates coefficient values across every observation point, and hard-codes
variable names.

`tidydraws` solves this with a clean data layer (Polars DataFrames) and a thin
lets-plot integration layer.

---

## 2. Goals


| Goal   | Description                                                                                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **G1** | Provide `spread_draws()`: extract posterior parameter draws into a tidy Polars DataFrame, respecting named dimensions, with no unnecessary duplication |
| **G2** | Provide `add_epred_draws()`: join prediction draws to a covariate grid, without conflating parameter-space and data-space variables                    |
| **G3** | Use Polars DataFrames throughout; the data is already materialised during extraction, so eager frames are honest about cost and a single `.to_pandas()` from any plotting backend |
| **G4** | Provide a thin lets-plot helper layer with `stat_hdi()` and `stat_pointinterval()` that accept Polars frames and return lets-plot layers               |
| **G5** | Work generically for any `InferenceData` — no hard-coded variable or coordinate names                                                                  |
| **G6** | Be a data transformation layer, not a new plotting package                                                                                             |


## 3. Non-Goals


| Non-Goal                                                  | Rationale                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| Replace ArviZ diagnostics (trace plots, MCSE, ESS, R-hat) | ArviZ already does this well                                  |
| Build a full ggplot2 clone                                | lets-plot already exists; we only add Bayesian-specific stats |
| Support non-PyMC samplers without `InferenceData`         | Out of scope for v1; can revisit                              |
| Provide a high-level "auto-plot" API                      | Declarative ≠ magic; keep it composable                       |
| Wrap matplotlib or seaborn                                | lets-plot is the chosen backend                               |


---

## 4. Core Concepts

### 4.1 The Two Semantic Levels

The fundamental insight driving the design: MCMC output lives at two distinct levels that
must not be conflated in a single flat table.

```
┌─────────────────────────────────────────────────────┐
│  PARAMETER SPACE                                    │
│  Dims: (chain, draw, [named dims like groups])      │
│  Variables: intercept[groups], beta[groups], sigma  │
│  Use: density plots, trace plots, forest plots      │
│  Function: spread_draws()                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  DATA / PREDICTION SPACE                            │
│  Dims: (chain, draw, obs_ind)                       │
│  Variables: mu, y_hat                               │
│  Joined to: covariate grid (x, group, ...)          │
│  Use: regression lines, ribbons, predictive checks  │
│  Function: add_epred_draws()                        │
└─────────────────────────────────────────────────────┘
```

Joining these two levels is only done explicitly when the user's plot genuinely
requires both in the same layer — which is uncommon.

### 4.2 Why Polars (eager `DataFrame`)?


| Problem                                               | Polars eager solution                                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 320k-row frame for a density plot that needs 16k rows | Filter the returned `DataFrame` with `.filter()`; Polars eager filters are fast and in-memory |
| Coefficient duplication across obs_ind                | Parameter and prediction spaces are kept in separate frames; no fan-out by default    |
| Slow pandas joins                                     | Polars join is typically 5–20× faster than pandas                                    |


The API always returns `pl.DataFrame` (eager). The extraction path is
`xarray.DataArray.to_dataframe()` (pandas, fully materialised) → `pl.from_pandas()`, so the
data is already in memory before any Polars object exists; wrapping it in a `LazyFrame`
would buy nothing — predicate pushdown and streaming only pay off when the *source* is lazy
(e.g. `scan_parquet`), which it is not here — and would tax every plotting call with a
`.collect()` step plus the `.to_pandas()` hop that lets-plot/plotnine require anyway. Eager
frames are honest about the cost and a single `.to_pandas()` away from any ggplot-like
backend. Users can still `.filter()` / `.select()` / `.group_by()` on the returned frame
before handing it to a plot.

### 4.3 Why lets-plot?

- Near-identical API to ggplot2 and plotnine — migration cost is low
- Actively maintained by JetBrains; much faster release cadence than plotnine
- Better defaults out of the box
- Supports both Jupyter and standalone HTML output
- Accepts pandas DataFrames today; Polars support via `.to_pandas()` or direct dict input

---

## 5. API Design

### 5.1 `spread_draws()`

Extract posterior draws for named parameters into a tidy Polars DataFrame.

```python
def spread_draws(
    dt: xr.DataTree,
    *var_specs: str,
    group: str = "posterior",
    chain_dim: str = "chain",
    draw_dim: str = "draw",
) -> pl.DataFrame:
    """
    Extract posterior draws for one or more variables into a tidy Polars DataFrame.

    Parameters
    ----------
    dt : xr.DataTree
        ArviZ DataTree object (xarray.DataTree) from PyMC sampling.
    *var_specs : str
        Variable specifications in the form "var_name" for scalar variables,
        or "var_name[dim1, dim2, ...]" for array variables. Supports nested/multi-dimensional
        specifications. The bracketed dimension names must match coordinate names in the
        InferenceData dataset.
    group : str
        Which InferenceData group to extract from (e.g., "posterior", "prior").
        Default "posterior".
    chain_dim, draw_dim : str
        Names of the chain and draw dimensions.

    Returns
    -------
    pl.DataFrame
        Tidy DataFrame with columns: chain, draw, [named dims...], [var_names...]
        One row per unique (chain, draw, [dim combo]).

    Examples
    --------
    # Scalar parameter (no duplication)
    spread_draws(dt, "sigma")
    # → columns: chain, draw, sigma
    # → 4 × 1000 = 4,000 rows

    # Array parameter spread over a named dim
    spread_draws(dt, "beta[groups]", "intercept[groups]")
    # → columns: chain, draw, groups, beta, intercept
    # → 4 × 1000 × 4 = 16,000 rows (NOT 320,000)

    # Mix of scalar and array (sigma broadcast-joined to group-level params)
    spread_draws(dt, "beta[groups]", "sigma")
    # → columns: chain, draw, groups, beta, sigma
    # → 4 × 1000 × 4 = 16,000 rows; sigma repeated per group (explicit and expected)

    # Different groups (prior vs posterior)
    spread_draws(dt, "beta[groups]", group="prior")
    # → extract prior draws for beta

    # Nested dimensions
    spread_draws(dt, "gamma[time, group]")
    # → columns: chain, draw, time, group, gamma
    """
```

**Key behaviour:**

- Infers dimension names automatically from xarray coordinates — no hard-coding
- Multiple variables sharing the same dims are extracted together (no fan-out join)
- Variables with different dims trigger an explicit cross-join, with a logged warning (e.g., scalar `sigma` broadcast across `beta[groups]`)
- The string syntax `"beta[groups]"` directly mirrors R's `tidybayes::spread_draws(model, beta[group])`
- Supports any group in the DataTree (posterior, prior, custom); no special casing
- Always returns `pl.DataFrame` (eager); call `.to_pandas()` when passing to a plotting backend

---

### 5.2 `add_epred_draws()`

Join posterior predictive draws to a covariate grid.

```python
def add_epred_draws(
    dt: xr.DataTree,
    newdata: pl.DataFrame | pd.DataFrame | None,
    var_name: str,
    idata_group: str = "predictions",
    constant_data_group: str = "predictions_constant_data",
    join_on: str | list[str] = "obs_ind",
) -> pl.DataFrame:
    """
    Join posterior predictive draws to a covariate DataFrame.

    The newdata frame is the *left* table. Draws are attached to it. This avoids
    the denormalisation problem where coefficient values are duplicated for every
    obs_ind — the join goes in the right direction.

    Parameters
    ----------
    dt : xr.DataTree
        ArviZ DataTree object (xarray.DataTree) containing prediction samples.
    newdata : pl.DataFrame | pd.DataFrame | None
        Covariate grid. If None, attempts to read from dt[constant_data_group].
        If the group is not found, raises a clear error directing user to pass
        newdata explicitly.
    var_name : str
        Name of the predictive variable to extract (e.g., "mu"). Supports
        nested specifications like "mu[time, group]" if the variable has
        multiple dimensions.
    idata_group : str
        InferenceData group containing the predictive draws ("predictions",
        "posterior_predictive", or custom). Default "predictions".
    constant_data_group : str
        InferenceData group name for the covariate grid that aligns with the
        prediction draws. Default "predictions_constant_data". Set this parameter
        if your DataTree uses a different naming convention.
    join_on : str | list[str]
        Column(s) to join newdata to the draws on. Default "obs_ind".

    Returns
    -------
    pl.DataFrame
        Tidy DataFrame with columns: chain, draw, [join_on cols], [covariate cols], var_name
        One row per (chain, draw, obs_ind).

    Examples
    --------
    # Basic usage — newdata read from dt
    pred_df = add_epred_draws(dt, newdata=None, var_name="mu")
    # → columns: chain, draw, obs_ind, x, group, mu
    # → 4 × 1000 × 80 = 320,000 rows

    # Filter before plotting — only group 0
    pred_df.filter(pl.col("group") == 0)
    # → 4 × 1000 × 20 = 80,000 rows

    # Provide custom newdata (e.g., a finer grid)
    fine_grid = pl.DataFrame({"x": np.linspace(0, 20, 200), "group": ...})
    add_epred_draws(dt, newdata=fine_grid, var_name="mu")
    """
```

**Key behaviour:**

- `newdata` is the left table — draws fan out from it, not the other way around
- Returns a `pl.DataFrame` (eager); filter/select/group_by on it before passing to a plot
- Deliberately does **not** attach parameter-space variables (`beta`, `intercept`) — those are for `spread_draws()`
- Works with both `predictions` and `posterior_predictive` groups
- If `newdata=None` and the `constant_data_group` is not found in the DataTree, raises a clear error with guidance
- Supports nested var_name specifications (e.g., `var_name="mu[time, group]"`)

---

### 5.3 `spread_draws_compare()` — Comparing Posterior and Prior

Helper function for stacking draws from multiple groups (e.g., posterior vs. prior) into a single frame for comparison plots.

```python
def spread_draws_compare(
    dt: xr.DataTree,
    *var_specs: str,
    groups: list[str] = ["posterior", "prior"],
    group_name: str = "source",
) -> pl.DataFrame:
    """
    Extract and stack draws from multiple groups (e.g., posterior and prior).

    Calls spread_draws() for each group, adds a column identifying the source group,
    and concatenates the results into a single DataFrame for easy comparison.

    Parameters
    ----------
    dt : xr.DataTree
        ArviZ DataTree object.
    *var_specs : str
        Variable specifications (as for spread_draws()).
    groups : list[str]
        Which groups to extract and stack. Default ["posterior", "prior"].
    group_name : str
        Name of the column identifying the source group. Default "source".

    Returns
    -------
    pl.DataFrame
        Stacked draws with an additional column (group_name) indicating source.

    Example
    -------
    # Extract posterior and prior for side-by-side forest plots
    compare_df = spread_draws_compare(dt, "beta[groups]", 
                                       groups=["posterior", "prior"])
    # → columns: chain, draw, groups, beta, source
    # → source ∈ {"posterior", "prior"}
    """
```

**Key behaviour:**

- Calls `spread_draws()` separately for each group, then stacks with `pl.concat()`
- Adds a grouping column (`source` by default) identifying which group each row came from
- Useful for prior vs. posterior comparison plots
- Returns a `pl.DataFrame` (eager)

---

### 5.4 lets-plot Helper Layer

These are thin helpers, not a new plotting package. They accept a Polars
DataFrame and return lets-plot layer objects.

```python
def stat_hdi(
    data: pl.DataFrame,
    x: str,
    y: str,
    prob: float = 0.89,
    interval_kind: str = "hdi",
    group: str | None = None,
    **geom_ribbon_kwargs,
) -> gg.LayerSpec:
    """
    Compute credible intervals and return a lets-plot geom_ribbon layer.

    Uses arviz.hdi() or arviz.eti() under the hood depending on interval_kind.
    HDI (highest-density interval) is default but equal-tailed intervals (eti) are also supported.

    Parameters
    ----------
    data : pl.DataFrame
        Tidy draws frame from add_epred_draws() or spread_draws().
    x : str
        Column to use as x aesthetic (aggregation key).
    y : str
        Column containing the draws to summarise.
    prob : float
        Credible interval probability mass. Default 0.89 (following ArviZ 1.0 convention).
    interval_kind : str
        Type of credible interval: "hdi" (highest-density) or "eti" (equal-tailed). Default "hdi".
    group : str | None
        Optional grouping column (e.g., "group") for per-group HDI ribbons.

    Returns
    -------
    lets-plot LayerSpec (geom_ribbon with pre-computed HDI bounds)

    Example (lets-plot backend)
    ---------------------------
    from letsplot import *
    pred = add_epred_draws(dt, newdata=None, var_name="mu")

    (
        ggplot()
        + stat_hdi(pred, x="x", y="mu", prob=0.89, group="group", alpha=0.2)
        + stat_median_line(pred, x="x", y="mu", group="group")
    )

    Example (plotnine backend)
    --------------------------
    from plotnine import *
    pred = add_epred_draws(dt, newdata=None, var_name="mu")

    (
        ggplot()
        + stat_hdi(pred, aes(x="x", y="mu", group="group"), alpha=0.2)
        + stat_median_line(pred, aes(x="x", y="mu", group="group"))
    )
    """


def stat_median_line(
    data: pl.DataFrame,
    x: str,
    y: str,
    group: str | None = None,
    **geom_line_kwargs,
) -> gg.LayerSpec:
    """Compute per-x medians and return a lets-plot geom_line layer."""


def stat_pointinterval(
    data: pl.DataFrame,
    x: str,
    y: str,
    prob: float = 0.89,
    prob_outer: float = 0.95,
    point_fn: callable = np.median,
    interval_kind: str = "hdi",
    **kwargs,
) -> list[gg.LayerSpec]:
    """
    Compute point estimate + credible interval(s) for parameter-space plots.
    Returns a list of plot layers (geom_point + geom_linerange).
    Compatible with both lets-plot and plotnine backends.

    Analogous to tidybayes::stat_pointinterval.

    Example (forest plot, lets-plot)
    --------------------------------
    beta_draws = spread_draws(dt, "beta[groups]")
    from letsplot import *
    (
        ggplot()
        + stat_pointinterval(beta_draws, x="groups", y="beta", prob=0.89)
        + coord_flip()
    )

    Example (forest plot, plotnine)
    --------------------------------
    from plotnine import *
    beta_draws = spread_draws(dt, "beta[groups]")
    (
        ggplot(aes(x="groups", y="beta"))
        + stat_pointinterval(beta_draws, prob=0.89)
        + coord_flip()
    )
    """
```

---

## 6. Data Model

### Output of `spread_draws(idata, "beta[groups]", "intercept[groups]")`


| chain | draw | groups | beta  | intercept |
| ----- | ---- | ------ | ----- | --------- |
| 0     | 0    | 0      | 0.944 | -0.475    |
| 0     | 0    | 1      | 0.775 | -1.427    |
| 0     | 0    | 2      | 1.157 | 0.467     |
| 0     | 0    | 3      | 1.240 | -0.122    |
| 0     | 1    | 0      | 1.146 | -0.989    |
| ...   | ...  | ...    | ...   | ...       |


**Rows: 4 × 1000 × 4 = 16,000** (not 320,000)

### Output of `add_epred_draws(idata, newdata=None, var_name="mu")`


| chain | draw | obs_ind | x     | group | mu    |
| ----- | ---- | ------- | ----- | ----- | ----- |
| 0     | 0    | 0       | 2.562 | 0     | 1.945 |
| 0     | 0    | 1       | 2.902 | 0     | 2.265 |
| ...   | ...  | ...     | ...   | ...   | ...   |


**Rows: 4 × 1000 × 80 = 320,000** — but crucially, `beta` and `intercept` are **not here**.
This table is pure prediction-space data. No duplication of parameters.

---

## 7. Documentation & Website Strategy

**Key principle:** Documentation and examples are the primary deliverable. Code implementation is minimal; most effort goes into comprehensive, annotated examples.

**Documentation Quality Standards:** When building docs, apply the `educational-narrative` and `figure-excellence` skills to ensure:
- Clear, progressive narrative flow that builds understanding step-by-step
- High-quality, publication-ready figures and visualizations
- Examples that teach concepts, not just show syntax

### Documentation Structure (Quarto + Great Docs)

- **User Guide** (`docs/user_guide/`): Ordered tutorials (numeric prefixes) covering core workflows
  - `01-quickstart.qmd` — 5-minute intro to `spread_draws()` and `add_epred_draws()`
  - `02-parameter-space.qmd` — Forest plots, ridgeline plots with `stat_pointinterval()`
  - `03-prediction-space.qmd` — Posterior predictive fits, ribbons, credible intervals
  - `04-migration-from-arviz.qmd` — How to replace ArviZ's imperative approach with tidydraws
  - `05-backends.qmd` — Side-by-side examples using lets-plot and plotnine
  - `06-filtering-and-aggregation.qmd` — Filtering, selecting, and summarising the returned frames

- **Examples Gallery** (`docs/examples/`): Reusable notebooks with real-world models
  - `linear-regression.qmd` — Regression with group-level effects
  - `hierarchical-model.qmd` — Multi-level structure (school data example)
  - `bayesian-workflow.qmd` — Full workflow: prior, posterior, posterior predictive

- **API Reference**: Auto-generated from docstrings via Great Docs
  - All public functions documented with examples
  - Cross-linked to user guide sections

### Configuration
- `great-docs.yml` — Configuration for Great Docs (wraps Quarto Markdown)
- `_quarto.yml` — Quarto-specific styling and build settings
- GitHub Pages deployment via Actions

### Backend Coverage
Every major function has examples in **both lets-plot and plotnine**:
- lets-plot for declarative, Jupyter-first workflows
- plotnine for static reports, familiar to ggplot2 users
- Showcases that tidydraws data layer is truly backend-agnostic

---

## 8. Technical Architecture

```
tidydraws/
├── __init__.py                      # Public API exports
├── _extract.py                      # spread_draws(), add_epred_draws()
│   ├── _parse_var_spec()            # "beta[groups]" → ("beta", ["groups"])
│   ├── _datatree_group_to_df()    # xarray DataTree group → pl.DataFrame
│   └── _align_dims()                # Handle cross-dim joins
├── _stats.py                        # stat_hdi(), stat_pointinterval(), stat_median_line()
│   ├── _compute_hdi()               # Wraps az.hdi(), returns Polars frame
│   ├── _compute_eti()               # Wraps az.eti(), for equal-tailed intervals
│   └── _backend_dispatch()          # Route to lets-plot or plotnine
├── _compat.py                       # pandas↔polars↔xarray conversion helpers
└── _utils.py                        # Logging, warnings, dim inference

docs/
├── _quarto.yml                      # Quarto build config
├── great-docs.yml                   # Great Docs (dash) config
├── user_guide/
│   ├── 01-quickstart.qmd
│   ├── 02-parameter-space.qmd
│   ├── 03-prediction-space.qmd
│   ├── 04-migration-from-arviz.qmd
│   ├── 05-backends.qmd
│   └── 06-filtering-and-aggregation.qmd
├── examples/
│   ├── linear-regression.qmd
│   ├── hierarchical-model.qmd
│   └── bayesian-workflow.qmd
└── reference/                       # Auto-generated API reference

pyproject.toml                       # UV-based project config
uv.lock                              # Locked dependencies
```

### DataTree → Polars Conversion Strategy

ArviZ 1.0's `xarray.DataTree` structure must be converted to Polars for the data layer. The conversion path is:

```
xarray.DataTree
    → dt.children[group].to_dataset()  # Access group as Dataset
    → .to_dataframe()                   # pandas (fast, uses xarray's own flattening)
    → pl.from_pandas()                  # Polars DataFrame (eager)
```

This is done once per group, per call. The resulting `pl.DataFrame` is then manipulated
(select, filter, join) directly — no `.collect()` step is needed because the frame is
already eager. For very large models, an alternative path using
`xarray → numpy → pl.from_numpy()` column-by-column avoids the intermediate pandas
allocation entirely.

### Join Strategy for `add_epred_draws`

```python
# Pseudocode — eager DataFrames throughout
pred_df = _datatree_group_to_df(dt, "predictions", var_name)
const_df = _datatree_group_to_df(dt, "predictions_constant_data")

# Join covariate data to predictions (newdata is small — broadcast join)
result = pred_df.join(const_df, on=join_on, how="left")

# User can filter the returned frame before plotting:
result.filter(pl.col("group") == 2)
```


---

## 8b. Complete Usage Example

**With lets-plot backend:**

```python
import arviz as az
import tidydraws as td
from letsplot import *

# ── Parameter space ──────────────────────────────────────────────────────────

dt = az.from_netcdf("model.nc")  # Load ArviZ 1.0 DataTree
beta_draws = td.spread_draws(dt, "beta[groups]", "intercept[groups]")
# 16,000-row DataFrame — no duplication

(
    ggplot()
    + td.stat_pointinterval(beta_draws, x="groups", y="beta", prob=0.89)
    + labs(x="Group", y="β", title="Posterior slopes by group")
    + coord_flip()
    + theme_classic()
)

# ── Data / prediction space ───────────────────────────────────────────────────

pred_draws = td.add_epred_draws(dt, newdata=None, var_name="mu")
# 320,000-row DataFrame — parameters NOT included (right semantic level)

obs_df = df  # Original observed data as pandas/polars frame

(
    ggplot()
    + td.stat_hdi(pred_draws, x="x", y="mu", prob=0.89, group="group", alpha=0.15)
    + td.stat_median_line(pred_draws, x="x", y="mu", group="group", size=1)
    + geom_point(data=obs_df, mapping=aes("x", "y", color="factor(group)"),
                 size=2, alpha=0.7)
    + facet_wrap("group")
    + theme_classic()
    + labs(title="Posterior predictive fit by group")
)

# ── Filtering before plotting ──────────────────────────────────────────────────

# Only plot group 0 for a quick check
group0 = pred_draws.filter(pl.col("group") == 0)
```

**With plotnine backend:**

```python
import arviz as az
import tidydraws as td
from plotnine import *

dt = az.from_netcdf("model.nc")  # Load ArviZ 1.0 DataTree

# Parameter draws (already eager)
beta_draws = td.spread_draws(dt, "beta[groups]", "intercept[groups]")

# Create forest plot
(
    ggplot(aes(x="groups", y="beta"))
    + td.stat_pointinterval(beta_draws, prob=0.89)
    + coord_flip()
    + theme_minimal()
)

# Posterior predictive plot
pred_draws = td.add_epred_draws(dt, newdata=None, var_name="mu")

(
    ggplot(aes(x="x", y="mu"))
    + td.stat_hdi(pred_draws, group="group", alpha=0.15)
    + td.stat_median_line(pred_draws, group="group", size=1)
    + facet_wrap("~group")
    + theme_minimal()
)
```

---

## 9. Testing Strategy

Testing focuses on correctness of the data transformation layer, with emphasis on row counts, dimension inference, and eager DataFrame semantics.

### Test Levels

| Level | Focus | Tool | Examples |
| --- | --- | --- | --- |
| **Unit Tests** | Row count correctness, dimension inference, variable spec parsing | `pytest` + synthetic DataTrees | `spread_draws("beta[groups]")` → 4 × 1000 × 4 rows exactly |
| **Integration Tests** | End-to-end extraction + validation against source DataTree | `pytest` + real PyMC samples | Load model.nc, extract, verify values match original |
| **Eager Semantics Tests** | Confirm `pl.DataFrame` return type, filtering works | `pytest` + Polars introspection | Filter the returned frame, verify row count |

### Test Coverage (v0.1)

**Row count validation (primary):**
- Scalar variables (e.g., `"sigma"`)
- Single-dim arrays (e.g., `"beta[groups]"`)
- Multi-dim arrays (e.g., `"gamma[time, group]"`)
- Cross-dim joins (scalar + array, e.g., `"beta[groups]", "sigma"`)
- Verify no duplication, no missing rows

**Numerical spot-check:**
- Extract 2–3 parameters, compare to direct xarray indexing
- Not exhaustive; sampling approach to catch value corruption bugs

**DataFrame semantics:**
- Confirm return type is `pl.DataFrame` (eager), not `pl.LazyFrame`
- Verify `.filter()` / `.height` work directly without `.collect()`

**Group parameter:**
- Test extraction from `"posterior"`, `"prior"` groups
- Test invalid group name raises clear error

**add_epred_draws:**
- Join correctness (newdata rows preserved)
- Constant data group auto-detection vs. explicit parameter
- Clear error when constant data group missing and newdata=None

**spread_draws_compare:**
- Correct stacking and group column creation
- No data loss or duplication during concatenation

### Test Data

**Synthetic models** (fast, deterministic):
- Minimal xarray.DataTree objects constructed directly
- Controlled dimensions and coordinate names
- Predictable row counts for validation

**Real samples** (optional, v0.2+):
- Small PyMC models (~2 chains, 100 draws) for integration tests
- Verification that extraction matches manual xarray operations

---

### Core Dependencies

These are required for `tidydraws` to function. Users installing `tidydraws` will get these automatically.

| Package  | Role                                                                          | Version | Notes                                               |
| -------- | ----------------------------------------------------------------------------- | ------- | --------------------------------------------------- |
| `polars` | DataFrame manipulation (eager)                                               | ≥ 0.20  | Fast in-memory joins and filters                   |
| `arviz`  | Source format (`xarray.DataTree`) and interval computation (`az.hdi/eti()`) | ≥ 1.0   | Modular; only arviz-base needed, others transitive |
| `xarray` | DataTree structure                                                            | (via arviz) | Transitive dependency                              |
| `numpy`  | Array operations                                                              | (via arviz) | Transitive dependency                              |

### Optional Dependencies (Plotting Backend)

Users choose one or both for the `stat_*` helper functions. If neither is installed, the data layer (`spread_draws`, `add_epred_draws`) works fine — users can pipe to any plotting library.

| Package    | Role                            | Use Case                                                  | Installation                |
| ---------- | ------------------------------- | --------------------------------------------------------- | --------------------------- |
| `lets-plot` | Plotting backend (stat helpers) | Jupyter-native, interactive plots, lets-plot syntax        | `uv add tidydraws[letsplot]` |
| `plotnine` | Plotting backend (stat helpers) | ggplot2-familiar syntax, static reports, integrated plots | `uv add tidydraws[plotnine]` |

**Both backends:** `uv add tidydraws[plotting]`

### Optional Convenience Dependencies

| Package  | Role                                      | When Needed | Notes                                    |
| -------- | ----------------------------------------- | ----------- | ---------------------------------------- |
| `pandas` | Intermediate conversion (DataTree→Polars) | Always      | Can be eliminated with numpy-only path   |

### Developer & Documentation Dependencies

These are only needed for contributing to tidydraws or building docs locally.

| Package        | Role                      | Notes                                                |
| -------------- | ------------------------- | ---------------------------------------------------- |
| `uv`           | Package manager           | All development, testing, building, docs             |
| `pytest`       | Testing framework         | Required for running test suite                      |
| `quarto`       | Documentation builder     | Required for building Great Docs site locally        |
| `great-docs`   | Doc theme & site builder  | Wraps Quarto; installs via quarto plugin             |
| `ruff`         | Code linting & formatting | Pre-commit, CI/CD                                    |
| `mypy`         | Type checking             | Pre-commit, CI/CD                                    |

---

**Key principle:** The core `tidydraws` package is lightweight and focused on the data layer. All plotting backends are optional and user-chosen. Users can use `spread_draws()` and `add_epred_draws()` without any plotting library installed.

---

## 10. Design Decisions & Rationale

### Why ArviZ 1.0 (DataTree) over 0.x?

ArviZ 1.0 introduces `xarray.DataTree`, which is more flexible and has better API consistency with xarray itself. The modular structure (arviz-base, arviz-stats, arviz-plots) also makes it easier to use just the components we need. The breaking changes are minor for tidydraws' use case — we only access groups and convert to DataFrames.

### Why not extend ArviZ directly?

ArviZ is heavily invested in matplotlib and its own imperative plotting API. A declarative data-layer package can be developed independently and used alongside ArviZ's diagnostics, rather than competing with them.

### Why `str` var specs (`"beta[groups]"`) rather than kwargs?

The string spec directly mirrors R's tidybayes syntax, which is already familiar to the target audience. It also naturally handles the case where a single call specifies multiple variables with different dimensions — which kwargs would make awkward.

### Why both lets-plot and plotnine support in v0.2?

The data layer is backend-agnostic; examples in both backends demonstrate this and give users choice. lets-plot is more Jupyter-native, plotnine is closer to classical ggplot2 (useful for static reports).

### Why HDI as default but support ETI?

ArviZ 1.0 defaults to equal-tailed intervals (ETI) via `ci_kind="eti"`. However, HDI is still widely used and more correct for asymmetric posteriors. We support both, with HDI as the default in our stat functions (can be overridden).

### Why 0.89 as the default interval probability?

Following `tidybayes` and Richard McElreath's convention. Clearly non-round, which communicates that the interval is one of many possible choices rather than a hard threshold. Configurable.

---

## 11. ArviZ 1.0 Migration Notes

The shift to ArviZ 1.0 impacts tidydraws in these ways:

| Change | Impact | Solution |
| --- | --- | --- |
| `InferenceData` → `xarray.DataTree` | Group access returns `DataTree`, not `Dataset` | Use `.children[group].to_dataset()` to convert |
| Default `ci_prob` changed from 0.94 → 0.89 | Matches our default! No change needed | Already aligned |
| Default `ci_kind` changed to "eti" | We support both `hdi` and `eti` via `interval_kind` parameter | Make it explicit and configurable |
| `az.hdi()` API unchanged | Still available for interval computation | No code changes required |
| `arviz-base`, `arviz-stats`, `arviz-plots` modular | Users only install what they need | Document optional deps clearly |

---

## 12. Milestones

### v0.1 — Data Layer + Core Docs (core value)

**Code:**
- [x] `_datatree_group_to_df()` — generic DataTree group → Polars DataFrame
- [x] `_parse_var_spec()` — string spec parser supporting nested dims (e.g., `"beta[groups, time]"`)
- [x] `spread_draws()` — extract parameter draws, supporting any group (posterior, prior, etc.)
- [x] `spread_draws_compare()` — helper for stacking and comparing multiple groups (e.g., posterior vs. prior)
- [x] `add_epred_draws()` — prediction draws joined to covariate grid, with configurable constant data group
- [x] Tests: row count validation (primary), numerical spot-checks, eager DataFrame semantics, group parameter, error handling

**Docs:**
- [x] `docs/user_guide/01-quickstart.qmd` — 5-min intro with both posterior and prior extraction
- [x] `docs/user_guide/02-parameter-space.qmd` — Forest plots, prior vs. posterior comparison
- [x] `docs/user_guide/03-prediction-space.qmd` — Posterior predictive fits and credible intervals
- [x] `docs/user_guide/04-migration-from-arviz.qmd` — Migration guide from ArviZ imperative approach
- [x] `docs/examples/linear-regression.qmd` — Worked example with group-level effects
- [x] API reference auto-generated and linked to guide sections
- [x] README with quick start
- [ ] GitHub Pages site deployed via Great Docs

### v0.2 — Plotting Stats Layer + Backend Parity

**Code:**
- [ ] `stat_hdi()` — HDI credible interval ribbon layer (lets-plot + plotnine)
- [ ] `stat_eti()` — Equal-tailed interval option (ArviZ 1.0 default)
- [ ] `stat_median_line()` — median line layer
- [ ] `stat_pointinterval()` — point + interval for forest plots
- [ ] Backend dispatch logic for lets-plot and plotnine

**Docs:**
- [x] `docs/user_guide/05-backends.qmd` — Side-by-side lets-plot and plotnine examples
- [ ] `docs/examples/hierarchical-model.qmd` — Multi-level example demonstrating both backends
- [x] Backend-agnostic examples showing data layer works with any plotting library

### v0.3 — Advanced Features & Comprehensive Docs

**Code:**
- [ ] Support `posterior_predictive` group in addition to `predictions`
- [ ] Support `observed_data` passthrough for posterior predictive checks
- [ ] `add_predicted_draws()` — full predictive draws including observation noise
- [ ] Chunked/streaming extraction path for very large models (lazy xarray scan → Polars)

**Docs:**
- [ ] `docs/user_guide/06-filtering-and-aggregation.qmd` — Filtering, selecting, and summarising the returned frames
- [ ] `docs/examples/bayesian-workflow.qmd` — Full pipeline with prior predictive, posterior, posterior predictive
- [ ] Troubleshooting guide (dimension mismatches, group name errors, etc.)

### Future (post-v1)

- Direct Polars integration in `xarray.to_dataframe()` (upstream PR)
- `stat_halfeye()` — distribution + interval combined
- Support for multidimensional coordinates (e.g., `beta[time, group]`)
- Integration with `pathmc` for structural causal model draws

---

## 13. Open Questions                                                                                                    |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Package name                                                                                 | `tidydraws`, `pydraws`, `bayesframe`, `posteriorframe`                    | `**tidydraws**` — clearest analogy to tidybayes                                                                   |
| Publish under PyMC Labs umbrella?                                                            | Standalone repo vs. PyMC Labs org                                         | Start standalone (own repo); propose to PyMC Labs once v0.2 is stable                                             |
| Documentation hosting                                                                        | GitHub Pages (Great Docs) vs. Read the Docs                               | **GitHub Pages + Great Docs (dash)** — matches pathmc setup, Quarto Markdown support                              |
| Interval computation default                                                                 | HDI vs. ETI (ArviZ 1.0 default)                                           | **HDI by default** but configurable; ETI support via `interval_kind="eti"` parameter                              |
| Does `add_epred_draws` need to support in-sample fits?                                      | Yes / No                                                                   | **Yes in v0.3** — needed for posterior predictive checks without `pm.sample_posterior_predictive`                 |
| Coordinate name inference                                                                    | Strict (fail on ambiguity) vs. lenient (warn and guess)                   | **Strict with helpful error messages** — silent guessing creates hard-to-debug bugs                               |
| Support `posterior_predictive` vs. only `predictions`?                                       | Both / predictions only                                                   | **Both in v0.2** — common model structures use posterior_predictive                                                |
| Documentation-to-code effort ratio                                                           | How much docs vs. code?                                                   | **70% docs / 30% code** — comprehensive examples and migration guides are the primary value                       |



---

## Appendix: The Problem Illustrated

```
Current approach (make_tidy):
──────────────────────────────
beta (chain, draw, groups)    →  16,000 unique values
mu   (chain, draw, obs_ind)   →  320,000 unique values

After merge on (chain, draw, group):
  beta is DUPLICATED 20× per group (one per obs_ind)
  Final frame: 320,000 rows, beta/intercept values wastefully repeated

tidydraws approach:
──────────────────────────────
spread_draws(idata, "beta[groups]")
  → 16,000-row DataFrame  ✓  (no duplication)

add_epred_draws(idata, var_name="mu")
  → 320,000-row DataFrame  ✓  (prediction space only; beta not here)

Join only if you genuinely need both:
  spread_draws(...).join(add_epred_draws(...), on=["chain", "draw", "group"])
  → Explicit, and the user understands what they're paying for
```

 