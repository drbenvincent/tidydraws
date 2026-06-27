"""Model factories for the tidydraws showcase page.

Each function returns the core objects needed for extraction and plotting.
All models use small dimensions to keep build times reasonable.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pymc as pm


# ---------------------------------------------------------------------------
# 1. Simple linear regression — scalar params only, no group dims
# ---------------------------------------------------------------------------


def simple_regression(seed: int = 2026):
    """y ~ normal(alpha + beta * x, sigma).  Scalar parameters only."""
    rng = np.random.default_rng(seed)
    n = 50
    x = rng.uniform(-3, 3, n)
    alpha_true, beta_true, sigma_true = 1.5, 2.0, 0.8
    y = rng.normal(alpha_true + beta_true * x, sigma_true)
    observed = pl.DataFrame({"obs_ind": np.arange(n), "x": x, "y": y})

    with pm.Model():
        pm.Data("x", x, dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3)
        beta = pm.Normal("beta", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.Deterministic("mu", alpha + beta * x, dims="obs_ind")
        pm.Normal("y", mu, sigma, observed=y, dims="obs_ind")
        prior = pm.sample_prior_predictive(draws=500, random_seed=seed + 1)
        dt = pm.sample(draws=400, tune=400, random_seed=seed, progressbar=False)
        pm.sample_posterior_predictive(
            dt,
            var_names=["mu"],
            predictions=True,
            extend_inferencedata=True,
            random_seed=seed + 2,
            progressbar=False,
        )
    dt.update(prior)
    return dt, observed


# ---------------------------------------------------------------------------
# 2. Varying intercepts — 1-d array with string coords
# ---------------------------------------------------------------------------


def varying_intercepts(seed: int = 2026):
    """y ~ normal(alpha[group] + beta * x, sigma).  String group labels."""
    rng = np.random.default_rng(seed)
    groups = ["Control", "Treatment A", "Treatment B", "Treatment C", "Treatment D"]
    n_groups = len(groups)
    n_per = 10
    n = n_groups * n_per
    group_idx = np.repeat(np.arange(n_groups), n_per)
    x = rng.uniform(-2, 3, n)
    alpha_true = np.array([0.0, 0.6, 1.2, 1.8, -0.2])
    beta_true = 1.5
    sigma_true = 0.5
    y = rng.normal(alpha_true[group_idx] + beta_true * x, sigma_true)
    group_labels = np.array(groups, dtype=object)[group_idx]
    observed = pl.DataFrame({
        "obs_ind": np.arange(n),
        "group": group_labels,
        "group_idx": group_idx,
        "x": x,
        "y": y,
    })

    coords = {"group": groups}
    with pm.Model(coords=coords):
        pm.Data("x", x, dims="obs_ind")
        pm.Data("group_idx", group_idx.astype(int), dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3, dims="group")
        beta = pm.Normal("beta", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.Deterministic(
            "mu",
            alpha[group_idx] + beta * x,
            dims="obs_ind",
        )
        pm.Normal("y", mu, sigma, observed=y, dims="obs_ind")
        dt = pm.sample(draws=400, tune=400, random_seed=seed, progressbar=False)
    return dt, observed


# ---------------------------------------------------------------------------
# 3. Varying slopes — cross-dim broadcast (beta[group] + scalar sigma)
# ---------------------------------------------------------------------------


def varying_slopes(seed: int = 2026):
    """y ~ normal(alpha + beta[group] * x, sigma).  Tests scalar × 1-d join."""
    rng = np.random.default_rng(seed)
    n_groups = 4
    n_per = 12
    n = n_groups * n_per
    group_idx = np.repeat(np.arange(n_groups), n_per)
    x = rng.uniform(-2, 3, n)
    alpha_true = 1.0
    beta_true = np.array([0.3, 0.9, 1.5, -0.5])
    sigma_true = 0.6
    y = rng.normal(alpha_true + beta_true[group_idx] * x, sigma_true)
    groups_arr = np.array(["G1", "G2", "G3", "G4"], dtype=object)[group_idx]
    observed = pl.DataFrame({
        "obs_ind": np.arange(n),
        "group": groups_arr,
        "group_idx": group_idx,
        "x": x,
        "y": y,
    })

    coords = {"group": ["G1", "G2", "G3", "G4"]}
    with pm.Model(coords=coords):
        pm.Data("x", x, dims="obs_ind")
        pm.Data("group_idx", group_idx.astype(int), dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3)
        beta = pm.Normal("beta", 0, 2, dims="group")
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.Deterministic(
            "mu",
            alpha + beta[group_idx] * x,
            dims="obs_ind",
        )
        pm.Normal("y", mu, sigma, observed=y, dims="obs_ind")
        prior = pm.sample_prior_predictive(
            draws=500,
            random_seed=seed + 1,
            var_names=["alpha", "beta", "sigma"],
        )
        dt = pm.sample(draws=400, tune=400, random_seed=seed, progressbar=False)
    dt.update(prior)
    return dt, observed


# ---------------------------------------------------------------------------
# 4. Varying intercepts + slopes — bivariate group-level posterior
# ---------------------------------------------------------------------------


def varying_both(seed: int = 2026):
    """y ~ normal(alpha[group] + beta[group] * x, sigma).  Two 1-d arrays."""
    rng = np.random.default_rng(seed)
    n_groups = 4
    n_per = 12
    n = n_groups * n_per
    group_idx = np.repeat(np.arange(n_groups), n_per)
    x = rng.uniform(-2, 3, n)
    alpha_true = np.array([-0.5, 0.2, 0.9, 1.8])
    beta_true = np.array([0.3, 0.8, 1.3, -0.4])
    sigma_true = 0.5
    y = rng.normal(alpha_true[group_idx] + beta_true[group_idx] * x, sigma_true)
    groups_arr = np.array(["A", "B", "C", "D"], dtype=object)[group_idx]
    observed = pl.DataFrame({
        "obs_ind": np.arange(n),
        "group": groups_arr,
        "group_idx": group_idx,
        "x": x,
        "y": y,
    })

    coords = {"group": ["A", "B", "C", "D"]}
    with pm.Model(coords=coords):
        pm.Data("x", x, dims="obs_ind")
        pm.Data("group_idx", group_idx.astype(int), dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3, dims="group")
        beta = pm.Normal("beta", 0, 2, dims="group")
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.Deterministic(
            "mu",
            alpha[group_idx] + beta[group_idx] * x,
            dims="obs_ind",
        )
        pm.Normal("y", mu, sigma, observed=y, dims="obs_ind")
        dt = pm.sample(draws=400, tune=400, random_seed=seed, progressbar=False)
    return dt, observed


# ---------------------------------------------------------------------------
# 5. Multiple regression — 3 predictors, varying intercepts
# ---------------------------------------------------------------------------


def multiple_regression(seed: int = 2026):
    """y ~ normal(alpha[group] + b1*x1 + b2*x2 + b3*x3, sigma)."""
    rng = np.random.default_rng(seed)
    n_groups = 4
    n_per = 15
    n = n_groups * n_per
    group_idx = np.repeat(np.arange(n_groups), n_per)
    x1 = rng.uniform(-2, 3, n)
    x2 = rng.uniform(-2, 3, n)
    x3 = rng.uniform(-2, 3, n)
    alpha_true = np.array([-0.5, 0.0, 0.5, 1.2])
    b1_true, b2_true, b3_true = 0.8, -0.4, 1.2
    sigma_true = 0.6
    y = rng.normal(
        alpha_true[group_idx] + b1_true * x1 + b2_true * x2 + b3_true * x3,
        sigma_true,
    )
    groups_arr = np.array(["G1", "G2", "G3", "G4"], dtype=object)[group_idx]
    observed = pl.DataFrame({
        "obs_ind": np.arange(n),
        "group": groups_arr,
        "group_idx": group_idx,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y": y,
    })

    coords = {"group": ["G1", "G2", "G3", "G4"]}
    with pm.Model(coords=coords):
        pm.Data("x1", x1, dims="obs_ind")
        pm.Data("x2", x2, dims="obs_ind")
        pm.Data("x3", x3, dims="obs_ind")
        pm.Data("group_idx", group_idx.astype(int), dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3, dims="group")
        b1 = pm.Normal("b1", 0, 2)
        b2 = pm.Normal("b2", 0, 2)
        b3 = pm.Normal("b3", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.Deterministic(
            "mu",
            alpha[group_idx] + b1 * x1 + b2 * x2 + b3 * x3,
            dims="obs_ind",
        )
        pm.Normal("y", mu, sigma, observed=y, dims="obs_ind")
        dt = pm.sample(draws=400, tune=400, random_seed=seed, progressbar=False)
    return dt, observed


# ---------------------------------------------------------------------------
# 6. Logistic regression — Bernoulli outcome
# ---------------------------------------------------------------------------


def logistic(seed: int = 2026):
    """y ~ Bernoulli(logit⁻¹(alpha[group] + beta * x)).  Non-gaussian model."""
    rng = np.random.default_rng(seed)
    groups = ["Low dose", "Medium dose", "High dose"]
    n_groups = len(groups)
    n_per = 20
    n = n_groups * n_per
    group_idx = np.repeat(np.arange(n_groups), n_per)
    x = rng.uniform(-3, 3, n)
    alpha_true = np.array([-1.0, 0.0, 1.2])
    beta_true = 1.5
    logit_p = alpha_true[group_idx] + beta_true * x
    p = 1 / (1 + np.exp(-logit_p))
    y = rng.binomial(1, p).astype(float)
    group_labels = np.array(groups, dtype=object)[group_idx]
    observed = pl.DataFrame({
        "obs_ind": np.arange(n),
        "group": group_labels,
        "group_idx": group_idx,
        "x": x,
        "y": y,
    })

    # Prediction grid for smooth probability curves
    grid_x = np.linspace(-3, 3, 80)
    grid_pieces = []
    for i, g in enumerate(groups):
        start_idx = i * len(grid_x)
        grid_pieces.append(
            pl.DataFrame({
                "obs_ind": np.arange(start_idx, start_idx + len(grid_x)),
                "group": g,
                "group_idx": i,
                "x": grid_x,
            })
        )
    grid = pl.concat(grid_pieces)

    coords = {"group": groups}
    with pm.Model(coords=coords):
        pm.Data("x", x, dims="obs_ind")
        pm.Data("group_idx", group_idx.astype(int), dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3, dims="group")
        beta = pm.Normal("beta", 0, 2)
        logit_p_var = pm.Deterministic(
            "logit_p",
            alpha[group_idx] + beta * x,
            dims="obs_ind",
        )
        pm.Bernoulli("y", logit_p=logit_p_var, observed=y, dims="obs_ind")
        dt = pm.sample(draws=400, tune=400, random_seed=seed, progressbar=False)
    return dt, observed, grid


# ---------------------------------------------------------------------------
# 7. Single-chain simple regression — edge case
# ---------------------------------------------------------------------------


def simple_regression_1chain(seed: int = 2026):
    """Same as simple_regression but with a single chain."""
    rng = np.random.default_rng(seed)
    n = 50
    x = rng.uniform(-3, 3, n)
    alpha_true, beta_true, sigma_true = 1.5, 2.0, 0.8
    y = rng.normal(alpha_true + beta_true * x, sigma_true)
    observed = pl.DataFrame({"obs_ind": np.arange(n), "x": x, "y": y})

    with pm.Model():
        pm.Data("x", x, dims="obs_ind")
        alpha = pm.Normal("alpha", 0, 3)
        beta = pm.Normal("beta", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.Deterministic("mu", alpha + beta * x, dims="obs_ind")
        pm.Normal("y", mu, sigma, observed=y, dims="obs_ind")
        dt = pm.sample(
            draws=400,
            tune=400,
            chains=1,
            random_seed=seed,
            progressbar=False,
        )
    return dt, observed


# ---------------------------------------------------------------------------
# 8. Prior vs posterior comparison — uses varying_slopes model with prior
# ---------------------------------------------------------------------------
# Reuse varying_slopes() — it already samples prior_predictive with
# var_names=["alpha", "beta", "sigma"], so compare_draws() works directly.


# ---------------------------------------------------------------------------
# Shared plotting helper
# ---------------------------------------------------------------------------


def interval_summary(
    data: pl.DataFrame,
    value: str,
    by,
    probs: list[float],
) -> pl.DataFrame:
    """Summarise draws into medians and central intervals."""
    by = [by] if isinstance(by, str) else list(by)
    pieces = []
    for prob in probs:
        tail = (1 - prob) / 2
        pieces.append(
            data
            .group_by(by)
            .agg(
                pl.col(value).quantile(tail).alias("lower"),
                pl.col(value).median().alias("median"),
                pl.col(value).quantile(1 - tail).alias("upper"),
            )
            .with_columns(
                pl.lit(prob).alias("prob"),
                pl.lit(f"{int(prob * 100)}%").alias("interval"),
            )
        )
    return pl.concat(pieces).sort([*by, "prob"])
