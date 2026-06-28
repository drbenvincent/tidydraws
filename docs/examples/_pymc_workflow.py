# Copyright (c) 2026 Benjamin Vincent
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Shared PyMC workflow helpers for executable documentation examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class WorkflowData:
    """Observed data and true parameters for the docs regression example."""

    observed: pl.DataFrame
    truth: pl.DataFrame
    group_names: list[str]
    sigma_true: float


def simulate_grouped_regression(seed: int = 2026) -> WorkflowData:
    """Simulate grouped regression data with intentionally different slopes."""
    rng = np.random.default_rng(seed)
    group_names = ["North", "South", "East", "West"]
    n_groups = len(group_names)
    n_per_group = 24
    group_idx = np.repeat(np.arange(n_groups), n_per_group)
    obs_ind = np.arange(n_groups * n_per_group)

    # Wide separation in both intercepts and slopes keeps the docs plots readable.
    intercept_true = np.array([-1.25, 0.15, 1.15, 2.05])
    beta_true = np.array([0.35, 0.85, 1.35, -0.55])
    sigma_true = 0.35

    x_base = np.tile(np.linspace(-2.0, 2.0, n_per_group), n_groups)
    x = x_base + rng.normal(0.0, 0.08, size=x_base.size)
    mu_true = intercept_true[group_idx] + beta_true[group_idx] * x
    y = rng.normal(mu_true, sigma_true)
    group = np.array(group_names, dtype=object)[group_idx]

    observed = pl.DataFrame({
        "obs_ind": obs_ind,
        "groups": group,
        "group_idx": group_idx,
        "x": x,
        "mu_true": mu_true,
        "y": y,
    })
    truth = pl.DataFrame({
        "groups": group_names,
        "intercept_true": intercept_true,
        "beta_true": beta_true,
    })
    return WorkflowData(
        observed=observed,
        truth=truth,
        group_names=group_names,
        sigma_true=sigma_true,
    )
