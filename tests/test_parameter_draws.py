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

import pytest
import polars as pl
import xarray as xr
import numpy as np
from tidydraws import parameter_draws


@pytest.fixture
def synthetic_dt():
    """
    Construct a minimal xarray.DataTree for testing.
    Dimensions: chain=2, draw=5, groups=3, time=2
    """
    # Create coordinates
    chains = np.arange(2)
    draws = np.arange(5)
    groups = np.arange(3)
    times = np.arange(2)

    # 1. Scalar variable: sigma (chain, draw)
    sigma_data = np.random.randn(2, 5)
    sigma_ds = xr.Dataset(
        {"sigma": (["chain", "draw"], sigma_data)},
        coords={"chain": chains, "draw": draws},
    )

    # 2. 1-d array: beta (chain, draw, groups)
    beta_data = np.random.randn(2, 5, 3)
    beta_ds = xr.Dataset(
        {"beta": (["chain", "draw", "groups"], beta_data)},
        coords={"chain": chains, "draw": draws, "groups": groups},
    )

    # 3. 2-d array: gamma (chain, draw, time, group)
    gamma_data = np.random.randn(2, 5, 2, 3)
    gamma_ds = xr.Dataset(
        {"gamma": (["chain", "draw", "time", "group"], gamma_data)},
        coords={"chain": chains, "draw": draws, "time": times, "group": groups},
    )

    # We need a single Dataset for the group because parameter_draws
    # does dt.children[group].to_dataset()
    # So we merge them into one dataset for 'posterior'
    posterior = xr.Dataset(
        {
            "sigma": (["chain", "draw"], sigma_data),
            "beta": (["chain", "draw", "groups"], beta_data),
            "gamma": (["chain", "draw", "time", "group"], gamma_data),
        },
        coords={
            "chain": chains,
            "draw": draws,
            "groups": groups,
            "time": times,
            "group": groups,  # using both 'groups' and 'group' to test dim names
        },
    )

    # Create DataTree’ (available in xarray >= 2024.x / current versions)
    dt = xr.DataTree()
    dt["posterior"] = posterior
    dt["prior"] = posterior * 0.5  # Simple differenct data for prior

    return dt


def test_row_count_scalar(synthetic_dt):
    # Scalar only: chain=2, draw=5 -> 10 rows
    lf = parameter_draws(synthetic_dt, "sigma", group="posterior")
    df = lf
    assert df.height == 2 * 5


def test_row_count_1d(synthetic_dt):
    # 1-d array: chain=2, draw=5, groups=3 -> 30 rows
    lf = parameter_draws(synthetic_dt, "beta", group="posterior")
    df = lf
    assert df.height == 2 * 5 * 3


def test_row_count_2d(synthetic_dt):
    # 2-d array: chain=2, draw=5, time=2, group=3 -> 60 rows
    lf = parameter_draws(synthetic_dt, "gamma", group="posterior")
    df = lf
    assert df.height == 2 * 5 * 2 * 3


def test_row_count_cross_dim(synthetic_dt):
    # Cross-dim (scalar + 1-d): beta[groups] is the driver -> 30 rows
    lf = parameter_draws(synthetic_dt, "beta", "sigma", group="posterior")
    df = lf
    assert df.height == 2 * 5 * 3


def test_eager_semantics(synthetic_dt):
    # Verify return type is pl.DataFrame (eager)
    df = parameter_draws(synthetic_dt, "sigma", group="posterior")
    assert isinstance(df, pl.DataFrame)

    # Eager frames expose .height directly
    assert df.height == 2 * 5


def test_filtering(synthetic_dt):
    # Test filtering on an eager DataFrame
    df = parameter_draws(synthetic_dt, "beta", group="posterior")
    # Filter for groups == 0 (should be 2 * 5 * 1 = 10 rows)
    filtered = df.filter(pl.col("groups") == 0)
    assert filtered.height == 10


def test_error_invalid_group(synthetic_dt):
    with pytest.raises(KeyError, match="Group 'nonexistent' not found"):
        parameter_draws(synthetic_dt, "sigma", group="nonexistent")


def test_error_variable_not_found(synthetic_dt):
    with pytest.raises(KeyError, match="Variable 'missing' not found"):
        parameter_draws(synthetic_dt, "missing", group="posterior")


def test_numerical_correctness(synthetic_dt):
    # Spot-check one value from beta
    lf = parameter_draws(synthetic_dt, "beta", group="posterior")
    df = lf

    # Find row for chain=0, draw=0, groups=0
    val = df.filter(
        (pl.col("chain") == 0) & (pl.col("draw") == 0) & (pl.col("groups") == 0)
    ).get_column("beta")[0]

    # Compare to direct xarray indexing
    expected = synthetic_dt["posterior"].beta.values[0, 0, 0]
    assert np.isclose(val, expected)


def test_numerical_correctness_2d(synthetic_dt):
    # Spot-check one value from gamma (chain, draw, time, group)
    lf = parameter_draws(synthetic_dt, "gamma", group="posterior")
    df = lf

    # Find row for chain=0, draw=0, time=1, group=2
    val = df.filter(
        (pl.col("chain") == 0)
        & (pl.col("draw") == 0)
        & (pl.col("time") == 1)
        & (pl.col("group") == 2)
    ).get_column("gamma")[0]

    expected = synthetic_dt["posterior"].gamma.values[0, 0, 1, 2]
    assert np.isclose(val, expected)
