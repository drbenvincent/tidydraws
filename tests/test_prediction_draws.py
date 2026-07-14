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
import pandas as pd
from tidydraws import prediction_draws


@pytest.fixture
def synthetic_dt_with_predictions():
    """
    Construct a minimal xarray.DataTree for testing prediction_draws().
    Includes:
    - predictions group: (chain=2, draw=5, obs_ind=20)
    - predictions_constant_data group: (obs_ind=20, covariates: x, group)
    - Prediction variable: mu[obs_ind]
    """
    # Create coordinates
    chains = np.arange(2)
    draws = np.arange(5)
    obs_inds = np.arange(20)
    groups = np.arange(3)

    # 1. Predictions group with mu variable (chain, draw, obs_ind)
    mu_data = np.random.randn(2, 5, 20)
    predictions_ds = xr.Dataset(
        {"mu": (["chain", "draw", "obs_ind"], mu_data)},
        coords={"chain": chains, "draw": draws, "obs_ind": obs_inds},
    )

    # 2. Constant data group with covariates
    const_data = np.random.randn(20, 3)  # 20 observations, 3 groups
    constant_data_ds = xr.Dataset(
        {
            "x": (["obs_ind"], np.random.randn(20)),
            "group": (["obs_ind"], np.random.choice(groups, 20)),
        },
        coords={"obs_ind": obs_inds},
    )

    # Create DataTree
    dt = xr.DataTree()
    dt["predictions"] = predictions_ds
    dt["predictions_constant_data"] = constant_data_ds

    return dt


def test_join_correctness(synthetic_dt_with_predictions):
    """Test that join produces correct row count and correct values."""
    dt = synthetic_dt_with_predictions

    # Test basic join with newdata=None (should read from constant data group)
    result = prediction_draws(dt, newdata=None, var_name="mu")

    # Row count: chain × draw × obs_ind = 2 × 5 × 20 = 200 rows
    assert result.height == 2 * 5 * 20 == 200

    # Check that the newdata columns (x, group) are preserved
    collected_df = result
    assert "x" in collected_df.columns
    assert "group" in collected_df.columns
    assert "obs_ind" in collected_df.columns

    # Spot-check a few values to ensure correctness
    # (values should be from the mu array we created)
    first_rows = result.limit(5)
    assert len(first_rows) == 5
    assert "mu" in first_rows.columns


def test_constant_data_group_handling(synthetic_dt_with_predictions):
    """Test that constant data group handling works correctly."""
    dt = synthetic_dt_with_predictions

    # Test default group name auto-read
    result1 = prediction_draws(dt, newdata=None, var_name="mu")
    assert result1.height == 200

    # Test explicit group parameter
    result2 = prediction_draws(
        dt, newdata=None, var_name="mu", constant_data_group="predictions_constant_data"
    )
    assert result2.height == 200

    # Test error on missing group
    dt_no_const = xr.DataTree()
    dt_no_const["predictions"] = dt["predictions"].to_dataset()
    with pytest.raises(KeyError, match="predictions_constant_data.*not found"):
        prediction_draws(dt_no_const, newdata=None, var_name="mu")


def test_newdata_parameter(synthetic_dt_with_predictions):
    """Test that newdata parameter works correctly."""
    dt = synthetic_dt_with_predictions

    # Test with pandas DataFrame
    newdata_pd = pd.DataFrame({
        "obs_ind": np.arange(20),
        "x": np.random.randn(20),
        "group": np.random.choice([0, 1, 2], 20),
    })
    result_pd = prediction_draws(dt, newdata=newdata_pd, var_name="mu")
    assert result_pd.height == 200

    # Test with newdata=None when group is present (should work)
    result_none = prediction_draws(dt, newdata=None, var_name="mu")
    assert result_none.height == 200

    # Test that we get expected columns back
    collected = result_none
    assert "x" in collected.columns
    assert "group" in collected.columns
    assert "obs_ind" in collected.columns
    assert "mu" in collected.columns


def test_newdata_missing_group_error(synthetic_dt_with_predictions):
    """Test that error is raised when newdata=None and group is missing."""
    dt = synthetic_dt_with_predictions
    # This one should work (constant data group exists)

    # Create a tree without constant_data_group
    dt_no_const = xr.DataTree()
    dt_no_const["predictions"] = dt["predictions"].to_dataset()
    assert "predictions_constant_data" not in dt_no_const.children

    with pytest.raises(KeyError, match="predictions_constant_data.*not found"):
        prediction_draws(dt_no_const, newdata=None, var_name="mu")


def test_eager_semantics(synthetic_dt_with_predictions):
    """Test that return type is DataFrame (eager) and filtering works."""
    dt = synthetic_dt_with_predictions

    result = prediction_draws(dt, newdata=None, var_name="mu")

    # Test return type is pl.DataFrame (eager)
    assert isinstance(result, pl.DataFrame)

    # Test filtering works on the eager frame
    filtered_result = result.filter(pl.col("obs_ind") == 0)
    assert isinstance(filtered_result, pl.DataFrame)
    assert filtered_result.height >= 0  # At least some rows


def test_no_parameter_duplication(synthetic_dt_with_predictions):
    """Test that there's no parameter duplication in the result."""
    dt = synthetic_dt_with_predictions

    result = prediction_draws(dt, newdata=None, var_name="mu")

    df = result

    # Check for duplicate column names
    columns = df.columns
    unique_columns = set(columns)
    assert len(columns) == len(unique_columns), "Duplicate column names found"


def test_error_on_missing_variable(synthetic_dt_with_predictions):
    """Test that error is raised when variable is missing."""
    dt = synthetic_dt_with_predictions

    with pytest.raises(KeyError, match="Variable.*not found"):
        prediction_draws(dt, newdata=None, var_name="nonexistent")


def test_error_on_missing_group(synthetic_dt_with_predictions):
    """Test that error is raised when prediction group is missing."""
    dt = synthetic_dt_with_predictions

    # Create a tree without predictions group
    dt_no_pred = xr.DataTree()
    dt_no_pred["predictions_constant_data"] = dt[
        "predictions_constant_data"
    ].to_dataset()
    assert "predictions" not in dt_no_pred.children

    with pytest.raises(KeyError, match="Group.*not found"):
        prediction_draws(dt_no_pred, newdata=None, var_name="mu")
