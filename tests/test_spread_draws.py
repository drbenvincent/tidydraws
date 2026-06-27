import pytest
import polars as pl
import xarray as xr
import numpy as np
from tidydraws import spread_draws

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
        coords={"chain": chains, "draw": draws}
    )

    # 2. 1-d array: beta (chain, draw, groups)
    beta_data = np.random.randn(2, 5, 3)
    beta_ds = xr.Dataset(
        {"beta": (["chain", "draw", "groups"], beta_data)},
        coords={"chain": chains, "draw": draws, "groups": groups}
    )

    # 3. 2-d array: gamma (chain, draw, time, group)
    gamma_data = np.random.randn(2, 5, 2, 3)
    gamma_ds = xr.Dataset(
        {"gamma": (["chain", "draw", "time", "group"], gamma_data)},
        coords={"chain": chains, "draw": draws, "time": times, "group": groups}
    )

    # We need a single Dataset for the group because spread_draws 
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
            "group": groups, # using both 'groups' and 'group' to test dim names
        }
    )

    # Create DataTree’ (available in xarray >= 2024.x / current versions)
    dt = xr.DataTree()
    dt["posterior"] = posterior
    dt["prior"] = posterior * 0.5 # Simple differenct data for prior
    
    return dt

def test_row_count_scalar(synthetic_dt):
    # Scalar only: chain=2, draw=5 -> 10 rows
    lf = spread_draws(synthetic_dt, "sigma", group="posterior")
    df = lf
    assert df.height == 2 * 5

def test_row_count_1d(synthetic_dt):
    # 1-d array: chain=2, draw=5, groups=3 -> 30 rows
    lf = spread_draws(synthetic_dt, "beta[groups]", group="posterior")
    df = lf
    assert df.height == 2 * 5 * 3

def test_row_count_2d(synthetic_dt):
    # 2-d array: chain=2, draw=5, time=2, group=3 -> 60 rows
    lf = spread_draws(synthetic_dt, "gamma[time, group]", group="posterior")
    df = lf
    assert df.height == 2 * 5 * 2 * 3

def test_row_count_cross_dim(synthetic_dt):
    # Cross-dim (scalar + 1-d): beta[groups] is the driver -> 30 rows
    lf = spread_draws(synthetic_dt, "beta[groups]", "sigma", group="posterior")
    df = lf
    assert df.height == 2 * 5 * 3

def test_eager_semantics(synthetic_dt):
    # Verify return type is pl.DataFrame (eager)
    df = spread_draws(synthetic_dt, "sigma", group="posterior")
    assert isinstance(df, pl.DataFrame)

    # Eager frames expose .height directly
    assert df.height == 2 * 5

def test_filtering(synthetic_dt):
    # Test filtering on an eager DataFrame
    df = spread_draws(synthetic_dt, "beta[groups]", group="posterior")
    # Filter for groups == 0 (should be 2 * 5 * 1 = 10 rows)
    filtered = df.filter(pl.col("groups") == 0)
    assert filtered.height == 10

def test_error_invalid_group(synthetic_dt):
    with pytest.raises(KeyError, match="Group 'nonexistent' not found"):
        spread_draws(synthetic_dt, "sigma", group="nonexistent")

def test_error_malformed_spec(synthetic_dt):
    # Unmatched brackets
    with pytest.raises(ValueError, match="Malformed variable specification"):
        spread_draws(synthetic_dt, "beta[groups")
    
    # Empty brackets
    with pytest.raises(ValueError, match="cannot have empty brackets"):
        spread_draws(synthetic_dt, "beta[]")

def test_error_variable_not_found(synthetic_dt):
    with pytest.raises(KeyError, match="Variable 'missing' not found"):
        spread_draws(synthetic_dt, "missing[groups]", group="posterior")

def test_error_dimension_mismatch(synthetic_dt):
    # beta has [groups], but we specify [time]
    with pytest.raises(ValueError, match="Dimension mismatch for 'beta'"):
        spread_draws(synthetic_dt, "beta[time]", group="posterior")

def test_numerical_correctness(synthetic_dt):
    # Spot-check one value from beta
    lf = spread_draws(synthetic_dt, "beta[groups]", group="posterior")
    df = lf
    
    # Find row for chain=0, draw=0, groups=0
    val = df.filter((pl.col("chain") == 0) & (pl.col("draw") == 0) & (pl.col("groups") == 0)).get_column("beta")[0]
    
    # Compare to direct xarray indexing
    expected = synthetic_dt["posterior"].beta.values[0, 0, 0]
    assert np.isclose(val, expected)

def test_numerical_correctness_2d(synthetic_dt):
    # Spot-check one value from gamma (chain, draw, time, group)
    lf = spread_draws(synthetic_dt, "gamma[time, group]", group="posterior")
    df = lf
    
    # Find row for chain=0, draw=0, time=1, group=2
    val = df.filter(
        (pl.col("chain") == 0) & 
        (pl.col("draw") == 0) & 
        (pl.col("time") == 1) & 
        (pl.col("group") == 2)
    ).get_column("gamma")[0]
    
    expected = synthetic_dt["posterior"].gamma.values[0, 0, 1, 2]
    assert np.isclose(val, expected)
