import pytest
import polars as pl
import xarray as xr
import numpy as np
from tidydraws import spread_draws_compare


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

    # Create DataTree (available in xarray >= 2024.x / current versions)
    dt = xr.DataTree()
    dt["posterior"] = posterior
    dt["prior"] = posterior * 0.5 # Simple different data for prior
    
    # Add a custom group for testing
    dt["prior_pred"] = posterior * 0.3

    return dt


def test_spread_draws_compare_basic(synthetic_dt):
    # Test basic functionality with default groups
    lf = spread_draws_compare(synthetic_dt, "beta[groups]")
    df = lf
    
    # Should have 2 * 5 * 3 * 2 (chains * draws * groups * groups) rows 
    # since we're comparing posterior and prior
    assert df.height == 2 * 5 * 3 * 2
    
    # Check that we have the source column
    assert "source" in df.columns
    assert set(df.get_column("source").unique().to_list()) == {"posterior", "prior"}
    
    # Check that we have the expected columns  
    assert "chain" in df.columns
    assert "draw" in df.columns
    assert "groups" in df.columns
    assert "beta" in df.columns


def test_spread_draws_compare_custom_groups(synthetic_dt):
    # Test with custom groups including a custom group
    lf = spread_draws_compare(synthetic_dt, "beta[groups]", groups=["posterior", "prior", "prior_pred"])
    df = lf
    
    # Should have 2 * 5 * 3 * 3 (chains * draws * groups * groups) rows 
    assert df.height == 2 * 5 * 3 * 3
    
    # Check that we have the source column with correct values
    assert "source" in df.columns
    assert set(df.get_column("source").unique().to_list()) == {"posterior", "prior", "prior_pred"}
    

def test_spread_draws_compare_multiple_vars(synthetic_dt):
    # Test with multiple variables
    lf = spread_draws_compare(synthetic_dt, "beta[groups]", "sigma")
    df = lf
    
    # Should have 2 * 5 * 3 * 2 (chains * draws * groups * groups) rows 
    assert df.height == 2 * 5 * 3 * 2
    
    # Check that we have the expected columns  
    assert "chain" in df.columns
    assert "draw" in df.columns
    assert "groups" in df.columns
    assert "beta" in df.columns
    assert "sigma" in df.columns
    assert "source" in df.columns


def test_spread_draws_compare_custom_group_name(synthetic_dt):
    # Test with custom group column name
    lf = spread_draws_compare(synthetic_dt, "beta[groups]", group_name="model_type")
    df = lf
    
    # Check that we have the custom group column
    assert "model_type" in df.columns
    assert "source" not in df.columns


def test_spread_draws_compare_eager_semantics(synthetic_dt):
    # Verify return type is pl.DataFrame (eager)
    df = spread_draws_compare(synthetic_dt, "beta[groups]")
    assert isinstance(df, pl.DataFrame)

    # Eager frames expose .height directly
    assert df.height > 0


def test_spread_draws_compare_error_invalid_group(synthetic_dt):
    # Test error handling for non-existent group
    with pytest.raises(KeyError, match="Group 'nonexistent' not found"):
        spread_draws_compare(synthetic_dt, "sigma", groups=["nonexistent"])


def test_spread_draws_compare_error_malformed_spec(synthetic_dt):
    # Test error handling for malformed spec (should be passed through from spread_draws)
    with pytest.raises(ValueError, match="Malformed variable specification"):
        spread_draws_compare(synthetic_dt, "beta[groups")


def test_spread_draws_compare_numerical_correctness(synthetic_dt):
    # Spot-check data integrity
    lf = spread_draws_compare(synthetic_dt, "beta[groups]")
    df = lf
    
    # Check some values from posterior group
    posterior_rows = df.filter(pl.col("source") == "posterior")
    assert posterior_rows.height > 0
    
    # Check that values make sense (non-zero) 
    beta_values = posterior_rows.get_column("beta")
    assert len(beta_values) > 0
    assert not all(val == 0 for val in beta_values)
