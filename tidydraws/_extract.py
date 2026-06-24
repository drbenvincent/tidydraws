import logging
import re
from typing import Tuple, List

import polars as pl
import xarray as xr

logger = logging.getLogger("tidydraws")

def _parse_var_spec(spec: str) -> Tuple[str, List[str]]:
    """
    Parse a variable specification string into (variable_name, list_of_dimensions).

    Example: "beta[groups]" -> ("beta", ["groups"])
    Example: "sigma" -> ("sigma", [])
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Variable specification cannot be empty.")

    # Regex pattern:
    # Group 1: var_name (sequence of non-bracket/non-whitespace characters)
    # Optional whitespace followed by optional [dimensions] block
    match = re.match(r"^([^\[\]\s]+)\s*(?:\[([^\]]*)\])?$", spec)
    if not match:
        raise ValueError(
            f"Malformed variable specification: '{spec}'. "
            "Expected format 'var_name' or 'var_name[dim1, dim2, ...]'."
        )

    var_name = match.group(1)
    dims_str = match.group(2)

    if dims_str is None:
        return var_name, []

    # Split dimensions by comma and strip whitespace
    dims = [d.strip() for d in dims_str.split(",") if d.strip()]

    # Handle "beta[]" case where brackets are present but empty or only contain whitespace
    if not dims and dims_str is not None:
        raise ValueError(f"Variable specification '{spec}' cannot have empty brackets.")

    return var_name, dims

def _datatree_group_to_lazy(dt, group: str) -> pl.LazyFrame:
    """
    Convert a DataTree group to a Polars LazyFrame containing all coordinates and variables.

    Warning: If the group contains many variables with different dimensions, 
    xarray's to_dataframe() can create a very large sparse DataFrame.
    """
    if group not in dt.children:
        raise KeyError(f"Group '{group}' not found in DataTree.")

    ds = dt.children[group].to_dataset()
    
    # If there are no data variables, we just want the coordinates.
    # to_dataframe() requires at least one data variable; otherwise it fails.
    if len(ds.data_vars) == 0:
        df = ds.coords.to_dataframe().reset_index()
    else:
        # Convert xarray Dataset to pandas DataFrame, then to Polars LazyFrame
        df = ds.to_dataframe().reset_index()
        
    return pl.from_pandas(df).lazy()

def _align_dims(frames: List[pl.LazyFrame], chain_dim: str = "chain", draw_dim: str = "draw") -> pl.LazyFrame:
    """
    Combine multiple LazyFrames by joining on shared dimensions.

    If frames share only (chain, draw), it performs a cross-join (broadcasting).
    """
    if not frames:
        raise ValueError("No frames provided for alignment.")

    result = frames[0]
    
    for i in range(1, len(frames)):
        next_frame = frames[i]
        
        # Find common columns (these are our shared dimensions)
        # Use collect_schema().names() to avoid PerformanceWarning from .columns
        cols_result = result.collect_schema().names()
        cols_next = next_frame.collect_schema().names()
        common_cols = list(set(cols_result).intersection(set(cols_next)))
        
        # We expect at least chain and draw to be present
        if chain_dim not in common_cols or draw_dim not in common_cols:
            raise RuntimeError(
                f"Frames must share '{chain_dim}' and '{draw_dim}' dimensions for alignment. "
                f"Found shared columns: {common_cols}"
            )

        # Check if this is a cross-join (only chain/draw shared)
        if len(common_cols) == 2:
            logger.warning(
                f"Cross-join detected between frame {i-1} and {i}. "
                f"Broadcasting scalar or differently-dimensioned variable on dims {common_cols}."
            )

        # Join on the intersection of columns
        result = result.join(next_frame, on=common_cols, how="inner")

    return result

def spread_draws(
    dt: xr.DataTree,
    *var_specs: str,
    group: str = "posterior",
    chain_dim: str = "chain",
    draw_dim: str = "draw",
) -> pl.LazyFrame:
    """
    Extract posterior draws for one or more variables into a tidy Polars LazyFrame.
    """
    if group not in dt.children:
        raise KeyError(f"Group '{group}' not found in DataTree.")

    ds = dt.children[group].to_dataset()
    
    # Ensure chain and draw dimensions are named as requested
    # (In ArviZ 1.0 they usually are, but we allow overrides)
    # This is tricky with xarray; typically we just rename the coords if needed.
    # For now, we assume they match or are handled by the dataset.

    frames = []
    for spec in var_specs:
        var_name, dims = _parse_var_spec(spec)
        
        if var_name not in ds.data_vars:
            raise KeyError(f"Variable '{var_name}' not found in group '{group}'.")

        da = ds[var_name]
        
        # Validate that the specified dims match the data array's dimensions
        # excluding chain and draw.
        actual_dims = [d for d in da.dims if d != chain_dim and d != draw_dim]
        if set(dims) != set(actual_dims):
            raise ValueError(
                f"Dimension mismatch for '{var_name}'. "
                f"Expected {dims}, but found {actual_dims}."
            )

        # Convert the specific DataArray to a LazyFrame
        # .to_dataframe() creates a multi-index DF with all coords.
        df = da.to_dataframe().reset_index()
        lf = pl.from_pandas(df).lazy()
        
        # Rename the value column to var_name
        # xarray's to_dataframe() names the value column as the variable name 
        # if it's a DataArray, but check just in case.
        # Use collect_schema().names() to avoid PerformanceWarning from .columns
        if var_name not in lf.collect_schema().names():
            # If da is scalar, it might have different naming logic
            # Let's ensure the value column is named correctly.
             lf = lf.rename({lf.collect_schema().names()[-1]: var_name})

        frames.append(lf)

    return _align_dims(frames, chain_dim=chain_dim, draw_dim=draw_dim)


def add_epred_draws(
    dt: xr.DataTree,
    newdata,
    var_name,
    idata_group="predictions",
    constant_data_group="predictions_constant_data",
    join_on="obs_ind",
) -> pl.LazyFrame:
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
    pl.LazyFrame
        Tidy LazyFrame with columns: chain, draw, [join_on cols], [covariate cols], var_name
        One row per (chain, draw, obs_ind).
    
    Examples
    --------
    # Basic usage — newdata read from dt
    pred_df = add_epred_draws(dt, newdata=None, var_name="mu")
    # → columns: chain, draw, obs_ind, x, group, mu
    # → 4 × 1000 × 80 = 320,000 rows
    
    # Filter before collecting — only group 0
    pred_df.filter(pl.col("group") == 0).collect()
    # → 4 × 1000 × 20 = 80,000 rows materialised
    
    # Provide custom newdata (e.g., a finer grid)
    fine_grid = pl.DataFrame({"x": np.linspace(0, 20, 200), "group": ...})
    add_epred_draws(dt, newdata=fine_grid, var_name="mu")
    """

    # Check if group exists in the DataTree
    if idata_group not in dt.children:
        raise KeyError(f"Group '{idata_group}' not found in DataTree.")

    # Extract the predictive draws for the specified variable
    ds = dt.children[idata_group].to_dataset()
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in group '{idata_group}'.")

    # Get the DataArray for the variable
    da = ds[var_name]

    # Convert to LazyFrame
    df = da.to_dataframe().reset_index()
    pred_lazy = pl.from_pandas(df).lazy()

    # Handle newdata parameter
    if newdata is None:
        # Check if constant_data_group exists
        if constant_data_group not in dt.children:
            raise KeyError(
                f"constant_data_group '{constant_data_group}' not found in DataTree. "
                "Pass newdata explicitly or check your DataTree structure."
            )

        # Read the constant data group
        const_lazy = _datatree_group_to_lazy(dt, constant_data_group)

        # Join pred_lazy with const_lazy on join_on
        result = pred_lazy.join(const_lazy, on=join_on, how="left")

    else:
        # Convert newdata to LazyFrame if needed
        if not isinstance(newdata, pl.LazyFrame):
            # Check if it's a pandas DataFrame or Polars DataFrame
            try:
                import pandas as pd
                if isinstance(newdata, pd.DataFrame):
                    newdata = pl.from_pandas(newdata).lazy()
                else:
                    # Try to convert to polars, might be a Polars DataFrame
                    newdata = pl.from_pandas(newdata.to_pandas()).lazy()
            except AttributeError:
                # If it fails, just try pl.from_pandas directly as a fallback
                try:
                    newdata = pl.from_pandas(newdata).lazy()
                except Exception:
                    raise TypeError("newdata must be either None, a pl.DataFrame, pd.DataFrame, or pl.LazyFrame.")

        # Join pred_lazy with newdata on join_on
        result = pred_lazy.join(newdata, on=join_on, how="left")

    return result
