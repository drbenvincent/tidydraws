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
