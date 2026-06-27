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


def _datatree_group_to_df(dt, group: str) -> pl.DataFrame:
    """
    Convert a DataTree group to a Polars DataFrame containing all coordinates and variables.

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
        # Convert xarray Dataset to pandas DataFrame, then to Polars DataFrame
        df = ds.to_dataframe().reset_index()

    return pl.from_pandas(df)


def _align_dims(
    frames: List[pl.DataFrame], chain_dim: str = "chain", draw_dim: str = "draw"
) -> pl.DataFrame:
    """
    Combine multiple DataFrames by joining on shared dimensions.

    If frames share only (chain, draw), it performs a cross-join (broadcasting).
    """
    if not frames:
        raise ValueError("No frames provided for alignment.")

    result = frames[0]

    for i in range(1, len(frames)):
        next_frame = frames[i]

        # Find common columns (these are our shared dimensions)
        common_cols = list(set(result.columns).intersection(set(next_frame.columns)))

        # We expect at least chain and draw to be present
        if chain_dim not in common_cols or draw_dim not in common_cols:
            raise RuntimeError(
                f"Frames must share '{chain_dim}' and '{draw_dim}' dimensions for alignment. "
                f"Found shared columns: {common_cols}"
            )

        # Check if this is a cross-join (only chain/draw shared)
        if len(common_cols) == 2:
            logger.warning(
                f"Cross-join detected between frame {i - 1} and {i}. "
                f"Broadcasting scalar or differently-dimensioned variable on dims {common_cols}."
            )

        # Join on the intersection of columns
        result = result.join(next_frame, on=common_cols, how="inner")

    return result


def parameter_draws(
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
    parameter_draws(dt, "sigma")
    # -> columns: chain, draw, sigma
    # -> 4 x 1000 = 4,000 rows

    # Array parameter spread over a named dim
    parameter_draws(dt, "beta[groups]", "intercept[groups]")
    # -> columns: chain, draw, groups, beta, intercept
    # -> 4 x 1000 x 4 = 16,000 rows (NOT 320,000)

    # Mix of scalar and array (sigma broadcast-joined to group-level params)
    parameter_draws(dt, "beta[groups]", "sigma")
    # -> columns: chain, draw, groups, beta, sigma
    # -> 4 x 1000 x 4 = 16,000 rows; sigma repeated per group (explicit and expected)

    # Different groups (prior vs posterior)
    parameter_draws(dt, "beta[groups]", group="prior")
    # -> extract prior draws for beta

    # Nested dimensions
    parameter_draws(dt, "gamma[time, group]")
    # -> columns: chain, draw, time, group, gamma
    """
    if group not in dt.children:
        raise KeyError(f"Group '{group}' not found in DataTree.")

    ds = dt.children[group].to_dataset()

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

        # Convert the specific DataArray to a DataFrame
        # .to_dataframe() creates a multi-index DF with all coords.
        df = da.to_dataframe().reset_index()
        lf = pl.from_pandas(df)

        # Rename the value column to var_name
        # xarray's to_dataframe() names the value column as the variable name
        # if it's a DataArray, but check just in case.
        if var_name not in lf.columns:
            # If da is scalar, it might have different naming logic
            # Let's ensure the value column is named correctly.
            lf = lf.rename({lf.columns[-1]: var_name})

        frames.append(lf)

    return _align_dims(frames, chain_dim=chain_dim, draw_dim=draw_dim)


def prediction_draws(
    dt: xr.DataTree,
    newdata,
    var_name,
    idata_group="predictions",
    constant_data_group="predictions_constant_data",
    join_on="obs_ind",
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
        prediction draws. Default "predictions_constant_data" (ArviZ 1.x
        convention for the constant data paired with a "predictions" group).
        Set this parameter if your DataTree uses a different naming convention.
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
    pred_df = prediction_draws(dt, newdata=None, var_name="mu")
    # -> columns: chain, draw, obs_ind, x, group, mu
    # -> 4 x 1000 x 80 = 320,000 rows

    # Filter before plotting — only group 0
    pred_df.filter(pl.col("group") == 0)
    # -> 4 x 1000 x 20 = 80,000 rows

    # Provide custom newdata (e.g., a finer grid)
    fine_grid = pl.DataFrame({"x": np.linspace(0, 20, 200), "group": ...})
    prediction_draws(dt, newdata=fine_grid, var_name="mu")
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

    # Convert to DataFrame
    df = da.to_dataframe().reset_index()
    pred_df = pl.from_pandas(df)

    # Handle newdata parameter
    if newdata is None:
        # Check if constant_data_group exists
        if constant_data_group not in dt.children:
            raise KeyError(
                f"constant_data_group '{constant_data_group}' not found in DataTree. "
                "Pass newdata explicitly or check your DataTree structure."
            )

        # Read the constant data group
        const_df = _datatree_group_to_df(dt, constant_data_group)

        # Join pred_df with const_df on join_on
        result = pred_df.join(const_df, on=join_on, how="left")

    else:
        newdata_df = _coerce_to_dataframe(newdata)
        # Join pred_df with newdata on join_on
        result = pred_df.join(newdata_df, on=join_on, how="left")

    return result


def _coerce_to_dataframe(newdata) -> pl.DataFrame:
    """Coerce newdata (pl.DataFrame, pl.LazyFrame, or pd.DataFrame) to a Polars DataFrame."""
    if isinstance(newdata, pl.DataFrame):
        return newdata
    if isinstance(newdata, pl.LazyFrame):
        return newdata.collect()
    try:
        import pandas as pd

        if isinstance(newdata, pd.DataFrame):
            return pl.from_pandas(newdata)
    except ImportError:
        pass
    raise TypeError(
        "newdata must be one of: pl.DataFrame, pl.LazyFrame, pd.DataFrame, or None."
    )


def compare_draws(
    dt: xr.DataTree,
    *var_specs: str,
    groups: list[str] = ["posterior", "prior"],
    group_name: str = "source",
) -> pl.DataFrame:
    """
    Extract and stack draws from multiple groups (e.g., posterior and prior).

    Calls parameter_draws() for each group, adds a column identifying the source group,
    and concatenates the results into a single DataFrame for easy comparison.

    Parameters
    ----------
    dt : xr.DataTree
        ArviZ DataTree object.
    *var_specs : str
        Variable specifications (as for parameter_draws()).
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
    compare_df = compare_draws(dt, "beta[groups]",
                                       groups=["posterior", "prior"])
    # -> columns: chain, draw, groups, beta, source
    # -> source in {"posterior", "prior"}
    """

    # Collect results from each group
    frames = []
    for group in groups:
        # Call parameter_draws for this group
        group_frame = parameter_draws(dt, *var_specs, group=group)
        # Add the source group identifier column
        group_frame = group_frame.with_columns(pl.lit(group).alias(group_name))
        frames.append(group_frame)

    # Concatenate all frames vertically using diagonal concat
    return pl.concat(frames, how="diagonal")
