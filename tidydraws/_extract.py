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

import logging

import polars as pl
import xarray as xr

logger = logging.getLogger("tidydraws")


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
    frames: list[pl.DataFrame], chain_dim: str = "chain", draw_dim: str = "draw"
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
    *var_names: str,
    group: str = "posterior",
    chain_dim: str = "chain",
    draw_dim: str = "draw",
) -> pl.DataFrame:
    """Extract posterior draws for one or more variables into a tidy Polars DataFrame.

    Non-sample dimensions (everything except ``chain`` and ``draw``) are detected
    automatically from the xarray DataArray's ``.dims`` -- no bracket syntax needed.

    Parameters
    ----------
    dt : xr.DataTree
        ArviZ DataTree object (xarray.DataTree) from PyMC sampling.
    *var_names : str
        Names of variables to extract. Dimensions are auto-detected.
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

    # Array parameter -- dims auto-detected from the DataArray
    parameter_draws(dt, "beta", "intercept")
    # -> columns: chain, draw, groups, beta, intercept
    # -> 4 x 1000 x 4 = 16,000 rows (NOT 320,000)

    # Mix of scalar and array (sigma broadcast-joined to group-level params)
    parameter_draws(dt, "beta", "sigma")
    # -> columns: chain, draw, groups, beta, sigma
    # -> 4 x 1000 x 4 = 16,000 rows; sigma repeated per group (explicit and expected)

    # Different groups (prior vs posterior)
    parameter_draws(dt, "beta", group="prior")
    # -> extract prior draws for beta

    # Multi-dimensional variable
    parameter_draws(dt, "gamma")
    # -> columns: chain, draw, time, group, gamma
    """
    if group not in dt.children:
        raise KeyError(f"Group '{group}' not found in DataTree.")

    ds = dt.children[group].to_dataset()

    frames = []
    for name in var_names:
        if name not in ds.data_vars:
            raise KeyError(f"Variable '{name}' not found in group '{group}'.")

        da = ds[name]

        # .to_dataframe() creates a multi-index DF with all coords.
        df = da.to_dataframe().reset_index()
        lf = pl.from_pandas(df)

        # xarray's to_dataframe() names the value column after the variable
        # (e.g. "beta", "sigma"). In rare cases (scalar DataArrays with
        # non-standard naming) the column may be "x" or something else --
        # normalise to the requested name.

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
        Name of the predictive variable to extract (e.g., "mu"). Dimensions
        are auto-detected from the DataArray.
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
    *var_names: str,
    groups: list[str] | None = None,
    group_name: str = "source",
) -> pl.DataFrame:
    """Extract and stack draws from multiple groups (e.g., posterior and prior).

    Calls :func:`parameter_draws` for each group, adds a column identifying the
    source group, and concatenates the results into a single DataFrame for easy
    comparison.

    Parameters
    ----------
    dt : xr.DataTree
        ArviZ DataTree object.
    *var_names : str
        Names of variables to extract (as for :func:`parameter_draws`).
    groups : list[str] | None
        Which groups to extract and stack. Defaults to ``["posterior", "prior"]``
        when ``None``.
    group_name : str
        Name of the column identifying the source group. Default "source".

    Returns
    -------
    pl.DataFrame
        Stacked draws with an additional column (group_name) indicating source.

    Example
    -------
    # Extract posterior and prior for side-by-side forest plots
    compare_df = compare_draws(dt, "beta", groups=["posterior", "prior"])
    # -> columns: chain, draw, groups, beta, source
    # -> source in {"posterior", "prior"}
    """

    if groups is None:
        groups = ["posterior", "prior"]

    # Collect results from each group
    frames = []
    for group in groups:
        # Call parameter_draws for this group
        group_frame = parameter_draws(dt, *var_names, group=group)
        # Add the source group identifier column
        group_frame = group_frame.with_columns(pl.lit(group).alias(group_name))
        frames.append(group_frame)

    # Concatenate all frames vertically using diagonal concat
    return pl.concat(frames, how="diagonal")
