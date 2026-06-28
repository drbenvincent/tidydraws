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

"""Public summary helpers for tidy draw DataFrames.

The key function is :func:`point_interval`, which computes point estimates
(median/mean) and uncertainty intervals from a tidy long-form DataFrame of
MCMC draws into one summary row per group.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

_VALID_POINT = frozenset({"median", "mean"})
_VALID_INTERVAL = frozenset({"eti", "hdi"})


def _hdi_bounds(values: np.ndarray, prob: float) -> tuple[float, float]:
    """Compute highest-density interval bounds for a 1-D array.

    Parameters
    ----------
    values : np.ndarray
        Sorted or unsorted 1-D array of draws.
    prob : float
        Probability mass to cover (0 < prob < 1).

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of the HDI.
    """
    n = len(values)
    sorted_vals = np.sort(values)
    window = max(1, int(np.ceil(prob * n)))

    if window >= n:
        return float(sorted_vals[0]), float(sorted_vals[-1])

    best_lower = sorted_vals[0]
    min_width = sorted_vals[-1] - sorted_vals[0]
    for i in range(n - window + 1):
        width = sorted_vals[i + window - 1] - sorted_vals[i]
        if width < min_width:
            min_width = width
            best_lower = sorted_vals[i]

    best_upper = best_lower + min_width
    return float(best_lower), float(best_upper)


def point_interval(
    data: pl.DataFrame | pl.LazyFrame,
    value: str,
    group_by: str | Sequence[str] | None = None,
    probs: tuple[float, ...] = (0.89,),
    point: str = "median",
    interval: str = "eti",
) -> pl.DataFrame:
    """Compute point estimates and uncertainty intervals from tidy draws.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Tidy DataFrame of MCMC draws (e.g. from :func:`parameter_draws` or
        :func:`compare_draws`). Must contain ``value`` as a column.
    value : str
        Name of the column to summarise.
    group_by : str | list[str] | None, optional
        Column(s) to group by. ``None`` (default) collapses all draws into
        a single summary row. Pass a column name (e.g. ``"groups"``) or a
        list (e.g. ``["groups", "source"]`` for combined output from
        :func:`compare_draws`).
    probs : tuple[float, ...], optional
        Probability mass for each interval width. Default ``(0.89,)``.
        Multiple values produce additional suffixed columns.
    point : str, optional
        Point estimate type: ``"median"`` (default) or ``"mean"``.
    interval : str, optional
        Interval type: ``"eti"`` (equal-tailed, default) or ``"hdi"``
        (highest-density interval).

    Returns
    -------
    pl.DataFrame
        Always returns an eager :class:`pl.DataFrame`.

        **Single prob** (e.g. ``probs=(0.89,)``):
            ``{value}``, ``{value}_lower``, ``{value}_upper``

        **Multiple probs** (e.g. ``probs=(0.5, 0.89)``):
            ``{value}``, ``{value}_lower``, ``{value}_upper``,
            ``{value}_lower_0.5``, ``{value}_upper_0.5``

    Examples
    --------
    Basic grouped summary::

        draws = parameter_draws(dt, "beta[groups]")
        summary = point_interval(draws, "beta", group_by="groups")

    Compare prior vs posterior::

        comp = compare_draws(dt, "beta[groups]")
        summary = point_interval(comp, "beta", group_by=["groups", "source"])
    """
    # Coerce to eager DataFrame — the data layer is eager by design
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    if value not in data.columns:
        raise ValueError(
            f"Column '{value}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    if point not in _VALID_POINT:
        raise ValueError(
            f"Unknown point type '{point}'. Must be one of {sorted(_VALID_POINT)}."
        )

    if interval not in _VALID_INTERVAL:
        raise ValueError(
            f"Unknown interval type '{interval}'. "
            f"Must be one of {sorted(_VALID_INTERVAL)}."
        )

    if not probs:
        raise ValueError("At least one probability must be specified in `probs`.")

    for p in probs:
        if not 0 < p < 1:
            raise ValueError(f"Each probability in `probs` must be in (0, 1). Got {p}.")

    # Normalise group_by to a list
    gb: list[str] = []
    if group_by is not None:
        gb = [group_by] if isinstance(group_by, str) else list(group_by)
        for col in gb:
            if col not in data.columns:
                raise ValueError(
                    f"Group-by column '{col}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )

    # ----- Point estimate expression -----
    if point == "median":
        point_expr: pl.Expr = pl.col(value).median().alias(value)
    else:
        point_expr = pl.col(value).mean().alias(value)

    single_prob = len(probs) == 1

    if interval == "eti":
        # ----- ETI: pure Polars quantile expressions (fast) -----
        interval_exprs: list[pl.Expr] = []
        for prob in probs:
            suffix = "" if single_prob else f"_{prob:.2f}"
            tail = (1.0 - prob) / 2.0
            interval_exprs.append(
                pl.col(value).quantile(tail).alias(f"{value}_lower{suffix}")
            )
            interval_exprs.append(
                pl.col(value).quantile(1.0 - tail).alias(f"{value}_upper{suffix}")
            )

        all_exprs = [point_expr, *interval_exprs]

        if gb:
            return data.group_by(gb).agg(all_exprs)
        else:
            return data.select(all_exprs)

    else:
        # ----- HDI: post-processing per group -----
        # First compute point estimate (pure Polars)
        if gb:
            point_df = data.group_by(gb).agg(point_expr)
        else:
            point_df = data.select(point_expr)

        # Then compute HDI bounds per group via numpy
        bounds_pieces: list[pl.DataFrame] = []
        if gb:
            grouped = data.group_by(gb)
            for group_keys, group_df in grouped:
                # group_keys is a tuple for multiple group columns
                if len(gb) == 1:
                    key_dict = {gb[0]: group_keys[0]}
                else:
                    key_dict = dict(zip(gb, group_keys))

                for prob in probs:
                    suffix = "" if single_prob else f"_{prob:.2f}"
                    lower, upper = _hdi_bounds(
                        group_df.get_column(value).to_numpy(), prob
                    )
                    key_dict[f"{value}_lower{suffix}"] = lower
                    key_dict[f"{value}_upper{suffix}"] = upper

                bounds_pieces.append(pl.DataFrame(key_dict))

            bounds_df = pl.concat(bounds_pieces)
            result = point_df.join(bounds_df, on=gb, how="left")
        else:
            # Ungrouped — compute on the whole column
            row: dict = {}
            for prob in probs:
                suffix = "" if single_prob else f"_{prob:.2f}"
                lower, upper = _hdi_bounds(data.get_column(value).to_numpy(), prob)
                row[f"{value}_lower{suffix}"] = lower
                row[f"{value}_upper{suffix}"] = upper
            bounds_df = pl.DataFrame(row)
            result = point_df.with_columns(bounds_df)

        return result
