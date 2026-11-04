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

"""Run with an installed artifact's Python, with isolation and no checkout imports."""

from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import polars as pl
import xarray as xr
import tidydraws as td


def main(expected: str, extra: str = "") -> None:
    module = Path(td.__file__).resolve()
    assert module.is_relative_to(Path(sys.prefix).resolve()), module
    assert td.__version__ == version("tidydraws") == expected
    exports = {"parameter_draws", "prediction_draws", "compare_draws", "point_interval"}
    assert exports <= set(td.__all__)
    assert all(callable(getattr(td, name)) for name in exports)
    ds = xr.Dataset(
        {"beta": (("chain", "draw", "groups"), np.arange(12).reshape(2, 3, 2))},
        coords={"chain": [0, 1], "draw": [0, 1, 2], "groups": ["a", "b"]},
    )
    tree = xr.DataTree.from_dict({"posterior": ds, "prior": ds, "custom": ds})
    draws = td.parameter_draws(tree, "beta", group="custom")
    assert isinstance(draws, pl.DataFrame) and draws.height == 12
    assert draws["beta"].sum() == 66
    assert td.compare_draws(tree, "beta").height == 24
    grid = pl.DataFrame({"groups": ["a", "b"], "x": [1.0, 2.0]})
    predicted = td.prediction_draws(
        tree, grid, "beta", idata_group="posterior", join_on="groups"
    )
    assert isinstance(predicted, pl.DataFrame) and predicted.height == 12
    assert predicted["x"].sum() == 18
    summary = td.point_interval(draws, "beta", group_by="groups", probs=(0.5,))
    assert summary.height == 2
    if extra == "letsplot":
        from lets_plot import ggplot, aes, geom_point

        assert ggplot(draws.to_pandas(), aes("groups", "beta")) + geom_point()
    elif extra == "plotnine":
        from plotnine import ggplot, aes, geom_point

        assert ggplot(draws.to_pandas(), aes("groups", "beta")) + geom_point()
    elif extra == "altair":
        import altair as alt

        chart = (
            alt.Chart(draws.to_pandas()).mark_point().encode(x="groups:N", y="beta:Q")
        )
        assert chart.to_dict()["mark"]
    print(f"Verified tidydraws {expected} ({extra or 'base'}): {module}")


if __name__ == "__main__":
    main(*sys.argv[1:])
