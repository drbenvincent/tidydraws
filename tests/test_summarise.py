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
import numpy as np
from tidydraws import point_interval


# ---------------------------------------------------------------------------
# Fixtures — tidy DataFrames that mimic parameter_draws / compare_draws output
# ---------------------------------------------------------------------------


@pytest.fixture
def tidy_df():
    """A tidy draws frame: chain=2, draw=5, groups=3, beta + sigma columns."""
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_groups = 2, 5, 3
    return pl.DataFrame({
        "chain": np.repeat(np.arange(n_chains), n_draws * n_groups),
        "draw": np.tile(np.repeat(np.arange(n_draws), n_groups), n_chains),
        "groups": np.tile(np.arange(n_groups), n_chains * n_draws),
        "beta": rng.normal(0, 1, n_chains * n_draws * n_groups),
        "sigma": rng.exponential(1, n_chains * n_draws * n_groups),
    })


@pytest.fixture
def tidy_with_source(tidy_df):
    """A draws frame with a 'source' column (as from compare_draws)."""
    posterior = tidy_df.with_columns(pl.lit("posterior").alias("source"))
    prior = tidy_df.with_columns(
        pl.lit("prior").alias("source"),
        (pl.col("beta") * 0.5).alias("beta"),
    )
    return pl.concat([posterior, prior])


# ---------------------------------------------------------------------------
# Single-prob (default) — column naming & row counts
# ---------------------------------------------------------------------------


class TestSingleProbDefaults:
    """Default probs=(0.89,) — columns have no prob suffix."""

    def test_ungrouped(self, tidy_df):
        result = point_interval(tidy_df, "beta")
        assert isinstance(result, pl.DataFrame)
        # Single summary row
        assert result.height == 1
        # Core columns
        assert result.columns == ["beta", "beta_lower", "beta_upper"]
        # Point estimate should be a valid float
        assert np.isfinite(result.get_column("beta")[0])

    def test_grouped(self, tidy_df):
        result = point_interval(tidy_df, "beta", group_by="groups")
        assert result.height == 3  # three groups
        assert set(result.columns) == {"groups", "beta", "beta_lower", "beta_upper"}
        # Groups in order
        assert set(result.get_column("groups").to_list()) == {0, 1, 2}

    def test_median_correctness(self, tidy_df):
        """Verify median matches manual computation."""
        expected = (
            tidy_df
            .group_by("groups")
            .agg(pl.col("beta").median().alias("beta"))
            .sort("groups")
        )
        result = point_interval(tidy_df, "beta", group_by="groups").sort("groups")
        assert np.allclose(
            result.get_column("beta").to_numpy(),
            expected.get_column("beta").to_numpy(),
        )

    def test_eti_correctness(self, tidy_df):
        """Verify ETI bounds match manual quantiles for probs=(0.89,)."""
        tail = 0.055  # (1 - 0.89) / 2
        expected = (
            tidy_df
            .group_by("groups")
            .agg([
                pl.col("beta").quantile(tail).alias("beta_lower"),
                pl.col("beta").quantile(1 - tail).alias("beta_upper"),
            ])
            .sort("groups")
        )
        result = point_interval(tidy_df, "beta", group_by="groups").sort("groups")
        assert np.allclose(
            result.get_column("beta_lower").to_numpy(),
            expected.get_column("beta_lower").to_numpy(),
        )
        assert np.allclose(
            result.get_column("beta_upper").to_numpy(),
            expected.get_column("beta_upper").to_numpy(),
        )


# ---------------------------------------------------------------------------
# Boolean draws — indicator variables summarised as proportions
# ---------------------------------------------------------------------------


class TestBooleanColumn:
    @pytest.fixture
    def bool_df(self):
        rng = np.random.default_rng(0)
        n = 50
        return pl.DataFrame({
            "chain": [0] * n,
            "draw": list(range(n)),
            "groups": ["A"] * n,
            "success": rng.integers(0, 2, n).astype(bool),
        })

    def test_hdi_on_boolean(self, bool_df):
        # HDI previously crashed with "numpy boolean subtract" on bool columns.
        result = point_interval(bool_df, "success", group_by="groups", interval="hdi")
        assert result.schema["success"] == pl.Float64

    def test_matches_manual_float_cast(self, bool_df):
        result = point_interval(bool_df, "success", group_by="groups", interval="hdi")
        casted = point_interval(
            bool_df.with_columns(pl.col("success").cast(pl.Float64)),
            "success",
            group_by="groups",
            interval="hdi",
        )
        assert result.equals(casted)


# ---------------------------------------------------------------------------
# Multiple probs — column naming
# ---------------------------------------------------------------------------


class TestMultipleProbs:
    def test_two_probs_all_suffixed(self, tidy_df):
        result = point_interval(tidy_df, "beta", probs=(0.5, 0.89))
        expected_cols = {
            "beta",
            "beta_lower_0.50",
            "beta_upper_0.50",
            "beta_lower_0.89",
            "beta_upper_0.89",
        }
        assert set(result.columns) == expected_cols

    def test_three_probs_columns(self, tidy_df):
        result = point_interval(tidy_df, "beta", probs=(0.5, 0.8, 0.95))
        expected = {"beta"}
        for p in (0.5, 0.8, 0.95):
            expected.add(f"beta_lower_{p:.2f}")
            expected.add(f"beta_upper_{p:.2f}")
        assert set(result.columns) == expected


# ---------------------------------------------------------------------------
# Point estimate variants
# ---------------------------------------------------------------------------


class TestPointEstimate:
    def test_mean(self, tidy_df):
        result = point_interval(tidy_df, "beta", group_by="groups", point="mean")
        expected = (
            tidy_df
            .group_by("groups")
            .agg(pl.col("beta").mean().alias("beta"))
            .sort("groups")
        )
        assert np.allclose(
            result.sort("groups").get_column("beta").to_numpy(),
            expected.get_column("beta").to_numpy(),
        )


# ---------------------------------------------------------------------------
# Group-by variants
# ---------------------------------------------------------------------------


class TestGroupBy:
    def test_string(self, tidy_df):
        result = point_interval(tidy_df, "beta", group_by="groups")
        assert result.height == 3

    def test_list(self, tidy_with_source):
        result = point_interval(tidy_with_source, "beta", group_by=["groups", "source"])
        # 3 groups x 2 sources = 6 rows
        assert result.height == 6
        assert "source" in result.columns
        assert "groups" in result.columns


# ---------------------------------------------------------------------------
# Integration with compare_draws
# ---------------------------------------------------------------------------


class TestCompareDrawsIntegration:
    def test_compare_draws_chain(self, tidy_with_source):
        result = point_interval(tidy_with_source, "beta", group_by=["groups", "source"])
        # One row per (group, source) combination
        distinct_sources = tidy_with_source.get_column("source").unique().to_list()
        assert result.height == 3 * len(distinct_sources)
        assert "beta" in result.columns
        assert "beta_lower" in result.columns
        assert "beta_upper" in result.columns


# ---------------------------------------------------------------------------
# Return type — eager only
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_dataframe_in_dataframe_out(self, tidy_df):
        result = point_interval(tidy_df, "beta", group_by="groups")
        assert isinstance(result, pl.DataFrame)

    def test_lazyframe_input(self, tidy_df):
        lazy = tidy_df.lazy()
        result = point_interval(lazy, "beta", group_by="groups")
        assert isinstance(result, pl.DataFrame)
        assert result.height == 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_bad_column(self, tidy_df):
        with pytest.raises(ValueError, match="Column.*not found"):
            point_interval(tidy_df, "nonexistent")

    def test_bad_point(self, tidy_df):
        with pytest.raises(ValueError, match="Unknown point type"):
            point_interval(tidy_df, "beta", point="mode")

    def test_bad_interval(self, tidy_df):
        with pytest.raises(ValueError, match="Unknown interval type"):
            point_interval(tidy_df, "beta", interval="hpd")

    def test_empty_probs(self, tidy_df):
        with pytest.raises(ValueError, match="At least one probability"):
            point_interval(tidy_df, "beta", probs=())

    def test_prob_out_of_range(self, tidy_df):
        with pytest.raises(ValueError, match="must be in"):
            point_interval(tidy_df, "beta", probs=(1.5,))

    def test_bad_group_by_column(self, tidy_df):
        with pytest.raises(ValueError, match="Group-by column.*not found"):
            point_interval(tidy_df, "beta", group_by="missing_col")

    def test_prob_collision(self, tidy_df):
        with pytest.raises(ValueError, match="produce the same suffix"):
            point_interval(tidy_df, "beta", probs=(0.891, 0.894))

    def test_duplicate_group_by(self, tidy_df):
        with pytest.raises(ValueError, match="Duplicate columns"):
            point_interval(tidy_df, "beta", group_by=["groups", "groups"])


# ---------------------------------------------------------------------------
# HDI interval type (smoke)
# ---------------------------------------------------------------------------


class TestHDI:
    def test_hdi_bounds_order(self, tidy_df):
        """HDI bounds should be: lower <= median, upper >= median."""
        result = point_interval(tidy_df, "beta", interval="hdi")
        assert result.get_column("beta_lower")[0] <= result.get_column("beta")[0]
        assert result.get_column("beta_upper")[0] >= result.get_column("beta")[0]

    def test_hdi_grouped(self, tidy_df):
        result = point_interval(tidy_df, "beta", group_by="groups", interval="hdi")
        assert result.height == 3
        for row in result.iter_rows(named=True):
            assert row["beta_lower"] <= row["beta"] <= row["beta_upper"]

    def test_hdi_multi_prob(self, tidy_df):
        """HDI with multiple probs — all suffixed columns."""
        result = point_interval(tidy_df, "beta", probs=(0.5, 0.89), interval="hdi")
        assert "beta_lower_0.50" in result.columns
        assert "beta_upper_0.50" in result.columns
        assert "beta_lower_0.89" in result.columns
        assert "beta_upper_0.89" in result.columns
        # The widest (89%) HDI must contain the median
        assert result.get_column("beta_lower_0.89")[0] <= result.get_column("beta")[0]
        assert result.get_column("beta_upper_0.89")[0] >= result.get_column("beta")[0]

    def test_hdi_grouped_multi_prob(self, tidy_df):
        """HDI with group_by + multiple probs."""
        result = point_interval(
            tidy_df, "beta", group_by="groups", probs=(0.5, 0.89), interval="hdi"
        )
        assert result.height == 3
        assert "beta_lower_0.50" in result.columns
        assert "beta_lower_0.89" in result.columns
        for row in result.iter_rows(named=True):
            # The widest (89%) HDI must contain the median
            assert row["beta_lower_0.89"] <= row["beta"] <= row["beta_upper_0.89"]
