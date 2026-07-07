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

"""Tests that verify the backend dispatch works for both DataTree and InferenceData.

We construct a duck-type ``InferenceData`` object (attribute-access groups, no
``.children``) and run the same assertions against both backends. This proves the
``_get_group`` / ``_has_group`` dispatch without needing two arviz versions.
"""

import numpy as np
import polars as pl
import pytest
import xarray as xr

from tidydraws import compare_draws, parameter_draws, prediction_draws


# ---------------------------------------------------------------------------
# Duck-type InferenceData
# ---------------------------------------------------------------------------


class _InferenceDataMock:
    """Minimal duck-type of ``arviz.InferenceData``: groups via attribute access.

    ``hasattr(dt, "children")`` is ``False``, which triggers the InferenceData
    branch in ``_get_group`` / ``_has_group``.  Every group is an ``xr.Dataset``.
    """

    def __init__(self, **groups: xr.Dataset):
        self.__dict__.update(groups)


def _datatree_to_idata(dt: xr.DataTree, *groups: str) -> _InferenceDataMock:
    """Convert selected DataTree groups to an InferenceData duck-type."""
    return _InferenceDataMock(**{g: dt.children[g].to_dataset() for g in groups})


# ---------------------------------------------------------------------------
# Shared data builders
# ---------------------------------------------------------------------------


def _make_parameter_data():
    """Return (datatree, idata) with posterior, prior, and prior_pred groups."""
    chains = np.arange(2)
    draws = np.arange(5)
    groups = np.arange(3)
    times = np.arange(2)

    sigma_data = np.random.randn(2, 5)
    beta_data = np.random.randn(2, 5, 3)
    gamma_data = np.random.randn(2, 5, 2, 3)

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
            "group": groups,
        },
    )

    dt = xr.DataTree()
    dt["posterior"] = posterior
    dt["prior"] = posterior * 0.5
    dt["prior_pred"] = posterior * 0.3

    idata = _InferenceDataMock(
        posterior=posterior,
        prior=posterior * 0.5,
        prior_pred=posterior * 0.3,
    )

    return dt, idata


def _make_prediction_data():
    """Return (datatree, idata) with predictions and predictions_constant_data groups."""
    chains = np.arange(2)
    draws = np.arange(5)
    obs_inds = np.arange(20)
    groups = np.arange(3)

    mu_data = np.random.randn(2, 5, 20)
    predictions_ds = xr.Dataset(
        {"mu": (["chain", "draw", "obs_ind"], mu_data)},
        coords={"chain": chains, "draw": draws, "obs_ind": obs_inds},
    )
    constant_data_ds = xr.Dataset(
        {
            "x": (["obs_ind"], np.random.randn(20)),
            "group": (["obs_ind"], np.random.choice(groups, 20)),
        },
        coords={"obs_ind": obs_inds},
    )

    dt = xr.DataTree()
    dt["predictions"] = predictions_ds
    dt["predictions_constant_data"] = constant_data_ds

    idata = _InferenceDataMock(
        predictions=predictions_ds,
        predictions_constant_data=constant_data_ds,
    )

    return dt, idata


# ---------------------------------------------------------------------------
# parameter_draws — both backends
# ---------------------------------------------------------------------------

BACKENDS = pytest.mark.parametrize("backend", ["datatree", "inferencedata"])


def _get_parameter_backend(request):
    dt, idata = _make_parameter_data()
    return dt if request.param == "datatree" else idata


@pytest.fixture(params=["datatree", "inferencedata"])
def parameter_backend(request):
    return _get_parameter_backend(request)


class TestParameterDrawsCompat:
    def test_row_count_scalar(self, parameter_backend):
        df = parameter_draws(parameter_backend, "sigma", group="posterior")
        assert df.height == 2 * 5

    def test_row_count_1d(self, parameter_backend):
        df = parameter_draws(parameter_backend, "beta", group="posterior")
        assert df.height == 2 * 5 * 3

    def test_row_count_2d(self, parameter_backend):
        df = parameter_draws(parameter_backend, "gamma", group="posterior")
        assert df.height == 2 * 5 * 2 * 3

    def test_row_count_cross_dim(self, parameter_backend):
        df = parameter_draws(parameter_backend, "beta", "sigma", group="posterior")
        assert df.height == 2 * 5 * 3

    def test_eager_semantics(self, parameter_backend):
        df = parameter_draws(parameter_backend, "sigma", group="posterior")
        assert isinstance(df, pl.DataFrame)

    def test_filtering(self, parameter_backend):
        df = parameter_draws(parameter_backend, "beta", group="posterior")
        filtered = df.filter(pl.col("groups") == 0)
        assert filtered.height == 10

    def test_numerical_correctness(self, parameter_backend):
        from tidydraws._extract import _get_group

        df = parameter_draws(parameter_backend, "beta", group="posterior")
        val = df.filter(
            (pl.col("chain") == 0) & (pl.col("draw") == 0) & (pl.col("groups") == 0)
        ).get_column("beta")[0]

        ds = _get_group(parameter_backend, "posterior")
        expected = ds.beta.values[0, 0, 0]
        assert np.isclose(val, expected)

    def test_prior_group(self, parameter_backend):
        """Groups other than posterior must also work."""
        df = parameter_draws(parameter_backend, "beta", group="prior")
        assert df.height == 2 * 5 * 3
        assert "beta" in df.columns

    def test_custom_group(self, parameter_backend):
        """Custom group (prior_pred) must work on both backends."""
        # DataTree has prior_pred, InferenceData mock does too.
        df = parameter_draws(parameter_backend, "sigma", group="prior_pred")
        assert df.height == 2 * 5
        assert "sigma" in df.columns


# ---------------------------------------------------------------------------
# compare_draws — both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["datatree", "inferencedata"])
def compare_backend(request):
    dt, idata = _make_parameter_data()
    return dt if request.param == "datatree" else idata


class TestCompareDrawsCompat:
    def test_basic(self, compare_backend):
        df = compare_draws(compare_backend, "beta")
        assert df.height == 2 * 5 * 3 * 2  # chains × draws × groups × (posterior+prior)
        assert "beta" in df.columns
        assert "source" in df.columns
        sources = set(df["source"].to_list())
        assert sources == {"posterior", "prior"}

    def test_custom_groups(self, compare_backend):
        df = compare_draws(compare_backend, "beta", groups=["posterior", "prior_pred"])
        sources = set(df["source"].to_list())
        assert sources == {"posterior", "prior_pred"}

    def test_multiple_vars(self, compare_backend):
        df = compare_draws(compare_backend, "beta", "sigma")
        assert "beta" in df.columns
        assert "sigma" in df.columns
        assert "source" in df.columns

    def test_custom_group_name(self, compare_backend):
        df = compare_draws(compare_backend, "beta", group_name="model_type")
        assert "model_type" in df.columns
        assert "source" not in df.columns

    def test_eager_semantics(self, compare_backend):
        df = compare_draws(compare_backend, "beta")
        assert isinstance(df, pl.DataFrame)

    def test_error_invalid_group(self, compare_backend):
        with pytest.raises(KeyError, match="Group 'nonexistent' not found"):
            compare_draws(compare_backend, "sigma", groups=["nonexistent"])

    def test_error_variable_not_found(self, compare_backend):
        with pytest.raises(KeyError, match="Variable 'missing' not found"):
            compare_draws(compare_backend, "missing")


# ---------------------------------------------------------------------------
# prediction_draws — both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["datatree", "inferencedata"])
def prediction_backend(request):
    dt, idata = _make_prediction_data()
    return dt if request.param == "datatree" else idata


class TestPredictionDrawsCompat:
    def test_join_correctness(self, prediction_backend):
        result = prediction_draws(prediction_backend, newdata=None, var_name="mu")
        assert result.height == 2 * 5 * 20
        assert "mu" in result.columns
        assert "x" in result.columns
        assert "group" in result.columns
        assert "obs_ind" in result.columns

    def test_eager_semantics(self, prediction_backend):
        result = prediction_draws(prediction_backend, newdata=None, var_name="mu")
        assert isinstance(result, pl.DataFrame)

    def test_error_missing_variable(self, prediction_backend):
        with pytest.raises(KeyError, match="Variable 'nonexistent' not found"):
            prediction_draws(prediction_backend, newdata=None, var_name="nonexistent")

    def test_error_missing_group(self, prediction_backend):
        # Build an object without predictions_constant_data
        dt = _make_prediction_data()[0]  # DataTree
        idata_no_const = _InferenceDataMock(
            predictions=dt.children["predictions"].to_dataset(),
        )
        with pytest.raises(KeyError, match="predictions_constant_data.*not found"):
            prediction_draws(idata_no_const, newdata=None, var_name="mu")

    def test_newdata_explicit(self, prediction_backend):
        import pandas as pd

        newdata = pd.DataFrame({
            "obs_ind": np.arange(20),
            "x": np.random.randn(20),
            "group": np.random.choice([0, 1, 2], 20),
        })
        result = prediction_draws(prediction_backend, newdata=newdata, var_name="mu")
        assert result.height == 2 * 5 * 20
        assert "mu" in result.columns
