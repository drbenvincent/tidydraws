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

"""Log versions and prove that the CI leg exercises its intended ArviZ backend."""

from importlib.metadata import version
import platform
import sys

import arviz as az
import xarray as xr

print(platform.platform(), sys.version)
for package in ("tidydraws", "arviz", "xarray", "numpy", "polars", "pyarrow"):
    print(f"{package}=={version(package)}")
legacy = int(version("arviz").split(".")[0]) == 0
expected = sys.argv[1]
assert legacy == (expected == "legacy"), f"Expected {expected} ArviZ"
if legacy:
    data = az.InferenceData(posterior=xr.Dataset({"x": (("chain", "draw"), [[1.0]])}))
    assert not hasattr(data, "children"), "Legacy leg must use real InferenceData"
else:
    assert hasattr(xr, "DataTree")
