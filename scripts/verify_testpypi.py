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

"""Verify the exact uploaded wheel from TestPyPI before production publication."""

import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from urllib.parse import urlparse
import zipfile

from check_artifacts import installed_smoke


def verify(directory: Path, attempts: int = 6, delay: float = 10) -> None:
    (wheel,) = directory.glob("tidydraws-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        ]
        from email.parser import BytesParser

        expected = BytesParser().parsebytes(archive.read(metadata))["Version"]
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    for attempt in range(attempts):
        try:
            with urlopen(
                f"https://test.pypi.org/pypi/tidydraws/{expected}/json", timeout=30
            ) as response:
                data = json.load(response)
            (entry,) = [item for item in data["urls"] if item["filename"] == wheel.name]
            assert entry["digests"]["sha256"] == digest, (
                "TestPyPI wheel differs from validated wheel"
            )
            assert urlparse(entry["url"]).hostname == "test-files.pythonhosted.org"
            with urlopen(entry["url"], timeout=60) as response:
                uploaded = response.read()
            assert hashlib.sha256(uploaded).hexdigest() == digest
            # Use only bytes retrieved from TestPyPI for the package itself.
            # uv obtains its dependencies separately from the default PyPI index.
            verified = directory / "testpypi" / wheel.name
            verified.parent.mkdir(exist_ok=True)
            verified.write_bytes(uploaded)
            installed_smoke(verified.resolve(), expected)
            return
        except (HTTPError, URLError, ValueError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(
                    f"TestPyPI {expected} unavailable after {attempts} attempts"
                ) from error
            time.sleep(delay)


if __name__ == "__main__":
    import sys

    verify(Path(sys.argv[1]).resolve())
