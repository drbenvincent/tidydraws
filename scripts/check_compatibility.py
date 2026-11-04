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

"""Test a disposable, freshly resolved installation without touching uv.lock."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lock = ROOT / "uv.lock"
    before = hashlib.sha256(lock.read_bytes()).digest()
    scratch = ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    lowest = "--lowest" in sys.argv
    with tempfile.TemporaryDirectory(prefix="compat-", dir=scratch) as tmp:
        env = Path(tmp) / "env"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(env)], check=True)
        python = env / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        command = [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--upgrade",
            "-e",
            str(ROOT),
            "pytest>=9.1.1",
            "pytest-cov>=7.1.0",
        ]
        if lowest:
            command.extend([
                "--constraint",
                str(ROOT / ".github" / "constraints" / "minimum.txt"),
            ])
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
        subprocess.run([str(python), "-m", "pytest"], cwd=ROOT, check=True)
        # Representative docs use public APIs and a plotting bridge.
        if not lowest:
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "great-docs>=0.15.0",
                    "plotnine",
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(python),
                    "-c",
                    "from great_docs import __version__; print(__version__)",
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [
                    str(python),
                    "-c",
                    "import plotnine; from tidydraws import point_interval; print(plotnine.__version__)",
                ],
                check=True,
            )
    assert hashlib.sha256(lock.read_bytes()).digest() == before, (
        "Compatibility check modified uv.lock"
    )


if __name__ == "__main__":
    main()
