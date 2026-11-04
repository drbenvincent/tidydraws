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

"""Register and inspect a project-owned kernel; never overwrite a user kernel."""

import os
from pathlib import Path
import subprocess
import sys

from jupyter_client.kernelspec import KernelSpecManager
import tidydraws

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / ".scratch" / "jupyter"


def prepare() -> dict[str, str]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(PREFIX),
            "--name",
            "python3",
        ],
        check=True,
    )
    path = PREFIX / "share" / "jupyter"
    manager = KernelSpecManager(kernel_dirs=[str(path / "kernels")])
    spec = manager.get_kernel_spec("python3")
    assert Path(spec.argv[0]).absolute() == Path(sys.executable).absolute(), spec.argv
    print(f"tidydraws {tidydraws.__version__}; kernel argv: {spec.argv}")
    env = dict(os.environ)
    env["JUPYTER_PATH"] = str(path) + (
        os.pathsep + env["JUPYTER_PATH"] if env.get("JUPYTER_PATH") else ""
    )
    env["QUARTO_PYTHON"] = sys.executable
    return env


if __name__ == "__main__":
    env = prepare()
    if len(sys.argv) > 1:
        subprocess.run(sys.argv[1:], cwd=ROOT, env=env, check=True)
        if sys.argv[1:] == ["great-docs", "build"]:
            for name in ("llms.txt", "llms-full.txt", "skill.md"):
                target = ROOT / "great-docs" / "_site" / name
                assert target.is_file() and target.stat().st_size, target
