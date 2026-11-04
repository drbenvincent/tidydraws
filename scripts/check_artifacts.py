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

"""Build once, validate metadata, and test isolated consumer installations."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "artifact_smoke.py"


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def installed_smoke(artifact: Path, expected: str, extra: str = "") -> None:
    # A sibling scratch directory keeps execution entirely outside the checkout.
    scratch = ROOT.parent / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tidydraws-artifact-", dir=scratch) as tmp:
        location = Path(tmp)
        env = location / "env"
        run("uv", "venv", "--python", sys.executable, str(env))
        python = env / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        requirement = str(artifact) + (f"[{extra}]" if extra else "")
        run("uv", "pip", "install", "--python", str(python), requirement)
        run("uv", "pip", "check", "--python", str(python))
        run(str(python), "-I", str(SMOKE), expected, extra, cwd=location)


def main() -> None:
    target = ROOT / ".scratch" / "dist"
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=target.parent) as tmp:
        output = Path(tmp)
        run("uv", "build", "--out-dir", str(output))
        artifacts = sorted([*output.glob("*.whl"), *output.glob("*.tar.gz")])
        assert len(artifacts) == 2, artifacts
        run("uv", "run", "--locked", "twine", "check", "--strict", *map(str, artifacts))
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        expected = config["tool"]["bumpver"]["current_version"]
        for artifact in artifacts:
            installed_smoke(artifact, expected)
            if artifact.suffix == ".whl":
                for extra in ("letsplot", "plotnine", "altair"):
                    installed_smoke(artifact, expected, extra)
        target.mkdir(exist_ok=True)
        for old in target.glob("tidydraws-*"):
            old.unlink()
        for artifact in artifacts:
            shutil.copy2(artifact, target)
    print(f"Validated distributions: {target}")


if __name__ == "__main__":
    main()
