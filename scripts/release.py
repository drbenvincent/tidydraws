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

"""Release guards shared by make release-* and the tag-triggered validation."""

from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_version() -> str:
    source = (ROOT / "tidydraws" / "__init__.py").read_text()
    match = re.search(r'^__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"$', source, re.M)
    if match is None:
        raise RuntimeError("Expected a numeric major.minor.patch package version")
    version = match[1]
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if config["tool"]["bumpver"]["current_version"] != version:
        raise RuntimeError("bumpver and package versions disagree")
    return version


def preflight() -> None:
    if git("branch", "--show-current") != "main":
        raise RuntimeError(
            "Releases must run from main; switch to an up-to-date main checkout"
        )
    if git("status", "--porcelain"):
        raise RuntimeError(
            "Release requires a clean working tree, including untracked files"
        )
    subprocess.run(["git", "fetch", "origin", "main", "--tags"], cwd=ROOT, check=True)
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise RuntimeError("Local main must match origin/main before releasing")
    package_version()
    subprocess.run(["uv", "lock", "--check"], cwd=ROOT, check=True)


def tag_check(tag: str) -> None:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise RuntimeError("Release tag must be major.minor.patch")
    if tag != package_version():
        raise RuntimeError("Release tag does not match package and bumpver versions")
    head = git("rev-parse", "HEAD")
    if git("rev-parse", f"refs/tags/{tag}^{{commit}}") != head:
        raise RuntimeError("Checkout is not the release tag's commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, "origin/main"],
        cwd=ROOT,
        check=True,
    )


def post_bump() -> None:
    version = package_version()
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{version}"],
        cwd=ROOT,
        capture_output=True,
    )
    if exists.returncode == 0:
        raise RuntimeError(
            f"Tag {version} already exists; published tags cannot be reused"
        )
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    subprocess.run(["make", "precommit", "test", "build", "docs"], cwd=ROOT, check=True)
    # Reject hook rewrites beyond the two version fields and refreshed lock.
    allowed = {"tidydraws/__init__.py", "pyproject.toml", "uv.lock"}
    changed = set(git("diff", "--name-only", "HEAD").splitlines())
    if not changed <= allowed or git("ls-files", "--others", "--exclude-standard"):
        raise RuntimeError(
            "Validation modified unrelated files; review them before retrying release"
        )
    subprocess.run(["git", "add", "uv.lock"], cwd=ROOT, check=True)


def main() -> None:
    action = sys.argv[1]
    if action == "post-bump":
        post_bump()
    elif action == "tag":
        tag_check(sys.argv[2])
    elif action in {"patch", "minor", "major"}:
        preflight()
        subprocess.run(
            ["uv", "run", "--locked", "bumpver", "update", f"--{action}"],
            cwd=ROOT,
            check=True,
        )
    else:
        raise RuntimeError(f"Unknown release action: {action}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"Release stopped: {error}")
