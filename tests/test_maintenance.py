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

"""Exercise release failure boundaries without creating commits or publishing."""

from importlib import import_module
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


@pytest.fixture
def release(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("release")


def test_release_rejects_feature_branch_before_any_mutation(release, monkeypatch):
    monkeypatch.setattr(release, "git", lambda *args: "feature/unfinished")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="from main"):
        release.preflight()
    assert calls == []


def test_release_rejects_dirty_tree_before_fetch(release, monkeypatch):
    def git(*args):
        return "main" if args[0] == "branch" else "?? unfinished.py"

    monkeypatch.setattr(release, "git", git)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="clean working tree"):
        release.preflight()
    assert calls == []


def test_release_rejects_remote_divergence(release, monkeypatch):
    def git(*args):
        if args[0] == "branch":
            return "main"
        if args[0] == "status":
            return ""
        return args[-1]

    monkeypatch.setattr(release, "git", git)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="match origin/main"):
        release.preflight()


@pytest.mark.parametrize("tag", ["v1.2.3", "1.2.4", "1.2.3; echo unsafe"])
def test_release_rejects_invalid_or_mismatched_tag(release, monkeypatch, tag):
    monkeypatch.setattr(release, "package_version", lambda: "1.2.3")
    with pytest.raises(RuntimeError, match="tag"):
        release.tag_check(tag)


def test_release_rejects_commit_outside_main(release, monkeypatch):
    monkeypatch.setattr(release, "package_version", lambda: "1.2.3")
    monkeypatch.setattr(release, "git", lambda *a: "same-commit")

    def not_ancestor(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", not_ancestor)
    with pytest.raises(subprocess.CalledProcessError):
        release.tag_check("1.2.3")


def test_validation_failure_stops_bump_without_staging(release, monkeypatch):
    monkeypatch.setattr(release, "package_version", lambda: "1.2.3")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "make":
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 1 if command[0] == "git" else 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        release.post_bump()
    assert calls[-1] == ["make", "precommit", "test", "build", "docs"]
    assert not any("add" in call for call in calls)


def test_release_one_command_harness(release, monkeypatch):
    """A successful invocation retains bumpver's commit/tag/push orchestration."""
    calls = []
    monkeypatch.setattr(release, "preflight", lambda: calls.append("preflight"))
    monkeypatch.setattr(sys, "argv", ["release.py", "patch"])
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append(command))
    release.main()
    assert calls == [
        "preflight",
        ["uv", "run", "--locked", "bumpver", "update", "--patch"],
    ]


@pytest.fixture
def index_verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("verify_testpypi")


def make_wheel(directory):
    wheel = directory / "tidydraws-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "tidydraws-1.2.3.dist-info/METADATA", "Name: tidydraws\nVersion: 1.2.3\n"
        )
    return wheel


def test_testpypi_conflicting_version_blocks_install(
    index_verifier, monkeypatch, tmp_path
):
    wheel = make_wheel(tmp_path)
    entry = {
        "filename": wheel.name,
        "digests": {"sha256": "wrong"},
        "url": "https://test-files.pythonhosted.org/exact.whl",
    }
    monkeypatch.setattr(
        index_verifier,
        "urlopen",
        lambda *a, **k: io.BytesIO(json.dumps({"urls": [entry]}).encode()),
    )
    calls = []
    monkeypatch.setattr(index_verifier, "installed_smoke", lambda *a: calls.append(a))
    with pytest.raises(AssertionError, match="differs"):
        index_verifier.verify(tmp_path, delay=0)
    assert calls == []


def test_testpypi_propagation_retry_tests_downloaded_bytes(
    index_verifier, monkeypatch, tmp_path
):
    wheel = make_wheel(tmp_path)
    import hashlib
    from urllib.error import HTTPError

    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    entry = {
        "filename": wheel.name,
        "digests": {"sha256": digest},
        "url": "https://test-files.pythonhosted.org/exact.whl",
    }
    urls = []

    def fetch(url, **kwargs):
        urls.append(url)
        if len(urls) == 1:
            raise HTTPError(url, 404, "not propagated", {}, None)
        return io.BytesIO(
            wheel.read_bytes()
            if url.endswith(".whl")
            else json.dumps({"urls": [entry]}).encode()
        )

    monkeypatch.setattr(index_verifier, "urlopen", fetch)
    calls = []
    monkeypatch.setattr(index_verifier, "installed_smoke", lambda *a: calls.append(a))
    index_verifier.verify(tmp_path, delay=0)
    assert urls[:2] == ["https://test.pypi.org/pypi/tidydraws/1.2.3/json"] * 2
    installed, expected = calls[0]
    assert (
        installed.parent.name == "testpypi"
        and installed.read_bytes() == wheel.read_bytes()
    )
    assert expected == "1.2.3"


@pytest.mark.parametrize("exit_code", [0, 17])
def test_release_hook_combines_streams_and_preserves_failure(
    monkeypatch, tmp_path, exit_code
):
    import os
    import shutil

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Release hook requires bash")
    stub = tmp_path / "uv"
    stub.write_text(
        "#!/bin/sh\nprintf 'validation output\\n'\n"
        f"printf 'validation error\\n' >&2\nexit {exit_code}\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    hook = Path(__file__).resolve().parents[1] / "scripts" / "pre-bump.sh"
    result = subprocess.run(
        [bash, str(hook)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == exit_code
    assert "validation output" in result.stdout
    assert "validation error" in result.stdout
    assert result.stderr == ""
