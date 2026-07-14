#!/bin/bash
# bumpver pre_commit_hook: re-derive uv.lock from the new __version__ and
# stage it so the version bump, lockfile, and tag all land in one commit.
# See https://brtkwr.com/posts/2026-01-14-bumpver-with-uv/ for the rationale.
set -euo pipefail
uv lock
git add uv.lock
