#!/usr/bin/env bash
# bumpver calls this after editing the version fields, before committing/tagging.
set -euo pipefail
# bumpver drains stdout before stderr; keep validation on a single pipe.
exec 2>&1
exec uv run --no-sync python scripts/release.py post-bump
