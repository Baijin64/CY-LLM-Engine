#!/usr/bin/env bash
# List all occurrences of 'ew' or 'EW_' that might need manual inspection
set -euo pipefail

echo "Searching for EW/ew references in repo (excluding common binary or cache dirs)..."
# Skip binary, build, and test artifacts. Use git grep to keep it fast.
git grep -n --full-name -I -- "\bEW_\w+\b\|\bew-gateway\b\|\bew-\w+\b\|\bew_ai_\w+\b" || true

echo "Done."
