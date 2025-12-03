#!/usr/bin/env bash
# Lightweight wrapper so users can run `./cy` in place of `./cy-llm`.
# If you're running this script from the repo root it will prefer to
# execute the canonical ./cy-llm. It falls back to `ew` if `cy-llm` is
# not found (legacy compatibility).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/cy-llm" ]]; then
  exec "$SCRIPT_DIR/cy-llm" "$@"
elif [[ -f "$SCRIPT_DIR/ew" ]]; then
  # `ew` exists as a legacy alias; forward the command and exit
  exec "$SCRIPT_DIR/ew" "$@"
else
  echo "cy wrapper requires 'cy-llm' or legacy 'ew' script in the repo root; Please ensure you run from repo root." >&2
  exit 1
fi
