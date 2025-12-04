#!/usr/bin/env bash
# Canonical alias for CY-LLM CLI; delegates to cy-llm.
# If you're running this script from the repo root it will prefer to
# execute ./cy-llm, or fallback to the implementation script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/cy-llm" ]]; then
  exec "$SCRIPT_DIR/cy-llm" "$@"
else
  echo "cy script requires cy-llm wrapper or implementation script in the repo root." >&2
  exit 1
fi
