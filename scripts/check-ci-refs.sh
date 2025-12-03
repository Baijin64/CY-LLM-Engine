#!/usr/bin/env bash
# Search for common CI config files referencing EW or ew image names
set -euo pipefail

echo "Searching common CI configs for EW or ew references..."

CI_FILES=(".github/workflows" ".circleci" ".gitlab-ci.yml" "Jenkinsfile" "ci" "docker-compose.yml")
for f in "${CI_FILES[@]}"; do
  if [ -e "$f" ]; then
    echo "-- Scanning: $f"
    git grep -n --full-name -I -- "ew-\w+|\bEW_\w+\b|EW_AI_Backend|EW_AI_Deployment" -- $f || true
  fi
done
echo "Done. If any matches found, please update CI/release pipelines to use cy-llm image names and CY_LLM_* env variables."
