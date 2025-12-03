#!/usr/bin/env bash
# Script: gradle-build.sh
# Purpose: Run Gradle builds for coordinator and gateway subprojects using the correct gradle wrapper.
# This script ensures we run the wrapper from the project directory, avoiding "Directory does not contain a Gradle build" errors.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COORDINATOR_DIR="$ROOT_DIR/CY_LLM_Backend/coordinator"
GATEWAY_DIR="$ROOT_DIR/CY_LLM_Backend/gateway"

export PATH="$JAVA_HOME/bin:$PATH" || true

echo "Running build for coordinator: $COORDINATOR_DIR"
if [ -d "$COORDINATOR_DIR" ]; then
  pushd "$COORDINATOR_DIR" > /dev/null
  if [ -x "./gradlew" ]; then
    ./gradlew clean build -x test || { echo "Coordinator build failed"; exit 1; }
  else
    echo "No gradlew found in $COORDINATOR_DIR. Skipping coordinator build.";
  fi
  popd > /dev/null
else
  echo "Coordinator dir not found: $COORDINATOR_DIR, skipping";
fi

echo "\nRunning build for gateway: $GATEWAY_DIR"
if [ -d "$GATEWAY_DIR" ]; then
  pushd "$GATEWAY_DIR" > /dev/null
  if [ -x "./gradlew" ]; then
    ./gradlew clean build -x test || { echo "Gateway build failed"; exit 1; }
  else
    echo "No gradlew found in $GATEWAY_DIR. Skipping gateway build.";
  fi
  popd > /dev/null
else
  echo "Gateway dir not found: $GATEWAY_DIR, skipping";
fi

echo "\nGradle builds completed successfully (skipped tests)."
