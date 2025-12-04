#!/usr/bin/env bash
# Script: gradle-build.sh
# Purpose: Run Gradle builds for coordinator and gateway subprojects using the correct gradle wrapper.
# This script ensures we run the wrapper from the project directory, avoiding "Directory does not contain a Gradle build" errors.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# CY_LLM_Backend paths (standard layout)
COORDINATOR_DIR="$ROOT_DIR/CY_LLM_Backend/coordinator"
GATEWAY_DIR="$ROOT_DIR/CY_LLM_Backend/gateway"

if [ -z "${JAVA_HOME:-}" ]; then
  if command -v java >/dev/null 2>&1; then
    JAVA_BIN_PATH=$(command -v java)
    JAVA_HOME=$(dirname $(dirname "$JAVA_BIN_PATH"))
    echo "Detected JAVA_HOME: $JAVA_HOME"
  else
    echo "WARNING: JAVA_HOME not set and no java binary detected in PATH. Please run scripts/setup-jdk.sh";
  fi
fi
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
