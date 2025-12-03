#!/usr/bin/env bash
# Script: gradle-build.sh
# Purpose: Run Gradle builds for coordinator and gateway subprojects using the correct gradle wrapper.
# This script ensures we run the wrapper from the project directory, avoiding "Directory does not contain a Gradle build" errors.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer CY_LLM_Backend paths (refactor branch), fallback to EW_AI_Backend for older layout
COORDINATOR_DIR="$ROOT_DIR/CY_LLM_Backend/coordinator"
if [ ! -d "$COORDINATOR_DIR" ]; then
  COORDINATOR_DIR="$ROOT_DIR/EW_AI_Backend/coordinator"
fi
GATEWAY_DIR="$ROOT_DIR/CY_LLM_Backend/gateway"
if [ ! -d "$GATEWAY_DIR" ]; then
  GATEWAY_DIR="$ROOT_DIR/EW_AI_Backend/gateway"
fi

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
    # try legacy EW_AI_Backend path
    if [ -x "$ROOT_DIR/EW_AI_Backend/coordinator/gradlew" ]; then
      echo "gradlew not found in $COORDINATOR_DIR, using legacy path EW_AI_Backend/coordinator/gradlew"
      (cd "$ROOT_DIR/EW_AI_Backend/coordinator" && ./gradlew clean build -x test) || { echo "Coordinator build failed via legacy path"; exit 1; }
    else
      echo "No gradlew found in $COORDINATOR_DIR nor in EW_AI_Backend/coordinator. Skipping coordinator build.";
    fi
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
    # try legacy EW_AI_Backend path
    if [ -x "$ROOT_DIR/EW_AI_Backend/gateway/gradlew" ]; then
      echo "gradlew not found in $GATEWAY_DIR, using legacy path EW_AI_Backend/gateway/gradlew"
      (cd "$ROOT_DIR/EW_AI_Backend/gateway" && ./gradlew clean build -x test) || { echo "Gateway build failed via legacy path"; exit 1; }
    else
      echo "No gradlew found in $GATEWAY_DIR nor in EW_AI_Backend/gateway. Skipping gateway build.";
    fi
  fi
  popd > /dev/null
else
  echo "Gateway dir not found: $GATEWAY_DIR, skipping";
fi

echo "\nGradle builds completed successfully (skipped tests)."
