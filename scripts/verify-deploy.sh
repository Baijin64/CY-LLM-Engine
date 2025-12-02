#!/usr/bin/env bash
# Quick smoke test to deploy cy-llm via docker compose and check health endpoints
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/CY_LLM_Backend/deploy"

echo "Starting CY-LLM via Docker Compose..."
cd "$DEPLOY_DIR"
docker compose up -d --build

echo "Waiting for services..."
sleep 10

echo "Gateway health:"
curl -sSI http://localhost:8080/actuator/health | head -n 1

echo "Coordinator health:"
curl -sSI http://localhost:8081/actuator/health | head -n 1 || true

echo "Worker health:"
curl -sSI http://localhost:9090/metrics | head -n 1 || true

echo "Done. To teardown run: docker compose down"
