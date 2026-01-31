#!/usr/bin/env bash
# Quick smoke test for CY-LLM Lite deployment
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting CY-LLM Lite via Docker Compose..."
cd "$ROOT_DIR"
docker compose -f docker-compose.community.yml up -d

echo "Waiting for services to start..."
sleep 15

echo "Testing Gateway Lite health endpoint..."
if curl -sSf http://localhost:8000/health > /dev/null; then
    echo "✓ Gateway Lite is healthy"
else
    echo "✗ Gateway Lite health check failed"
    exit 1
fi

echo "Testing chat completions endpoint..."
response=$(curl -sSf http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}')

if echo "$response" | grep -q '"choices"'; then
    echo "✓ Chat completions endpoint is working"
else
    echo "✗ Chat completions endpoint failed"
    echo "Response: $response"
    exit 1
fi

echo ""
echo "✓ All checks passed!"
echo ""
echo "To view logs: docker compose -f docker-compose.community.yml logs -f"
echo "To stop: docker compose -f docker-compose.community.yml down"
