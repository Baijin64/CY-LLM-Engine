#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CY_LLM_BIN="${CY_LLM_BIN:-${ROOT_DIR}/cy-llm}"
LITE_PORT="${CY_LLM_LITE_PORT:-8000}"
MODEL_ID="${CY_LLM_DEFAULT_MODEL:-facebook/opt-2.7b}"

LOG_DIR="${LITE_LOG_DIR:-/tmp}"
GATEWAY_LOG="${LOG_DIR}/cy-llm-lite-gateway.log"

if [[ ! -x "$CY_LLM_BIN" ]]; then
  echo "[e2e] cy-llm not found or not executable: $CY_LLM_BIN" >&2
  exit 1
fi

cleanup() {
  "$CY_LLM_BIN" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$CY_LLM_BIN" stop >/dev/null 2>&1 || true

"$CY_LLM_BIN" lite --engine "${CY_LLM_ENGINE:-cuda-vllm}" --model "$MODEL_ID" -d

# wait for health
for _ in {1..60}; do
  if curl -fsS "http://localhost:${LITE_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [[ -f "$GATEWAY_LOG" ]]; then
    if grep -q "ERROR" "$GATEWAY_LOG"; then
      echo "[e2e] gateway error detected" >&2
      tail -n 20 "$GATEWAY_LOG" >&2 || true
      exit 1
    fi
  fi
  if [[ $_ -eq 60 ]]; then
    echo "[e2e] timeout waiting for /health" >&2
    exit 1
  fi
done

payload=$(cat <<EOF
{"model":"default","messages":[{"role":"user","content":"你好"}]}
EOF
)

resp=$(curl -fsS "http://localhost:${LITE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$payload")

if ! echo "$resp" | grep -q "\"choices\""; then
  echo "[e2e] invalid response" >&2
  echo "$resp" >&2
  exit 1
fi

echo "[e2e] ok"