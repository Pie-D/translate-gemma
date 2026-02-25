#!/usr/bin/env bash
# Chạy vLLM server với model google/translategemma-4b-it
# Cần chạy script này trước, sau đó chạy app.py (hoặc uvicorn app:app --port 8001)

set -e
MODEL_ID="${MODEL_ID:-google/translategemma-4b-it}"
PORT="${VLLM_PORT:-8000}"

echo "Starting vLLM server: $MODEL_ID on port $PORT"
exec vllm serve "$MODEL_ID" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
