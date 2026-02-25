# Image chạy vLLM engine (model server)
FROM vllm/vllm-openai:latest

ENV MODEL_ID=google/translategemma-4b-it
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

# Có thể mount volume ./models:/models khi chạy
RUN mkdir -p /models

EXPOSE 8000

# vllm-openai image entrypoint thường là vllm serve; truyền thêm args
CMD ["--model", "google/translategemma-4b-it", "--port", "8000", "--trust-remote-code", "--max-model-len", "2048", "--gpu-memory-utilization", "0.9"]
