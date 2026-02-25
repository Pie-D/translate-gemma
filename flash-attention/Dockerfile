# Dùng cuda-devel để compile flash-attn (cần nvcc)
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV TORCH_HOME=/models
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài PyTorch & dependencies trước
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt
RUN pip3 install bitsandbytes accelerate

# Cài flash-attn (cần --no-build-isolation để compile với CUDA)
RUN pip3 install flash-attn --no-build-isolation

COPY main.py /app/main.py

RUN mkdir -p /models
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
