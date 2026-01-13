FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    clang \
    libgl1 \
    cuda-cudart-dev-12-1 \
    cuda-nvrtc-dev-12-1 \
  && rm -rf /var/lib/apt/lists/*
