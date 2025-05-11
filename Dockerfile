FROM nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen

# Install rsync
RUN apt-get update && apt-get install -y rsync && rm -rf /var/lib/apt/lists/*

# Setup TensorRT-LLM backend in temporary directory
WORKDIR /tmp
RUN git lfs install && \
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git && \
    cd tensorrtllm_backend && \
    git submodule update --init --recursive

WORKDIR /app
COPY src/job /app/src/job
WORKDIR /app/src/job

LABEL org.opencontainers.image.source https://github.com/elamribadrayour/tensorrt-llm-model-builder

ENTRYPOINT ["uv", "run", "main.py"]
