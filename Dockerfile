FROM nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen

COPY src/job /app/src/job
WORKDIR /app/src/job

LABEL org.opencontainers.image.source https://github.com/elamribadrayour/tensorrt-llm-model-builder

ENTRYPOINT ["uv", "run", "main.py"]
