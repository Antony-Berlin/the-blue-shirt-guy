FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY envs/codeforge_env/ ./
COPY tasks/ /app/tasks/

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.0" \
    "fastmcp>=3.0.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.23.0" \
    "pydantic>=2.0.0" \
    "anthropic>=0.25.0" \
    "rank-bm25>=0.2.2" \
    "pyflakes>=3.0.0"

ENV PYTHONPATH=/app
ENV BENCHMARK_PATH=/app/tasks/benchmark.json

EXPOSE 7860

CMD ["python", "-m", "server.app", "--host", "0.0.0.0", "--port", "7860"]
