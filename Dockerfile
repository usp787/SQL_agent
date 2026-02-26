# SQL Agent (Python) container
# This container assumes Ollama is running on the HOST (outside the container).
# It only packages the agent + deps (LangGraph, ChromaDB client, Ollama python client).

FROM python:3.11-slim

# Avoid prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - ca-certificates/curl: sanity + debugging
# - build-essential: occasional wheels may need compilation (safe default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy agent code + tiny CLI wrapper
COPY sql_agent_v3.py /app/sql_agent_v3.py
COPY run_query.py /app/run_query.py

# Default envs (override at runtime)
# Important: set OLLAMA_HOST to reach the host Ollama.
ENV SQL_AGENT_DB_PATH=/data/chinook.db \
    SQL_AGENT_CHROMA_DIR=/data/chroma \
    SQL_AGENT_MODEL=qwen2.5-coder:7b \
    OLLAMA_HOST=http://host.docker.internal:11434

# Create a writable data dir (use volume mount in practice)
RUN mkdir -p /data/chroma

# Default: show help
CMD ["python", "run_query.py", "--help"]
