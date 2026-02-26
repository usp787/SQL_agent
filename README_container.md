# Agent container + host Ollama: concrete setup

This setup **packages only the Python agent** into a Docker image.
**Ollama runs on the host** (outside Docker). Model weights stay on the host.

## 0) Preconditions

- Docker installed
- Ollama installed on the host and running:
  - Start: `ollama serve`
  - Verify: `curl http://127.0.0.1:11434/api/tags`

Pull your model once on the host (example):
- `ollama pull qwen2.5-coder:7b`

## 1) Files in this folder

- `Dockerfile`          : builds the agent image
- `requirements.txt`    : Python deps
- `run_query.py`        : simple CLI runner
- `.dockerignore`       : keep builds clean
- `sql_agent_v3.py`     : your agent module (copied from your project)

## 2) Build the agent image

From this folder:

```bash
docker build -t sql-agent:dev -f Dockerfile .
```

## 3) Make a data folder on host

You need:
- a SQLite DB file (e.g., Chinook)
- a place for Chroma persistence

Example:

```bash
mkdir -p ./data/chroma
# put your db at ./data/chinook.db
```

## 4) Run the container (recommended commands)

### A) Linux (best): use host networking

This makes `127.0.0.1:11434` inside the container refer to the host Ollama.

```bash
docker run --rm -it \
  --network=host \
  -v "$(pwd)/data:/data" \
  -e SQL_AGENT_DB_PATH=/data/chinook.db \
  -e SQL_AGENT_CHROMA_DIR=/data/chroma \
  -e SQL_AGENT_MODEL=qwen2.5-coder:7b \
  -e OLLAMA_HOST=http://127.0.0.1:11434 \
  sql-agent:dev \
  python run_query.py --question "Top 5 customers by total spend"
```

### B) Mac/Windows: use host.docker.internal

```bash
docker run --rm -it \
  -v "$(pwd)/data:/data" \
  -e SQL_AGENT_DB_PATH=/data/chinook.db \
  -e SQL_AGENT_CHROMA_DIR=/data/chroma \
  -e SQL_AGENT_MODEL=qwen2.5-coder:7b \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  sql-agent:dev \
  python run_query.py --question "Top 5 customers by total spend"
```

If `host.docker.internal` doesn't resolve (rare), you can:
- Use your host IP (e.g., `http://192.168.x.y:11434`)
- Or add `--add-host=host.docker.internal:host-gateway` (Docker 20.10+)

## 5) Troubleshooting quick checks

### Check container can reach Ollama
Inside container:

```bash
python -c "import os,requests; print(os.environ.get('OLLAMA_HOST')); import urllib.request as u; print(u.urlopen(os.environ['OLLAMA_HOST']+'/api/tags').read()[:200])"
```

### Common errors

- `Connection refused`:
  - Ollama not running (`ollama serve`)
  - Wrong OLLAMA_HOST value
  - Docker networking mismatch (use `--network=host` on Linux)

- `model not found`:
  - Pull on host: `ollama pull <model>`
  - Ensure `SQL_AGENT_MODEL` matches Ollama tag exactly

## 6) Why this pattern is best

- Agent image stays small and fast to rebuild
- Ollama and model weights remain persistent on the host
- You can change models without rebuilding the agent image
