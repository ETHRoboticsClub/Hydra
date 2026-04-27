#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Embedder server entrypoint
# =============================================================================
# Starts a long-running FastAPI server that serves vision embeddings from a
# pretrained vision encoder (DINOv2 / SigLIP / CLIP — pick your poison in
# server_embedder.py). The trainer calls /embed for each batch.
#
# Environment variables:
#   EMBEDDER_MODEL  HuggingFace model id (default: facebook/dinov2-base)
#   PORT            HTTP port (default: 8001 — matches deploy.yaml's containerPort)
# =============================================================================

EMBEDDER_MODEL="${EMBEDDER_MODEL:-facebook/dinov2-base}"
PORT="${PORT:-8001}"

export VIRTUAL_ENV="/data/.venv-act-embedder"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "[embedder] ============================================"
echo "[embedder] Embedder server"
echo "[embedder] Model: ${EMBEDDER_MODEL}"
echo "[embedder] Port:  ${PORT}"
echo "[embedder] ============================================"

nvidia-smi 2>/dev/null || echo "[embedder] No GPU detected (CPU-only)"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VIRTUAL_ENV" ]; then
  uv venv "$VIRTUAL_ENV"
fi
# shellcheck source=/dev/null
. "${VIRTUAL_ENV}/bin/activate"

if [ -f "/workspace/pyproject.toml" ]; then
  uv sync --active --no-install-project --no-dev
fi

echo "[embedder] Starting uvicorn on 0.0.0.0:${PORT}"
exec uv run --active --no-sync uvicorn server_embedder:app \
  --host 0.0.0.0 \
  --port "${PORT}"
