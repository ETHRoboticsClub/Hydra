#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ACT Inference Server Entrypoint
# =============================================================================
# Long-running FastAPI server that loads an ACT policy checkpoint and exposes
# /predict for action inference. Health check at /healthz.
#
# Environment variables:
#   POLICY_REPO_ID    HuggingFace repo to pull the policy from (default: ETHRC/act-towelspring26)
#   CHECKPOINT_DIR    Local checkpoint directory (default: /checkpoints/act)
#   PORT              HTTP port to listen on (default: 8000)
# =============================================================================

POLICY_REPO_ID="${POLICY_REPO_ID:-ETHRC/act-towelspring26}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoints/act}"
PORT="${PORT:-8000}"

# Re-use the same uv-managed venv as training so package fetches stay cached.
export VIRTUAL_ENV="/data/.venv-act-inference"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "[inference] ============================================"
echo "[inference] ACT Inference Server"
echo "[inference] Policy: ${POLICY_REPO_ID}"
echo "[inference] Checkpoints: ${CHECKPOINT_DIR}"
echo "[inference] Port: ${PORT}"
echo "[inference] ============================================"

nvidia-smi 2>/dev/null || echo "[inference] No GPU detected (CPU-only)"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VIRTUAL_ENV" ]; then
  echo "[inference] Creating virtual environment at ${VIRTUAL_ENV}..."
  uv venv "$VIRTUAL_ENV"
fi
# shellcheck source=/dev/null
. "${VIRTUAL_ENV}/bin/activate"

if [ -f "/workspace/pyproject.toml" ]; then
  echo "[inference] Syncing dependencies..."
  uv sync --active --no-install-project --no-dev
fi

echo "[inference] Starting uvicorn on 0.0.0.0:${PORT}"
exec uv run --active --no-sync uvicorn server:app \
  --host 0.0.0.0 \
  --port "${PORT}"
