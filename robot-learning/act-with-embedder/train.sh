#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Training entrypoint that uses the sidecar embedder
# =============================================================================
# By the time this runs, the wait-for-embedder initContainer has already
# verified the embedder is healthy at $EMBEDDER_URL. This script:
#   1. Sanity-checks the embedder is reachable (cheap belt-and-braces)
#   2. Sets up the training venv on the persistent /data volume
#   3. Hands off to lerobot-train (or your preferred trainer)
#
# Environment variables:
#   EMBEDDER_URL    Set by deploy.yaml — points at the embedder Service
#   DATASET_REPO_ID HuggingFace dataset (default: ETHRC/act-towelspring26_3)
#   CHECKPOINT_DIR  Where checkpoints land (default: /checkpoints/act)
# =============================================================================

EMBEDDER_URL="${EMBEDDER_URL:?EMBEDDER_URL not set — deploy.yaml should inject it}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ETHRC/act-towelspring26_3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoints/act}"

export VIRTUAL_ENV="/data/.venv-act-with-embedder"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "[trainer] ============================================"
echo "[trainer] Training with sidecar embedder"
echo "[trainer] Embedder: ${EMBEDDER_URL}"
echo "[trainer] Dataset:  ${DATASET_REPO_ID}"
echo "[trainer] Output:   ${CHECKPOINT_DIR}"
echo "[trainer] ============================================"

# Sanity-check the embedder. The initContainer already polled /healthz, but
# we also verify /embed works end-to-end with a tiny dummy payload before
# burning compute on the training loop.
echo "[trainer] Sanity-checking embedder /embed endpoint..."
curl -fsS -X POST -H 'Content-Type: application/json' \
  -d '{"images":[[[[0.0,0.0,0.0]]]]}' \
  "${EMBEDDER_URL}" > /dev/null
echo "[trainer] Embedder responded OK."

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VIRTUAL_ENV" ]; then
  uv venv "$VIRTUAL_ENV"
fi
# shellcheck source=/dev/null
. "${VIRTUAL_ENV}/bin/activate"

if [ -f "/workspace/pyproject.toml" ]; then
  uv sync --active --no-install-project --no-dev
fi

mkdir -p "${CHECKPOINT_DIR}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Minimal training stub. Replace with your real trainer — the only thing the
# trainer needs from this scaffold is that EMBEDDER_URL is reachable.
# When you're ready, drop in the same `lerobot-train ...` invocation as
# robot-learning/act-new/entrypoint.sh.
echo "[trainer] Starting training (stub: 30s sleep — replace with lerobot-train)"
sleep 30
echo "[trainer] Training step complete."
