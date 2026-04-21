#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ACT Training Entrypoint
# =============================================================================
# This script runs ACT (Action Chunking with Transformers) training using
# lerobot. It handles dataset download, environment setup, and training.
#
# Environment variables (set by launch-new or manually):
#   DATASET_REPO_ID     - HuggingFace dataset repo (e.g., "ETHRC/act-towelspring26_3")
#   DATASET_REVISION    - Dataset revision/tag to use
#   CHECKPOINT_DIR      - Where to save checkpoints (default: /checkpoints/act)
#   POLICY_REPO_ID      - HuggingFace repo for pushing trained policy
#   TRAIN_BATCH_SIZE    - Training batch size (default: 29)
#   NUM_WORKERS         - DataLoader workers (default: 0)
#   VIDEO_BACKEND       - Video backend: torchcodec or pyav (default: torchcodec)
# =============================================================================

# User-facing configuration (override via environment)
DATASET_REPO_ID="${DATASET_REPO_ID:-ETHRC/act-towelspring26_3}"
DATASET_REVISION="${DATASET_REVISION:-main}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoints/act}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ETHRC/act-towelspring26}"

# Training configuration
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-29}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"

# Cache uv packages and venv on persistent storage
export VIRTUAL_ENV="/data/.venv-act"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "[entrypoint] ============================================"
echo "[entrypoint] ACT Training Job"
echo "[entrypoint] ============================================"
echo "[entrypoint] Dataset: ${DATASET_REPO_ID}@${DATASET_REVISION}"
echo "[entrypoint] Checkpoints: ${CHECKPOINT_DIR}"
echo "[entrypoint] Policy output: ${POLICY_REPO_ID}"
echo "[entrypoint] Batch size: ${TRAIN_BATCH_SIZE}"
echo "[entrypoint] Video backend: ${VIDEO_BACKEND}"
echo "[entrypoint] ============================================"

# Show GPU info
nvidia-smi 2>/dev/null || echo "[entrypoint] No GPU detected (running CPU-only)"

# ── 1. Sync dependencies ──────────────────────────────────────────────────────
echo "[entrypoint] Setting up Python environment..."
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VIRTUAL_ENV" ]; then
  echo "[entrypoint] Creating virtual environment at ${VIRTUAL_ENV}..."
  uv venv "$VIRTUAL_ENV"
fi

# shellcheck source=/dev/null
. "${VIRTUAL_ENV}/bin/activate"

# Sync dependencies (pyproject.toml should be in /workspace from initContainer)
if [ -f "/workspace/pyproject.toml" ]; then
  echo "[entrypoint] Syncing dependencies..."
  uv sync --active --no-install-project --no-dev
else
  echo "[entrypoint] Warning: pyproject.toml not found, installing default dependencies..."
  uv pip install lerobot huggingface-hub
fi

# ── 2. Checkpoint guard ───────────────────────────────────────────────────────
# If checkpoint artifacts already exist (excluding bootstrap.log), skip training
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 ! -name 'bootstrap.log' -print -quit 2>/dev/null)" ]; then
  echo "[entrypoint] Checkpoints found at ${CHECKPOINT_DIR} - training already complete. Exiting."
  exit 0
fi

# ── 3. Dataset check / download ───────────────────────────────────────────────
DATASET_ROOT="/data"
DATA_DIR="${DATASET_ROOT}/${DATASET_REPO_ID}"
DATASET_REVISION_FILE="${DATA_DIR}/.dataset_revision"
CURRENT_DATASET_REVISION=""

if [ -f "${DATASET_REVISION_FILE}" ]; then
  CURRENT_DATASET_REVISION="$(<"${DATASET_REVISION_FILE}")"
fi

if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ] || [ "${CURRENT_DATASET_REVISION}" != "${DATASET_REVISION}" ]; then
  if [ "${CURRENT_DATASET_REVISION}" != "" ] && [ "${CURRENT_DATASET_REVISION}" != "${DATASET_REVISION}" ]; then
    echo "[entrypoint] Dataset revision mismatch: found ${CURRENT_DATASET_REVISION}, need ${DATASET_REVISION}. Refreshing..."
  else
    echo "[entrypoint] Dataset not found. Downloading from Hugging Face..."
  fi

  rm -rf "${DATA_DIR}"
  mkdir -p "${DATA_DIR}"

  uv run --active --no-sync hf download "${DATASET_REPO_ID}" \
    --local-dir "${DATA_DIR}"

  printf '%s\n' "${DATASET_REVISION}" > "${DATASET_REVISION_FILE}"
  echo "[entrypoint] Dataset download complete."
else
  echo "[entrypoint] Dataset revision ${DATASET_REVISION} already present. Skipping download."
fi

# ── 4. Train ──────────────────────────────────────────────────────────────────
echo "[entrypoint] Starting lerobot-train..."
echo "[entrypoint] ============================================"

# Ensure checkpoint directory exists with proper permissions
mkdir -p "${CHECKPOINT_DIR}"

# Set W&B mode (online by default, can be disabled via env)
export WANDB_MODE="${WANDB_MODE:-online}"

uv run --active --no-sync lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.revision="${DATASET_REVISION}" \
  --policy.type=act \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --output_dir="${CHECKPOINT_DIR}" \
  --batch_size="${TRAIN_BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --save_freq=900 \
  --log_freq=20 \
  --policy.push_to_hub=true \
  --job_name=act_training \
  --wandb.project=act \
  --wandb.enable=true \
  --policy.device=cuda

echo "[entrypoint] ============================================"
echo "[entrypoint] Training complete!"
echo "[entrypoint] Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "[entrypoint] Policy pushed to: ${POLICY_REPO_ID}"
echo "[entrypoint] ============================================"
