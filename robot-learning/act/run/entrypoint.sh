#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_REPO_ID="ETHRC/towelspring26"
DATASET_ROOT="/data"
CHECKPOINT_DIR="/checkpoints/act"
DATA_DIR="${DATASET_ROOT}/${DATASET_REPO_ID}"

# Cache uv packages and venv on persistent storage
export UV_CACHE_DIR="/data/.uv-cache"
export VIRTUAL_ENV="/data/.venv"

nvidia-smi

# ── 1. Sync dependencies ──────────────────────────────────────────────────────
echo "[run.sh] Syncing uv environment..."
cd "${SCRIPT_DIR}"
if [ ! -d "$VIRTUAL_ENV" ]; then
  uv venv "$VIRTUAL_ENV"
fi
uv pip sync pyproject.toml

# ── 2. Checkpoint guard ───────────────────────────────────────────────────────
# If a checkpoint already exists, there is nothing to do — exit cleanly so the
# job is not accidentally re-run (and does not burn GPU time).
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
  echo "[run.sh] Checkpoints found at ${CHECKPOINT_DIR} — training already complete. Exiting."
  exit 0
fi

# ── 3. Dataset check / download ───────────────────────────────────────────────
if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
  echo "[run.sh] Dataset not found at ${DATA_DIR}. Downloading from Hugging Face..."
  mkdir -p "${DATA_DIR}"
  uv run huggingface-cli download "${DATASET_REPO_ID}" \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"
  echo "[run.sh] Dataset download complete."
else
  echo "[run.sh] Dataset found at ${DATA_DIR}. Skipping download."
fi

# ── 4. Train ──────────────────────────────────────────────────────────────────
echo "[run.sh] Starting lerobot-train..."

export WANDB_MODE=disabled

uv run --no-sync lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --output_dir="${CHECKPOINT_DIR}" \
  --job_name=act_training \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
