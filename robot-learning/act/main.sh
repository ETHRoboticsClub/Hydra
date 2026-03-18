#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_REPO_ID="ETHRC/towel_base_with_rewards"
DATASET_ROOT="/data"
CHECKPOINT_DIR="/checkpoints/act"
DATA_DIR="${DATASET_ROOT}/${DATASET_REPO_ID}"

# Create directory with fallback to sudo if needed (handles kube PVC mount permissions)
ensure_dir() {
  local dir="$1"
  if [ -d "$dir" ]; then return 0; fi
  if mkdir -p "$dir" 2>/dev/null; then return 0; fi
  if command -v sudo >/dev/null 2>&1; then
    sudo mkdir -p "$dir"
    sudo chown "$(id -u):$(id -g)" "$dir"
  else
    echo "[run.sh] ERROR: Cannot create $dir (no sudo available)"
    exit 1
  fi
}

nvidia-smi

# ── 1. Sync dependencies ──────────────────────────────────────────────────────
echo "[run.sh] Syncing uv environment..."
cd "${SCRIPT_DIR}"
uv sync
uv pip install huggingface-hub[cli]

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
  ensure_dir "${DATA_DIR}"
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
