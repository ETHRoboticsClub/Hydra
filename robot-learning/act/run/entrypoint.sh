#!/usr/bin/env bash
set -euo pipefail

# User-facing configuration
DATASET_REPO_ID="ETHRC/towelspring26-cleaned"
DATASET_REVISION="${DATASET_REVISION:-trimmed}"
CHECKPOINT_DIR="/checkpoints/act"

# Cache uv packages and venv on persistent storage
export UV_CACHE_DIR="/data/.uv-cache"
export VIRTUAL_ENV="/data/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

nvidia-smi

# ── 1. Sync dependencies ──────────────────────────────────────────────────────
echo "[entrypoint.sh] Syncing uv environment..."
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$VIRTUAL_ENV" ]; then
  uv venv "$VIRTUAL_ENV"
fi
. "${VIRTUAL_ENV}/bin/activate"
uv sync --active --no-install-project --no-dev

# TODO: REMOVE
rm -rf /checkpoints/act

# ── 2. Checkpoint guard ───────────────────────────────────────────────────────
# If checkpoint artifacts already exist, there is nothing to do. Ignore the
# bootstrap log so diagnostics do not trip the completion guard.
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 ! -name 'bootstrap.log' -print -quit 2>/dev/null)" ]; then
  echo "[entrypoint.sh] Checkpoints found at ${CHECKPOINT_DIR} — training already complete. Exiting."
  exit 0
fi

# ── 3. Dataset check / download ───────────────────────────────────────────────
# Use hf, do not use huggingface-cli.
DATASET_ROOT="/data"
DATA_DIR="${DATASET_ROOT}/${DATASET_REPO_ID}"
DATASET_REVISION_FILE="${DATA_DIR}/.dataset_revision"
CURRENT_DATASET_REVISION=""
if [ -f "${DATASET_REVISION_FILE}" ]; then
  CURRENT_DATASET_REVISION="$(<"${DATASET_REVISION_FILE}")"
fi

if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ] || [ "${CURRENT_DATASET_REVISION}" != "${DATASET_REVISION}" ]; then
  if [ "${CURRENT_DATASET_REVISION}" != "" ] && [ "${CURRENT_DATASET_REVISION}" != "${DATASET_REVISION}" ]; then
    echo "[entrypoint.sh] Dataset revision mismatch at ${DATA_DIR}: found ${CURRENT_DATASET_REVISION}, need ${DATASET_REVISION}. Refreshing dataset."
  else
    echo "[entrypoint.sh] Dataset not found at ${DATA_DIR}. Downloading from Hugging Face..."
  fi

  rm -rf "${DATA_DIR}"
  mkdir -p "${DATA_DIR}"
  uv run --active --no-sync hf download "${DATASET_REPO_ID}" \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir "${DATA_DIR}"
  printf '%s\n' "${DATASET_REVISION}" > "${DATASET_REVISION_FILE}"
  echo "[entrypoint.sh] Dataset download complete."
else
  echo "[entrypoint.sh] Dataset revision ${DATASET_REVISION} already present at ${DATA_DIR}. Skipping download."
fi

# ── 4. Train ──────────────────────────────────────────────────────────────────
echo "[entrypoint.sh] Starting lerobot-train..."

# export WANDB_MODE=disabled
export WANDB_MODE=online

uv run --active --no-sync lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.revision="${DATASET_REVISION}" \
  --policy.type=act \
  --policy.repo_id="${POLICY_REPO_ID:-ETHRC/act-towelspring26}" \
  --output_dir="${CHECKPOINT_DIR}" \
  --batch_size=28 \
  --policy.push_to_hub=true \
  --job_name=act_training_1 \
  --wandb.project=act \
  --wandb.enable=true \
  --policy.device=cuda
