#!/usr/bin/env bash
# Full unattended workflow for cosmos-predict2 libero LoRA fine-tuning.
# Run this inside the interactive pod once you've validated each step manually.
#
# Requires: HF_TOKEN env var set (or run `hf auth login` before calling this)
# GPU count: defaults to 1 for smoke-test mode; set NPROC=8 for full training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/data/cosmos-predict2"
DATASET_REPO_ID="nvidia/LIBERO-Cosmos-Policy"
DATASET_ROOT="${REPO_DIR}/datasets/libero_cosmos"
MP4_ROOT="${REPO_DIR}/datasets/libero_cosmos_mp4"
CHECKPOINT_DIR="/data/checkpoints"
NPROC="${NPROC:-1}"

cd "${REPO_DIR}"

# ── 0. Ensure uv is available ────────────────────────────────────────────────
UV_BIN="/data/.uv-bin"
mkdir -p "${UV_BIN}"
if [ ! -x "${UV_BIN}/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${UV_BIN}" sh
fi
export PATH="${UV_BIN}:${PATH}"

# ── 1. Dependencies ──────────────────────────────────────────────────────────
echo "[entrypoint] Syncing uv environment..."
uv sync --extra cu126
source .venv/bin/activate

# ── 2. HF auth ───────────────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "${HF_TOKEN}"
fi

# ── 3. Download model (idempotent — skips if already present) ────────────────
MODEL_MARKER="/data/.model-downloaded"
if [ ! -f "${MODEL_MARKER}" ]; then
  echo "[entrypoint] Downloading Cosmos-Predict2-2B-Video2World model..."
  python scripts/download_checkpoints.py \
    --model_types video2world --model_sizes 2B --resolution 480 --fps 10
  touch "${MODEL_MARKER}"
else
  echo "[entrypoint] Model already downloaded, skipping."
fi

# ── 4. Download dataset (~27 GB, safe to re-run if interrupted) ───────────────
echo "[entrypoint] Downloading LIBERO dataset..."
huggingface-cli download "${DATASET_REPO_ID}" \
  --repo-type dataset --include "all_episodes/*" \
  --local-dir "${DATASET_ROOT}"

# ── 5. Convert HDF5 → MP4 + captions ────────────────────────────────────────
if [ ! -d "${MP4_ROOT}/train" ]; then
  echo "[entrypoint] Converting HDF5 → MP4..."
  uv run --with h5py --with pillow --with tqdm \
    python scripts/prepare_libero_cosmos_dataset.py \
    --src "${DATASET_ROOT}/all_episodes" \
    --out "${MP4_ROOT}" \
    --fps 10
else
  echo "[entrypoint] MP4 dataset already converted, skipping."
fi

# ── 6. T5 embeddings ─────────────────────────────────────────────────────────
for SPLIT in train val; do
  EMBED_MARKER="/data/.t5-${SPLIT}-done"
  if [ ! -f "${EMBED_MARKER}" ]; then
    echo "[entrypoint] Generating T5 embeddings for ${SPLIT}..."
    python -m scripts.get_t5_embeddings --dataset_path "${MP4_ROOT}/${SPLIT}"
    touch "${EMBED_MARKER}"
  else
    echo "[entrypoint] T5 embeddings for ${SPLIT} already done, skipping."
  fi
done

# ── 7. Smoke test (quick sanity check, exits cleanly) ────────────────────────
echo "[entrypoint] Running smoke test (${NPROC} GPU(s))..."
IMAGINAIRE_OUTPUT_ROOT=outputs torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos \
  trainer.max_iter=5 \
  trainer.validation_iter=1 \
  trainer.max_val_iter=2 \
  checkpoint.save_iter=999999
echo "[entrypoint] Smoke test passed."

# ── 8. Full training ─────────────────────────────────────────────────────────
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
  echo "[entrypoint] Checkpoints found at ${CHECKPOINT_DIR} — training already complete. Exiting."
  exit 0
fi

echo "[entrypoint] Starting full training (${NPROC} GPU(s))..."
IMAGINAIRE_OUTPUT_ROOT="${CHECKPOINT_DIR}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos

echo "[entrypoint] Training complete. Checkpoints at ${CHECKPOINT_DIR}"
