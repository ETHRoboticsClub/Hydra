#!/usr/bin/env bash
set -euo pipefail

# Minimal lerobot training entrypoint
# Configure via environment variables:
#   DATASET_REPO_ID   - HuggingFace dataset (e.g., "lerobot/pusht")
#   OUTPUT_DIR        - Checkpoint directory (default: /checkpoints)

DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/pusht}"
OUTPUT_DIR="${OUTPUT_DIR:-/checkpoints/testing}"

# Setup environment
export VIRTUAL_ENV="~/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "$(dirname "$0")"

if [ ! -d "$VIRTUAL_ENV" ]; then
  uv venv "$VIRTUAL_ENV"
fi

# shellcheck source=/dev/null
. "${VIRTUAL_ENV}/bin/activate"

uv sync --active --no-install-project --no-dev 2>/dev/null || uv pip install lerobot

rm -rf "${OUTPUT_DIR}"

exec uv run --active --no-sync lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root=/data \
  --output_dir="${OUTPUT_DIR}" \
  --policy.type=act \
  --batch_size=8 \
  --num_workers=4 \
  --save_freq=1000 \
  --log_freq=50
