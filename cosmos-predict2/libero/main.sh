#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/cosmos-predict2"

SHUTDOWN_AFTER="${SHUTDOWN_AFTER:-43200}"  # default 12h, set to 0 to disable

echo "[cosmos-libero] Container bootstrap starting..."
nvidia-smi || true

# Install system packages (non-fatal if they fail)
if ! command -v tmux &>/dev/null || ! command -v ffmpeg &>/dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux ffmpeg || echo "[cosmos-libero] Warning: some packages failed to install, continuing."
fi

# Install uv if not present (not included in the AWS DLC image)
# Install to /data/.uv-bin so it survives pod restarts without re-downloading
UV_BIN="/data/.uv-bin"
mkdir -p "${UV_BIN}"
if [ ! -x "${UV_BIN}/uv" ]; then
  echo "[cosmos-libero] Installing uv to ${UV_BIN}..."
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${UV_BIN}" sh
fi
export PATH="${UV_BIN}:${PATH}"
# Make uv available in all future exec'd shells (e.g. kubectl exec -it)
grep -qxF "export PATH=${UV_BIN}:\$PATH" "${HOME}/.bashrc" 2>/dev/null \
  || echo "export PATH=${UV_BIN}:\$PATH" >> "${HOME}/.bashrc"

# Clone / refresh repo onto the persistent volume so it survives pod restarts
if [ -d "${REPO_DIR}/.git" ]; then
  cd "${REPO_DIR}"
  git fetch origin libero
  git reset --hard origin/libero
else
  git clone --branch libero --single-branch \
    https://github.com/ETHRoboticsClub/cosmos-predict2.git "${REPO_DIR}"
fi

cat <<'EOF'

[cosmos-libero] Ready. Repo is at /data/cosmos-predict2

Quick-start — run steps manually inside this shell:

  cd /data/cosmos-predict2

  # 1. Install deps
  uv sync --extra cu126 && source .venv/bin/activate

  # 2. HF auth (paste token when prompted, or set HF_TOKEN first)
  hf auth login

  # 3. Download model
  python scripts/download_checkpoints.py \
    --model_types video2world --model_sizes 2B --resolution 480 --fps 10

  # 4. Download dataset (~27 GB, safe to re-run if interrupted)
  huggingface-cli download nvidia/LIBERO-Cosmos-Policy \
    --repo-type dataset --include "all_episodes/*" \
    --local-dir datasets/libero_cosmos

  # 5. Convert HDF5 → MP4 + captions
  uv run --with h5py --with pillow --with tqdm \
    python scripts/prepare_libero_cosmos_dataset.py \
    --src datasets/libero_cosmos/all_episodes \
    --out datasets/libero_cosmos_mp4 --fps 10

  # 6. T5 embeddings
  python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/train
  python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/val

  # 7. Smoke test (1 GPU)
  IMAGINAIRE_OUTPUT_ROOT=outputs torchrun \
    --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment=predict2_video2world_training_2b_libero_cosmos \
    trainer.max_iter=5 trainer.validation_iter=1 \
    trainer.max_val_iter=2 checkpoint.save_iter=999999

  # Checkpoints go to: /data/checkpoints

  # Or run everything unattended:
  bash /data/cosmos-predict2/scripts/run/entrypoint.sh

Pod stays alive until the TrainJob is deleted.
EOF

if [ "${SHUTDOWN_AFTER}" -gt 0 ]; then
  echo "[cosmos-libero] Pod will shut down in $((SHUTDOWN_AFTER / 3600))h."
  sleep "${SHUTDOWN_AFTER}"
else
  sleep infinity
fi
