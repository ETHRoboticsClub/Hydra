#!/usr/bin/env bash
set -euo pipefail

DATASET_LOCAL_PATH="$HOME/dataset/G1-sim"
CHECKPOINT_DIR="/checkpoints/ETHRC/g1_finetune/checkpoints"
REPO_URL="https://github.com/LucaFrat/Isaac-GR00T.git"
REPO_DIR="$HOME/Isaac-GR00T"


echo "[run.sh] Installing lerobot..."

pip install lerobot --break-system-packages

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ── 1. Checkpoint guard ──────────────────────────────────────────────────────
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
  echo "[run.sh] Checkpoints found at ${CHECKPOINT_DIR} — training already complete. Exiting."
  exit 0
fi

# ── 2. Dataset check / download ──────────────────────────────────────────────
HF_REPO_ID="LucaFrat/G1-sim"

if [ -d "${DATASET_LOCAL_PATH}" ] && [ -n "$(ls -A "${DATASET_LOCAL_PATH}" 2>/dev/null)" ]; then
  echo "[run.sh] Dataset found at ${DATASET_LOCAL_PATH}. Skipping download."
else
  echo "[run.sh] Dataset not found. Downloading from Hugging Face..."
  # pip install huggingface_hub --break-system-packages
  mkdir -p "${DATASET_LOCAL_PATH}"
  python -m huggingface_hub.cli download "${HF_REPO_ID}" \
    --repo-type dataset \
    --local-dir "${DATASET_LOCAL_PATH}"
  echo "[run.sh] Dataset downloaded from Hugging Face to ${DATASET_LOCAL_PATH}."
fi

# ── 3. Install uv + Isaac-GR00T ─────────────────────────────────────────────
echo "[run.sh] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "[run.sh] Cloning Isaac-GR00T..."
git clone --recurse-submodules "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

echo "[run.sh] Installing Isaac-GR00T dependencies (this may take ~15 min for flash-attn)..."
uv sync --python 3.10
uv pip install -e .

# ── 4. Verify CUDA ──────────────────────────────────────────────────────────
echo "[run.sh] Verifying CUDA..."
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
print(f'CUDA version:    {torch.version.cuda}')
print(f'GPU count:       {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── 5. Train ─────────────────────────────────────────────────────────────────
echo "[run.sh] Starting GR00T fine-tuning..."

export NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path "${DATASET_LOCAL_PATH}" \
    --embodiment_tag UNITREE_G1 \
    --num_gpus $NUM_GPUS \
    --output_dir "${CHECKPOINT_DIR}" \
    --save_total_limit 5 \
    --save_steps 500 \
    --max_steps 5000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

echo "[run.sh] Training complete. Checkpoints saved to ${CHECKPOINT_DIR}."
