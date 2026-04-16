#!/usr/bin/env bash
set -euo pipefail

DATASET_LOCAL_PATH="/data/PnPCubeLine"
CHECKPOINT_DIR="/checkpoints/g1_finetune"
REPO_URL="https://github.com/LucaFrat/Isaac-GR00T.git"
REPO_DIR="$HOME/Isaac-GR00T"


export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "NOVEMBER ==================="

# Fix EBS volume permissions
sudo chown -R $(id -u):$(id -g) /data /checkpoints

# Container has CUDA runtime but no toolkit (nvcc). Create a stub nvcc so
# libraries (deepspeed/transformers) that check for it at import time don't crash.
export CUDA_HOME="/tmp/.cuda_stub"
mkdir -p "${CUDA_HOME}/bin"
printf '#!/bin/sh\necho "nvcc: NVIDIA (R) Cuda compiler driver"\necho "Cuda compilation tools, release 12.4, V12.4.131"\n' > "${CUDA_HOME}/bin/nvcc"
chmod +x "${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export DS_BUILD_OPS=0
echo "[run.sh] Created stub nvcc at ${CUDA_HOME}/bin/nvcc"

echo "[run.sh] Installing huggingface_hub..."
pip install huggingface_hub[cli] --break-system-packages

# ── 1. Checkpoint guard ──────────────────────────────────────────────────────
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
  echo "[run.sh] Checkpoints found at ${CHECKPOINT_DIR} — training already complete. Exiting."
  exit 0
fi

# ── 2. Dataset check / download ──────────────────────────────────────────────
HF_REPO_ID="LucaFrat/PnPCubeLine"

if [ -f /secrets/hf/HF_TOKEN ]; then
  export HF_TOKEN=$(cat /secrets/hf/HF_TOKEN)
fi
export HF_HUB_DOWNLOAD_TIMEOUT=60

if [ -d "${DATASET_LOCAL_PATH}" ] && [ -n "$(ls -A "${DATASET_LOCAL_PATH}" 2>/dev/null)" ]; then
  echo "[run.sh] Dataset found at ${DATASET_LOCAL_PATH}. Skipping download."
else
  echo "[run.sh] Dataset not found. Downloading from Hugging Face..."
  mkdir -p "${DATASET_LOCAL_PATH}"
  for attempt in 1 2 3 4 5; do
    echo "[run.sh] snapshot_download attempt ${attempt}/5..."
    if python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO_ID}',
    repo_type='dataset',
    local_dir='${DATASET_LOCAL_PATH}',
    max_workers=4,
    token=os.environ.get('HF_TOKEN'),
)
"; then
      echo "[run.sh] Dataset downloaded from Hugging Face to ${DATASET_LOCAL_PATH}."
      break
    fi
    if [ "${attempt}" = "5" ]; then
      echo "[run.sh] ERROR: snapshot_download failed after 5 attempts."
      exit 1
    fi
    backoff=$((attempt * 10))
    echo "[run.sh] Download attempt ${attempt} failed; retrying in ${backoff}s..."
    sleep "${backoff}"
  done
fi

# ── 3. Install uv + Isaac-GR00T ─────────────────────────────────────────────
echo "[run.sh] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "[run.sh] Cloning Isaac-GR00T..."
git clone --recurse-submodules "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

# Remove flash-attn from dependencies (no nvcc in container, need pre-built wheel)
echo "[run.sh] Patching out flash-attn from dependencies (will install pre-built wheel)..."
sed -i 's/"flash-attn[^"]*",\?//' pyproject.toml

echo "[run.sh] Installing Isaac-GR00T dependencies..."
uv sync --python 3.10

# Detect PyTorch version and install matching pre-built flash-attn wheel
TORCH_VER=$(uv run python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
echo "[run.sh] Detected PyTorch ${TORCH_VER}, installing pre-built flash-attn..."
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TORCH_VER}cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

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

export NUM_GPUS=4

uv run python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path "${DATASET_LOCAL_PATH}" \
    --embodiment_tag UNITREE_G1 \
    --no_tune_llm \
    --no_tune_visual \
    --num_gpus $NUM_GPUS \
    --output_dir "${CHECKPOINT_DIR}" \
    --save_total_limit 3 \
    --save_steps 12 \
    --max_steps 36 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 512 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

echo "[run.sh] Training complete. Checkpoints saved to ${CHECKPOINT_DIR}."

# ── 6. Upload checkpoints to HuggingFace ────────────────────────────────────
UPLOAD_OK=false
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[run.sh] Uploading checkpoints to HuggingFace..."
  if python -c "
import os
from huggingface_hub import upload_folder
upload_folder(
    folder_path='${CHECKPOINT_DIR}',
    repo_id='LucaFrat/PnPCubeLineChecks',
    repo_type='model',
    token=os.environ['HF_TOKEN'],
)
print('Upload complete.')
"; then
    UPLOAD_OK=true
  else
    echo "[run.sh] ERROR: HuggingFace upload failed."
  fi
else
  echo "[run.sh] WARNING: HF_TOKEN not set, skipping checkpoint upload."
fi

if [ "$UPLOAD_OK" = false ]; then
  echo "[run.sh] =================================================="
  echo "[run.sh] Checkpoints are at: ${CHECKPOINT_DIR}"
  echo "[run.sh] Run: kubectl cp humanoid/<pod-name>:${CHECKPOINT_DIR} ./g1_finetune -c node"
  echo "[run.sh] Container will stay alive for 2 hours for manual download."
  echo "[run.sh] =================================================="
  sleep 7200
fi
