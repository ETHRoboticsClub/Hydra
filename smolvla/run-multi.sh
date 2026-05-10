#!/usr/bin/env bash
set -euo pipefail

DATASET_LOCAL_PATH="/data/Insertion"
CHECKPOINT_DIR="/checkpoints/smolvla_training"
HF_REPO_ID="LucaFrat/Insertion"
# HF_MODEL_REPO defaults to LucaFrat/smolVLA but can be overridden
# by the TrainJob's inline export (e.g. LucaFrat/smolVLAbig for the
# parallel multi-GPU "big" training).
HF_MODEL_REPO="${HF_MODEL_REPO:-LucaFrat/smolVLA}"

# Tunables — passed in from the TrainJob env, defaults are smoke-test values.
# Note: lerobot's --batch_size is per-process (per-GPU) under accelerate, so
# effective_batch = NUM_GPUS * BATCH_SIZE_PER_GPU.
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
STEPS="${STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-20}"
LOG_FREQ="${LOG_FREQ:-20}"

export PATH="$HOME/.local/bin:$PATH"

echo "[run.sh] ===== SmolVLA fine-tune (multi-GPU) ====="
echo "[run.sh] NUM_GPUS=$NUM_GPUS  BATCH_PER_GPU=$BATCH_SIZE_PER_GPU  STEPS=$STEPS  SAVE_FREQ=$SAVE_FREQ"

# ── 0. Fix EBS volume permissions ────────────────────────────────────────────
sudo chown -R "$(id -u):$(id -g)" /data /checkpoints

# ── 1. Read HF token ─────────────────────────────────────────────────────────
if [ -f /secrets/hf/HF_TOKEN ]; then
  export HF_TOKEN="$(cat /secrets/hf/HF_TOKEN)"
  echo "[run.sh] HF_TOKEN loaded."
else
  echo "[run.sh] WARNING: /secrets/hf/HF_TOKEN not found."
fi
export HF_HUB_DOWNLOAD_TIMEOUT=60

# ── 2. Checkpoint guard ──────────────────────────────────────────────────────
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
  echo "[run.sh] Checkpoints found at ${CHECKPOINT_DIR} — exiting."
  exit 0
fi

# ── 3. Install lerobot[smolvla] FIRST so it constrains huggingface_hub ───────
echo "[run.sh] Installing lerobot[smolvla]..."
pip install 'lerobot[smolvla]' --break-system-packages
export PATH="$HOME/.local/bin:$PATH"

# ── 4. Dataset bulk download (hf_transfer enabled, max_workers=16) ───────────
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_REPO_ID DATASET_LOCAL_PATH

mkdir -p "${DATASET_LOCAL_PATH}"
if [ -d "${DATASET_LOCAL_PATH}" ] && [ -n "$(ls -A "${DATASET_LOCAL_PATH}" 2>/dev/null)" ]; then
  echo "[run.sh] Dataset dir non-empty at ${DATASET_LOCAL_PATH} — skipping bulk download, will verify."
else
  echo "[run.sh] Downloading dataset from HF..."
  for attempt in 1 2 3 4 5; do
    echo "[run.sh] snapshot_download attempt ${attempt}/5..."
    if python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${HF_REPO_ID}', repo_type='dataset',
    local_dir='${DATASET_LOCAL_PATH}', max_workers=16,
    token=os.environ.get('HF_TOKEN'))"; then
      echo "[run.sh] snapshot_download attempt ${attempt} returned ok."
      break
    fi
    [ "$attempt" = "5" ] && { echo "[run.sh] ERROR: snapshot_download failed."; exit 1; }
    sleep $((attempt * 10))
  done
fi

# ── 4b. Verify-and-fill loop ─────────────────────────────────────────────────
echo "[run.sh] Verifying dataset completeness..."
python <<'PYEOF'
import os, sys, time
from huggingface_hub import HfApi, hf_hub_download
REPO_ID = os.environ['HF_REPO_ID']
LOCAL_DIR = os.environ['DATASET_LOCAL_PATH']
TOKEN = os.environ['HF_TOKEN']
api = HfApi(token=TOKEN)
remote = api.list_repo_files(repo_id=REPO_ID, repo_type='dataset')
print(f"[verify] HF inventory: {len(remote)} files")
missing = [r for r in remote
           if not os.path.exists(os.path.join(LOCAL_DIR, r))
           or os.path.getsize(os.path.join(LOCAL_DIR, r)) == 0]
print(f"[verify] Missing locally: {len(missing)}")
if not missing:
    print("[verify] Dataset complete."); sys.exit(0)
print(f"[verify] Filling {len(missing)} files...")
ok, failed = 0, []
for i, rel in enumerate(missing, 1):
    for attempt in range(3):
        try:
            hf_hub_download(repo_id=REPO_ID, filename=rel, repo_type='dataset',
                            local_dir=LOCAL_DIR, token=TOKEN)
            ok += 1
            if i % 50 == 0 or i == len(missing):
                print(f"[verify] {i}/{len(missing)} ok={ok} failed={len(failed)}")
            break
        except Exception as e:
            if attempt == 2:
                failed.append((rel, str(e)))
            else:
                time.sleep(2 ** attempt)
print(f"[verify] Done: filled {ok}/{len(missing)}, failed {len(failed)}")
if failed: sys.exit(1)
PYEOF

# ── 5. Verify CUDA + lerobot ─────────────────────────────────────────────────
echo "[run.sh] Verifying CUDA and lerobot..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
import lerobot
print('lerobot imported OK')
"
which lerobot-train accelerate

# ── 6. Train via accelerate launch (multi-GPU DDP on a single node) ─────────
echo "[run.sh] Starting SmolVLA fine-tuning on $NUM_GPUS GPUs..."

accelerate launch \
  --multi_gpu \
  --num_processes="$NUM_GPUS" \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${HF_MODEL_REPO}" \
  --dataset.repo_id="${HF_REPO_ID}" \
  --dataset.root="${DATASET_LOCAL_PATH}" \
  --dataset.revision=main \
  --output_dir="${CHECKPOINT_DIR}" \
  --job_name=smolvla_training \
  --batch_size="$BATCH_SIZE_PER_GPU" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false

echo "[run.sh] Training complete. Checkpoints saved to ${CHECKPOINT_DIR}."
ls -la "${CHECKPOINT_DIR}" || true

# ── 7. Upload checkpoints to HuggingFace ─────────────────────────────────────
UPLOAD_OK=false
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[run.sh] Uploading checkpoints to ${HF_MODEL_REPO}..."
  if python -c "
import os
from huggingface_hub import upload_folder, create_repo
create_repo(repo_id='${HF_MODEL_REPO}', repo_type='model', private=True, exist_ok=True, token=os.environ['HF_TOKEN'])
upload_folder(folder_path='${CHECKPOINT_DIR}', repo_id='${HF_MODEL_REPO}',
              repo_type='model', token=os.environ['HF_TOKEN'])
print('Upload complete.')"; then
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
  echo "[run.sh] Run: kubectl cp humanoid/<pod-name>:${CHECKPOINT_DIR} ./smolvla_training -c node"
  echo "[run.sh] Container will stay alive for 2 hours for manual download."
  echo "[run.sh] =================================================="
  sleep 7200
fi
