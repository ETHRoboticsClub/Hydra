#!/usr/bin/env bash
set -euo pipefail

DATASET_LOCAL_PATH="/data/Insertion"
CHECKPOINT_DIR="/checkpoints/smolvla_training"
HF_REPO_ID="LucaFrat/Insertion"
HF_MODEL_REPO="LucaFrat/smolVLA"

export PATH="$HOME/.local/bin:$PATH"

echo "[run.sh] ===== SmolVLA fine-tune smoke test ====="

# ── 0. Fix EBS volume permissions ────────────────────────────────────────────
sudo chown -R "$(id -u):$(id -g)" /data /checkpoints

# ── 1. Read HF token ─────────────────────────────────────────────────────────
if [ -f /secrets/hf/HF_TOKEN ]; then
  export HF_TOKEN="$(cat /secrets/hf/HF_TOKEN)"
  echo "[run.sh] HF_TOKEN loaded from /secrets/hf/HF_TOKEN."
else
  echo "[run.sh] WARNING: /secrets/hf/HF_TOKEN not found. Private repo access will fail."
fi
export HF_HUB_DOWNLOAD_TIMEOUT=60

# ── 2. Checkpoint guard ──────────────────────────────────────────────────────
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
  echo "[run.sh] Checkpoints found at ${CHECKPOINT_DIR} — training already complete. Exiting."
  exit 0
fi

# ── 3. Install lerobot[smolvla] FIRST so it constrains huggingface_hub ───────
# (Installing huggingface_hub[cli] separately pulls in 1.x, which breaks
#  lerobot's RevisionNotFoundError raise — keep lerobot's pinned version.)
echo "[run.sh] Installing lerobot[smolvla]..."
pip install 'lerobot[smolvla]' --break-system-packages
export PATH="$HOME/.local/bin:$PATH"

# ── 4. Dataset bulk download (hf_transfer enabled, max_workers=16) ───────────
# hf_transfer is the Rust-based parallel downloader (3-10x faster than the
# default Python downloader for many small files). Pulled in by lerobot via
# huggingface-hub[hf-transfer].
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_REPO_ID  # for the verify python block below
export DATASET_LOCAL_PATH

mkdir -p "${DATASET_LOCAL_PATH}"
if [ -d "${DATASET_LOCAL_PATH}" ] && [ -n "$(ls -A "${DATASET_LOCAL_PATH}" 2>/dev/null)" ]; then
  echo "[run.sh] Dataset dir non-empty at ${DATASET_LOCAL_PATH} — skipping bulk download, will verify."
else
  echo "[run.sh] Dataset not found. Downloading from Hugging Face..."
  for attempt in 1 2 3 4 5; do
    echo "[run.sh] snapshot_download attempt ${attempt}/5 (hf_transfer=1, max_workers=16)..."
    if python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO_ID}',
    repo_type='dataset',
    local_dir='${DATASET_LOCAL_PATH}',
    max_workers=16,
    token=os.environ.get('HF_TOKEN'),
)
"; then
      echo "[run.sh] snapshot_download attempt ${attempt} returned ok."
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

# ── 4b. Verify-and-fill loop (snapshot_download silently drops files) ────────
# We saw 25% of files missing despite snapshot_download exit 0. Always compare
# HF inventory to disk and hf_hub_download whatever is missing.
echo "[run.sh] Verifying dataset completeness against HF inventory..."
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
    print("[verify] Dataset complete.")
    sys.exit(0)

print(f"[verify] Filling {len(missing)} files via hf_hub_download...")
ok, failed = 0, []
for i, rel in enumerate(missing, 1):
    for attempt in range(3):
        try:
            hf_hub_download(repo_id=REPO_ID, filename=rel,
                            repo_type='dataset', local_dir=LOCAL_DIR, token=TOKEN)
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
if failed:
    print("[verify] Failed files (first 10):")
    for rel, err in failed[:10]:
        print(f"  {rel}: {err[:120]}")
    sys.exit(1)
PYEOF

# ── 6. Verify CUDA + lerobot ─────────────────────────────────────────────────
echo "[run.sh] Verifying CUDA and lerobot..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
print(f'CUDA version:    {torch.version.cuda}')
print(f'GPU count:       {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
import lerobot
print('lerobot imported OK')
"
which lerobot-train

# ── 7. Train (smoke test: 200 steps) ─────────────────────────────────────────
echo "[run.sh] Starting SmolVLA fine-tuning..."

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${HF_MODEL_REPO}" \
  --dataset.repo_id="${HF_REPO_ID}" \
  --dataset.root="${DATASET_LOCAL_PATH}" \
  --dataset.revision=main \
  --output_dir="${CHECKPOINT_DIR}" \
  --job_name=smolvla_training \
  --batch_size=64 \
  --steps=200 \
  --save_freq=100 \
  --log_freq=20 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false

echo "[run.sh] Training complete. Checkpoints saved to ${CHECKPOINT_DIR}."
ls -la "${CHECKPOINT_DIR}" || true

# ── 8. Upload checkpoints to HuggingFace ─────────────────────────────────────
UPLOAD_OK=false
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[run.sh] Uploading checkpoints to ${HF_MODEL_REPO}..."
  if python -c "
import os
from huggingface_hub import upload_folder, create_repo
create_repo(repo_id='${HF_MODEL_REPO}', repo_type='model', private=True, exist_ok=True, token=os.environ['HF_TOKEN'])
upload_folder(
    folder_path='${CHECKPOINT_DIR}',
    repo_id='${HF_MODEL_REPO}',
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
  echo "[run.sh] Run: kubectl cp humanoid/<pod-name>:${CHECKPOINT_DIR} ./smolvla_training -c node"
  echo "[run.sh] Container will stay alive for 2 hours for manual download."
  echo "[run.sh] =================================================="
  sleep 7200
fi
