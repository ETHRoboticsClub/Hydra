#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/cosmos-predict2"
NPROC="${NPROC:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
SMOKETEST="${SMOKETEST:-0}"

echo "[cosmos-libero] Starting (${NPROC} GPU(s), SMOKETEST=${SMOKETEST})..."
nvidia-smi || true

# System packages
if ! command -v ffmpeg &>/dev/null; then
  apt-get update -qq && apt-get install -y -qq ffmpeg || echo "[cosmos-libero] Warning: ffmpeg install failed, continuing."
fi

# uv — persisted on the data volume so it survives pod restarts without re-downloading
UV_BIN="/data/.uv-bin"
mkdir -p "${UV_BIN}"
if [ ! -x "${UV_BIN}/uv" ]; then
  echo "[cosmos-libero] Installing uv to ${UV_BIN}..."
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${UV_BIN}" sh
fi
export PATH="${UV_BIN}:${PATH}"

# Clone / refresh repo onto the persistent volume
if [ -d "${REPO_DIR}/.git" ]; then
  cd "${REPO_DIR}"
  git fetch origin libero
  git reset --hard origin/libero
else
  git clone --branch libero --single-branch \
    https://github.com/ETHRoboticsClub/cosmos-predict2.git "${REPO_DIR}"
fi

cd "${REPO_DIR}"
uv sync --extra cu126
source .venv/bin/activate

# HF auth (token injected from hf-secret)
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "${HF_TOKEN}"
fi

# Sync data from S3 (IRSA provides credentials via the service account)
ulimit -n 65535
echo "[cosmos-libero] Syncing checkpoints from S3..."
aws s3 sync s3://ethrc-ml-data-916780037007/cosmos-predict2-libero/checkpoints \
  "${REPO_DIR}/checkpoints" --region us-east-1
echo "[cosmos-libero] Syncing datasets from S3..."
aws s3 sync s3://ethrc-ml-data-916780037007/cosmos-predict2-libero/datasets \
  "${REPO_DIR}/datasets" --region us-east-1

CKPT_DIR=outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints

if [ "${SMOKETEST}" = "1" ]; then
  rm -f "${CKPT_DIR}/latest_checkpoint.txt" \
        outputs/posttraining/video2world_lora/2b_libero_cosmos/config.pkl \
        outputs/posttraining/video2world_lora/2b_libero_cosmos/config.yaml

  echo "[cosmos-libero] Smoke test: 50 iters, ${NPROC} GPU(s)..."
  IMAGINAIRE_OUTPUT_ROOT=outputs uv run torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment=predict2_video2world_training_2b_libero_cosmos \
    model_parallel.context_parallel_size=2 \
    dataloader_train.batch_size=16 \
    dataloader_val.batch_size=1 \
    trainer.grad_accum_iter=2 \
    trainer.max_iter=50 \
    trainer.validation_iter=5 \
    trainer.max_val_iter=2 \
    trainer.callbacks.draw_sample.every_n=25 \
    trainer.callbacks.draw_sample.is_sample=True \
    trainer.callbacks.draw_sample.show_all_frames=True \
    "trainer.callbacks.draw_sample.guidance=[7.0]" \
    checkpoint.save_iter=1000

  ITER=$(cat "${CKPT_DIR}/latest_checkpoint.txt")
  echo "[cosmos-libero] Smoke test: evaluating..."
  uv run python scripts/eval_libero_cosmos.py --out eval/base
  uv run python scripts/eval_libero_cosmos.py --out eval/finetuned \
    --lora-checkpoint "${CKPT_DIR}/model/${ITER}"
  echo "[cosmos-libero] Smoke test complete."
  exit 0
fi

# Full training run
rm -f "${CKPT_DIR}/latest_checkpoint.txt"
echo "[cosmos-libero] Full training: 7000 iters, ${NPROC} GPU(s)..."
IMAGINAIRE_OUTPUT_ROOT=outputs uv run torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos \
  model_parallel.context_parallel_size=1 \
  dataloader_train.batch_size=8 \
  dataloader_val.batch_size=2 \
  trainer.max_iter=7000 \
  trainer.grad_accum_iter=2 \
  trainer.validation_iter=50 \
  trainer.max_val_iter=10 \
  trainer.callbacks.draw_sample.every_n=200 \
  trainer.callbacks.draw_sample.is_sample=True \
  trainer.callbacks.draw_sample.show_all_frames=True \
  "trainer.callbacks.draw_sample.guidance=[7.0]" \
  checkpoint.save_iter=500

echo "[cosmos-libero] Evaluating..."
ITER=$(cat "${CKPT_DIR}/latest_checkpoint.txt")
uv run python scripts/eval_libero_cosmos.py --out eval/base
uv run python scripts/eval_libero_cosmos.py --out eval/finetuned \
  --lora-checkpoint "${CKPT_DIR}/model/${ITER}"
echo "[cosmos-libero] Full run complete. Checkpoints at ${REPO_DIR}/${CKPT_DIR}"
