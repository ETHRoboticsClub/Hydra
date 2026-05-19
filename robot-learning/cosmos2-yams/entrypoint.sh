#!/usr/bin/env bash
# Cosmos-Predict2 YAMS video2world LoRA fine-tune entrypoint.
# Reference: https://github.com/ETHRoboticsClub/cosmos-predict2/blob/main/scripts/mimic.md
# Mounted volumes (from deploy.yaml):
#   /data  — persistent (uv venv, cloned repo, raw + prepared datasets, T5 embeds, local outputs)
#   /s3    — s3-data PVC; final artifacts copied here under robot-learning/runs/$JOB_NAME/
#
# Pod restarts within the same launch are recovery-safe: every heavy step
# (clone, sync, download, prepare, T5, checkpoint dl) is gated on an idempotent
# disk check so re-running this script picks up where it left off.

set -euo pipefail

# ---------- config ----------
REPO_DIR="/data/cosmos-predict2"
HF_DATASET="${HF_DATASET:-ETHRC/robot-learning-fs26}"
DATASET_LOCAL_DIR="datasets/yams_lerobot"
DATASET_PREPARED_DIR="datasets/yams_cosmos_mp4"
EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_yams}"
CAMERA="${CAMERA:-top}"
NPROC="${NPROC:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
SMOKETEST="${SMOKETEST:-0}"
RUN_NAME="${JOB_NAME:-cosmos2-yams-$(date +%Y%m%dT%H%M%S)}"
S3_RUN_DIR="/s3/robot-learning/runs/${RUN_NAME}"

echo "[cosmos2-yams] start: run=${RUN_NAME} nproc=${NPROC} smoketest=${SMOKETEST} dataset=${HF_DATASET}"
nvidia-smi || true

# ---------- 0. system deps ----------
if ! command -v ffmpeg &>/dev/null; then
  echo "[cosmos2-yams] installing ffmpeg..."
  apt-get update -qq && apt-get install -y -qq ffmpeg
fi

# ---------- 1. uv (persisted on /data) ----------
UV_BIN="/data/.uv-bin"
mkdir -p "${UV_BIN}"
if [ ! -x "${UV_BIN}/uv" ]; then
  echo "[cosmos2-yams] installing uv to ${UV_BIN}..."
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${UV_BIN}" sh
fi
export PATH="${UV_BIN}:${PATH}"

# ---------- 2. clone / refresh cosmos-predict2 (main branch — yams cfg lives there) ----------
if [ -d "${REPO_DIR}/.git" ]; then
  cd "${REPO_DIR}"
  git fetch --depth 1 origin main
  git reset --hard FETCH_HEAD
else
  git clone --depth 1 --branch main --single-branch \
    https://github.com/ETHRoboticsClub/cosmos-predict2.git "${REPO_DIR}"
  cd "${REPO_DIR}"
fi

# ---------- 3. uv sync + flash-attn ----------
# flash-attn is a runtime req per Nico and is NOT in cosmos-predict2's pyproject.
# `uv add` mutates pyproject which we don't want; install into the venv with `uv pip`.
uv sync --extra cu126
uv pip install flash-attn --no-build-isolation || \
  echo "[cosmos2-yams] WARN: flash-attn install failed (already present? continuing)"

# Activate venv so subsequent python/huggingface-cli/torchrun run from it.
# shellcheck source=/dev/null
source .venv/bin/activate

# ---------- 4. patch prepare script: tolerate bad episodes ----------
# Nico: "the affected episodes will be logged when you try to convert them, just
# rm those in the entrypoint." Upstream only catches CalledProcessError; broaden
# to Exception so any per-episode failure (short clip / corrupt mp4 / ValueError
# on 0-frame ffmpeg output) is skipped+logged rather than crashing the whole
# conversion. Also rewrite the handler body — it accesses `e.stderr.decode()`
# which exists on CalledProcessError but blows up with AttributeError on
# ValueError. Use str(e) for the non-subprocess branch. Both sed expressions
# are idempotent: after the first run their targets are gone.
sed -i 's/except subprocess.CalledProcessError as e:/except Exception as e:/' \
  scripts/prepare_lerobot_cosmos_dataset.py || true
sed -i "s|e\.stderr\.decode()\[-300:\]|(e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e))[-300:]|g" \
  scripts/prepare_lerobot_cosmos_dataset.py || true

# ---------- 5. HF auth ----------
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[cosmos2-yams] logging in to HuggingFace..."
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || \
    huggingface-cli login --token "${HF_TOKEN}"
fi

# ---------- 6. download model checkpoint (gated: nvidia/Cosmos-Predict2-2B-Video2World) ----------
# download_checkpoints.py grabs the full cosmos suite (v2w + cosmos-reason1 + t5
# + cosmos-guardrail1 + meta-llama/llama-guard-3-8b). For fine-tuning only v2w
# is *required* — the guard models are runtime safety filters we don't use.
# The script logs per-model 403s and exits 0 regardless, so we re-check the v2w
# file explicitly and bail when it's still missing (the failure mode that
# definitely blocks training).
#
# STRICT_MODEL_DOWNLOAD=1 (default): bail if v2w is missing post-download.
# STRICT_MODEL_DOWNLOAD=0          : warn and continue (useful for prepare-only /
#                                    T5-only smoke testing without v2w access).
V2W_MODEL_FILE="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt"
if [ ! -f "${V2W_MODEL_FILE}" ]; then
  echo "[cosmos2-yams] downloading model checkpoints..."
  uv run python scripts/download_checkpoints.py \
    --model_types video2world \
    --model_sizes 2B \
    --resolution 480 \
    --fps 10
fi
if [ ! -f "${V2W_MODEL_FILE}" ]; then
  echo "[cosmos2-yams] WARN: ${V2W_MODEL_FILE} still missing after download." >&2
  echo "[cosmos2-yams]   Likely the HF token in hf-secret has no access to" >&2
  echo "[cosmos2-yams]   nvidia/Cosmos-Predict2-2B-Video2World (gated repo)." >&2
  echo "[cosmos2-yams]   Accept terms at https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World" >&2
  if [ "${STRICT_MODEL_DOWNLOAD:-1}" = "1" ]; then
    echo "[cosmos2-yams] STRICT_MODEL_DOWNLOAD=1 — bailing." >&2
    exit 1
  else
    echo "[cosmos2-yams] STRICT_MODEL_DOWNLOAD=0 — continuing; training will crash on model load." >&2
  fi
fi

# ---------- 7. download dataset ----------
if [ ! -d "${DATASET_LOCAL_DIR}/meta" ]; then
  echo "[cosmos2-yams] downloading dataset ${HF_DATASET}..."
  mkdir -p "${DATASET_LOCAL_DIR}"
  huggingface-cli download "${HF_DATASET}" \
    --repo-type dataset \
    --local-dir "${DATASET_LOCAL_DIR}"
fi

# ---------- 8. convert LeRobot → VideoDataset (MP4 + captions) ----------
# Sentinel is a "done" marker that's only created after a successful run.
# If a previous attempt crashed mid-way the dir exists but the marker doesn't,
# so we wipe + rerun. ffmpeg is invoked with -y so re-encoding is idempotent.
PREPARE_DONE="${DATASET_PREPARED_DIR}/.prepare_done"
if [ ! -f "${PREPARE_DONE}" ]; then
  echo "[cosmos2-yams] preparing dataset → ${DATASET_PREPARED_DIR} (camera=${CAMERA})..."
  rm -rf "${DATASET_PREPARED_DIR}"
  uv run --with pyarrow --with pandas --with tqdm \
    python scripts/prepare_lerobot_cosmos_dataset.py \
      --src "${DATASET_LOCAL_DIR}" \
      --out "${DATASET_PREPARED_DIR}" \
      --camera "${CAMERA}" \
      --fps 10 \
      --video-size 480 640
  touch "${PREPARE_DONE}"
fi

# ---------- 9. T5 embeddings (train + val) ----------
t5_done() { ls "${1}/t5_xxl"/*.pkl >/dev/null 2>&1; }
for split in train val; do
  if ! t5_done "${DATASET_PREPARED_DIR}/${split}"; then
    echo "[cosmos2-yams] generating T5 embeddings (${split})..."
    uv run python -m scripts.get_t5_embeddings --dataset_path "${DATASET_PREPARED_DIR}/${split}"
  fi
done

# ---------- 10. train ----------
export WANDB_PROJECT="${WANDB_PROJECT:-cosmos-yams}"
RUN_ROOT="outputs/posttraining/video2world_lora/2b_yams"
CKPT_DIR="${RUN_ROOT}/checkpoints"

if [ "${SMOKETEST}" = "1" ]; then
  # Clear any stale state from a previous smoke/full run so the smoke test
  # rebuilds config + starts from iter 0.
  rm -f "${CKPT_DIR}/latest_checkpoint.txt" \
        "${RUN_ROOT}/config.pkl" \
        "${RUN_ROOT}/config.yaml"
  echo "[cosmos2-yams] smoke test: 5 iters, ${NPROC} GPU(s)"
  uv run torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment="${EXPERIMENT}" \
    dataloader_train.batch_size=1 \
    dataloader_val.batch_size=1 \
    trainer.max_iter=5 \
    trainer.grad_accum_iter=4 \
    trainer.validation_iter=1 \
    trainer.max_val_iter=2 \
    checkpoint.save_iter=999999
  echo "[cosmos2-yams] smoke test complete (no checkpoint saved by design)"
else
  # A100-40 (p4d.24xlarge) variant per Nico's screenshot: bs=1, grad_accum=4.
  # Effective batch = 1 (per-rank) * 8 (DP ranks) * 4 (grad accum) = 32.
  # Drops the heavy draw_sample decode steps (is_sample=False, is_x0=False)
  # since VAE decode doesn't co-fit on 40 GB.
  echo "[cosmos2-yams] full training: 7000 iters, ${NPROC} GPU(s) (a100-40 variant)"
  uv run torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment="${EXPERIMENT}" \
    dataloader_train.batch_size=1 \
    dataloader_val.batch_size=1 \
    trainer.max_iter=7000 \
    trainer.grad_accum_iter=4 \
    trainer.validation_iter=50 \
    trainer.max_val_iter=10 \
    trainer.callbacks.draw_sample.every_n=200 \
    trainer.callbacks.draw_sample.is_x0=False \
    trainer.callbacks.draw_sample.is_sample=False \
    "trainer.callbacks.draw_sample.guidance=[7.0]" \
    checkpoint.save_iter=500
fi

# ---------- 11. persist artifacts to s3-data PVC ----------
# Pod runs as root (deploy.yaml securityContext) so PVC writes succeed.
echo "[cosmos2-yams] copying outputs to ${S3_RUN_DIR}..."
mkdir -p "${S3_RUN_DIR}"
if [ -d "${RUN_ROOT}" ]; then
  cp -r "${RUN_ROOT}" "${S3_RUN_DIR}/"
fi
{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "experiment=${EXPERIMENT}"
  echo "smoketest=${SMOKETEST}"
  echo "nproc=${NPROC}"
  echo "hf_dataset=${HF_DATASET}"
  echo "camera=${CAMERA}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${S3_RUN_DIR}/run-info.txt"

echo "[cosmos2-yams] done."
