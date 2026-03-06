#!/usr/bin/env bash
# One-time setup: download a LeRobot dataset from Hugging Face and upload it
# to the team S3 bucket so the training PV is pre-populated.
#
# Usage:
#   BUCKET_NAME=ethrc-xxxx-xxxx ./setup/upload-dataset.sh
#
# Optional overrides:
#   DATASET_REPO_ID=lerobot/pusht   (default)
#   S3_PREFIX=                      (sub-path inside the bucket, default: none)
#
# Requirements:  aws CLI (configured with write access to the bucket)
#                huggingface_hub Python package  (pip install huggingface_hub)
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/pusht}"
BUCKET_NAME="${BUCKET_NAME:-ethrc-ml-data-916780037007}"
S3_PREFIX="${S3_PREFIX:-}"   # leave empty to place dataset at bucket root

TMPDIR_BASE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BASE"' EXIT

LOCAL_PATH="${TMPDIR_BASE}/${DATASET_REPO_ID}"

# ── 1. Download from Hugging Face ────────────────────────────────────────────
echo "[upload-dataset.sh] Downloading '${DATASET_REPO_ID}' from Hugging Face..."
pip install --quiet "huggingface_hub[cli]"
huggingface-cli download "${DATASET_REPO_ID}" \
  --repo-type dataset \
  --local-dir "${LOCAL_PATH}"
echo "[upload-dataset.sh] Download complete."

# ── 2. Sync to S3 ────────────────────────────────────────────────────────────
# Files land at:  s3://BUCKET/${S3_PREFIX}${DATASET_REPO_ID}/
# which maps to:  /data/lerobot/pusht/  inside the training pod
S3_DEST="s3://${BUCKET_NAME}/${S3_PREFIX}${DATASET_REPO_ID}/"
echo "[upload-dataset.sh] Uploading to ${S3_DEST} ..."
aws s3 sync "${LOCAL_PATH}/" "${S3_DEST}" \
  --no-progress \
  --only-show-errors

echo ""
echo "[upload-dataset.sh] Done!"
echo "  S3 path : ${S3_DEST}"
echo "  Pod path: /data/${DATASET_REPO_ID}  (via testing1-data PV)"
echo ""
echo "Next: kubectl apply -k training/"
