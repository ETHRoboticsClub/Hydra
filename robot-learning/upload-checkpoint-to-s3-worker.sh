#!/bin/sh
set -eu

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/checkpoints/act}"
S3_BUCKET_URI="s3://ethrc-ml-data-916780037007"
CHECKPOINT_NAME="${1:-}"
S3_TARGET_DIR="${2:-}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"

usage() {
  cat >&2 <<'EOF'
Usage:
  upload-checkpoint-to-s3-worker.sh <checkpoint-name> <s3-dir>
EOF
}

log() {
  printf '[upload-checkpoint-to-s3] %s\n' "$*" >&2
}

die() {
  log "$*"
  exit 1
}

require_args() {
  if [ -z "${CHECKPOINT_NAME}" ] || [ -z "${S3_TARGET_DIR}" ]; then
    usage
    exit 1
  fi

  case "${S3_TARGET_DIR}" in
    s3://*)
    die "Pass an S3 directory like act/run-001, not a full s3:// path."
      ;;
  esac
}

resolve_checkpoint_path() {
  root="${CHECKPOINT_ROOT%/}"
  candidate=""
  fallback=""

  if [ ! -d "${root}" ]; then
    die "Checkpoint root ${root} does not exist."
  fi

  if [ -d "${root}/${CHECKPOINT_NAME}" ]; then
    candidate="${root}/${CHECKPOINT_NAME}"
    if [ "$(count_files "${candidate}")" -gt 0 ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    fallback="${candidate}"
  fi

  if [ -d "${root}/checkpoints/${CHECKPOINT_NAME}" ]; then
    candidate="${root}/checkpoints/${CHECKPOINT_NAME}"
    if [ "$(count_files "${candidate}")" -gt 0 ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    fallback="${candidate}"
  fi

  if [ -n "${fallback}" ]; then
    log "Checkpoint ${CHECKPOINT_NAME} exists but contains no files: ${fallback}"
  else
    log "Checkpoint ${CHECKPOINT_NAME} was not found under ${root}."
  fi

  log "Available checkpoint directories:"
  find "${root}" "${root}/checkpoints" -mindepth 1 -maxdepth 1 -type d 2>/dev/null -exec basename {} \; | sort -u >&2 || true
  exit 1
}

count_files() {
  source_dir="$1"
  find -L "${source_dir}" -type f | wc -l | awk '{print $1}'
}

compute_size_kib() {
  source_dir="$1"
  du -sk "${source_dir}" | awk '{print $1}'
}

human_size() {
  kib="$1"
  awk -v kib="${kib}" '
    function human(x) {
      split("KiB MiB GiB TiB", units, " ")
      i = 1
      while (x >= 1024 && i < 5) {
        x /= 1024
        i++
      }
      return sprintf("%.1f %s", x, units[i])
    }
    BEGIN { print human(kib) }
  '
}

build_destination_uri() {
  source_dir="$1"
  normalized_dir="${S3_TARGET_DIR#/}"
  normalized_dir="${normalized_dir%/}"
  basename="$(basename "${source_dir}")"
  printf '%s/%s/%s\n' "${S3_BUCKET_URI}" "${normalized_dir}" "${basename}"
}

upload_with_aws_cli() {
  source_dir="$1"
  destination_uri="$2"

  log "Uploading with aws cli"
  aws s3 cp "${source_dir}" "${destination_uri}" --recursive
}

main() {
  require_args

  checkpoint_path="$(resolve_checkpoint_path)"
  destination_uri="$(build_destination_uri "${checkpoint_path}")"
  file_count="$(count_files "${checkpoint_path}")"
  if [ "${file_count}" = "0" ]; then
    die "Checkpoint ${checkpoint_path} contains no files."
  fi

  size_kib="$(compute_size_kib "${checkpoint_path}")"

  log "Checkpoint root: ${CHECKPOINT_ROOT}"
  log "Selected checkpoint: ${CHECKPOINT_NAME}"
  log "Source directory: ${checkpoint_path}"
  log "File count: ${file_count}"
  log "Total size: $(human_size "${size_kib}")"
  log "Destination: ${destination_uri}"

  command -v aws >/dev/null 2>&1 || die "aws CLI is not available in this container."
  upload_with_aws_cli "${checkpoint_path}" "${destination_uri}"
  log "Upload succeeded: ${destination_uri}"
}

main "$@"
