#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
VOLUMECLAIMS_MANIFEST="${VOLUMECLAIMS_MANIFEST:-${REPO_ROOT}/launch.d/volumeclaims.yaml}"
INSTANCE_TYPES_MANIFEST="${INSTANCE_TYPES_MANIFEST:-${SCRIPT_DIR}/instancetypes.yaml}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./lib/update_volumes.sh [--dry-run]

Applies the PersistentVolumeClaims and instance types ConfigMap used by ./launch.

Environment:
  VOLUMECLAIMS_MANIFEST   Manifest to apply (default: launch.d/volumeclaims.yaml)
  INSTANCE_TYPES_MANIFEST Instance types ConfigMap (default: lib/instancetypes.yaml)
  DRY_RUN=1 or --dry-run  Print the server-side dry-run result only
EOF
}

log() {
  printf '[update_volumes] %s\n' "$*" >&2
}

die() {
  log "$*"
  exit 1
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
  fi

  if [[ "$#" -ne 0 ]]; then
    usage
    exit 1
  fi

  [[ -f "${VOLUMECLAIMS_MANIFEST}" ]] || die "Missing volume claims manifest: ${VOLUMECLAIMS_MANIFEST}"
  [[ -f "${INSTANCE_TYPES_MANIFEST}" ]] || die "Missing instance types manifest: ${INSTANCE_TYPES_MANIFEST}"

  if [[ "${DRY_RUN}" == 1 ]]; then
    log "Dry run applying ${VOLUMECLAIMS_MANIFEST}"
    kubectl apply --dry-run=server -f "${VOLUMECLAIMS_MANIFEST}"
    log "Dry run applying ${INSTANCE_TYPES_MANIFEST}"
    kubectl apply --dry-run=server -f "${INSTANCE_TYPES_MANIFEST}"
    exit 0
  fi

  log "Applying ${VOLUMECLAIMS_MANIFEST}"
  kubectl apply -f "${VOLUMECLAIMS_MANIFEST}"
  log "Applying ${INSTANCE_TYPES_MANIFEST}"
  kubectl apply -f "${INSTANCE_TYPES_MANIFEST}"
}

main "$@"
