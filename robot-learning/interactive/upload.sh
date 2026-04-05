#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="robot-learning"
POD_PREFIX="interactive-gpu"
DEFAULT_DEST="/tmp/workspace"

usage() {
  cat <<EOF
Usage: $0 LOCAL_PATH [REMOTE_PATH]

Copy a local file or directory into the current interactive pod.

Arguments:
  LOCAL_PATH   Local file or directory to copy.
  REMOTE_PATH  Optional destination path inside the pod.
               Defaults to ${DEFAULT_DEST}/<basename of LOCAL_PATH>
EOF
}

pod_name() {
  kubectl -n "${NAMESPACE}" get pods \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep "^${POD_PREFIX}" \
    | tail -n 1
}

remote_dir_target() {
  local remote_path="$1"

  if [[ "${remote_path}" == */ ]]; then
    printf '%s\n' "${remote_path%/}"
  else
    dirname "${remote_path}"
  fi
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage
  exit 1
fi

LOCAL_PATH="$1"

if [ ! -e "${LOCAL_PATH}" ]; then
  echo "Local path does not exist: ${LOCAL_PATH}" >&2
  exit 1
fi

POD_NAME="$(pod_name || true)"
if [ -z "${POD_NAME}" ]; then
  echo "No running ${POD_PREFIX} pod found in namespace ${NAMESPACE}." >&2
  echo "Start one first with ./robot-learning/interactive/start.sh" >&2
  exit 1
fi

BASENAME="$(basename "${LOCAL_PATH}")"
REMOTE_PATH="${2:-${DEFAULT_DEST}/${BASENAME}}"
REMOTE_DIR="$(remote_dir_target "${REMOTE_PATH}")"

if ! kubectl -n "${NAMESPACE}" exec "${POD_NAME}" -- test -d "${REMOTE_DIR}" >/dev/null 2>&1; then
  read -r -p "Remote directory ${REMOTE_DIR} does not exist on ${POD_NAME}. Create it? [y/N] " CREATE_REMOTE_DIR
  if [[ ! "${CREATE_REMOTE_DIR}" =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 1
  fi

  kubectl -n "${NAMESPACE}" exec "${POD_NAME}" -- mkdir -p "${REMOTE_DIR}"
fi

echo "Copying ${LOCAL_PATH} to ${POD_NAME}:${REMOTE_PATH}"
kubectl -n "${NAMESPACE}" cp "${LOCAL_PATH}" "${POD_NAME}:${REMOTE_PATH}"
echo "Upload complete."
