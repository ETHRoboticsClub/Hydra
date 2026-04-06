#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="robot-learning"
TRAINJOB_NAME="interactive-gpu"
CONFIGMAP_NAME="interactive-gpu-files"
CONFIG_INPUT="${1:-}"

if [[ -z "${CONFIG_INPUT}" ]]; then
  echo "Config name is required." >&2
  echo "Usage: $0 <config-name|config-name.yaml|path/to/config.yaml>" >&2
  exit 1
fi

if [[ "${CONFIG_INPUT}" != *.yaml ]]; then
  CONFIG_INPUT="${CONFIG_INPUT}.yaml"
fi

if [[ "${CONFIG_INPUT}" == /* || "${CONFIG_INPUT}" == */* ]]; then
  TRAINJOB_MANIFEST="${CONFIG_INPUT}"
else
  TRAINJOB_MANIFEST="${SCRIPT_DIR}/configs/${CONFIG_INPUT}"
fi

cd "${SCRIPT_DIR}"

if [[ "${TRAINJOB_MANIFEST}" != /* ]]; then
  TRAINJOB_MANIFEST="${SCRIPT_DIR}/${TRAINJOB_MANIFEST}"
fi

if [[ ! -f "${TRAINJOB_MANIFEST}" ]]; then
  echo "TrainJob manifest not found: ${TRAINJOB_MANIFEST}" >&2
  echo "Usage: $0 <config-name|config-name.yaml|path/to/config.yaml>" >&2
  exit 1
fi

pod_name() {
  kubectl -n "${NAMESPACE}" get pods \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep '^interactive-gpu' \
    | tail -n 1
}

if ! kubectl -n "${NAMESPACE}" get trainjob "${TRAINJOB_NAME}" >/dev/null 2>&1; then
  kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP_NAME}" \
    --from-file=main.sh \
    --dry-run=client \
    -o yaml \
    | kubectl apply -f -

  kubectl apply -f "${TRAINJOB_MANIFEST}"
fi

for _ in $(seq 1 60); do
  POD_NAME="$(pod_name || true)"
  if [ -n "${POD_NAME}" ]; then
    break
  fi
  sleep 2
done

if [ -z "${POD_NAME:-}" ]; then
  echo "Timed out waiting for the interactive-gpu pod to appear."
  exit 1
fi

kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD_NAME}" --timeout=15m
kubectl -n "${NAMESPACE}" exec -it "${POD_NAME}" -- bash -lc 'mkdir -p /tmp/workspace && cd /tmp/workspace && exec bash'
