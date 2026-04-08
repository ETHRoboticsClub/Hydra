#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="robot-learning"
TRAINJOB_NAME="cosmos-wrapper-interactive"
CONFIGMAP_NAME="cosmos-wrapper-files"

TRAINJOB_MANIFEST="${1:-trainjob.yaml}"

cd "${SCRIPT_DIR}"

if [[ "${TRAINJOB_MANIFEST}" != /* ]]; then
  TRAINJOB_MANIFEST="${SCRIPT_DIR}/${TRAINJOB_MANIFEST}"
fi

if [[ ! -f "${TRAINJOB_MANIFEST}" ]]; then
  echo "TrainJob manifest not found: ${TRAINJOB_MANIFEST}" >&2
  echo "Usage: $0 [trainjob.yaml]" >&2
  exit 1
fi

pod_name() {
  kubectl -n "${NAMESPACE}" get pods \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep '^cosmos-wrapper-interactive' \
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
  echo "Timed out waiting for the cosmos-wrapper-interactive pod to appear."
  exit 1
fi

kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD_NAME}" --timeout=15m

# Ephemeral shell cwd; put large / long-lived artifacts under /persist (PVC).
kubectl -n "${NAMESPACE}" exec -it "${POD_NAME}" -- bash -lc '
  set -euo pipefail
  PERSIST_ROOT=/persist
  mkdir -p "${PERSIST_ROOT}/cosmos" "${PERSIST_ROOT}/hf_hub" "${PERSIST_ROOT}/xdg_cache" /tmp/workspace
  export PERSIST_ROOT HF_HOME="${PERSIST_ROOT}/hf_hub" XDG_CACHE_HOME="${PERSIST_ROOT}/xdg_cache"
  cd /tmp/workspace
  echo "[cosmos-wrapper] Ephemeral cwd: /tmp/workspace"
  echo "[cosmos-wrapper] Persistent Cosmos & caches: ${PERSIST_ROOT}/cosmos (+ HF_HOME, XDG_CACHE_HOME under ${PERSIST_ROOT})"
  exec bash
'
