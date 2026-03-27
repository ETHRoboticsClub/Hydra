#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="robot-learning"
TRAINJOB_NAME="cosmos-libero-interactive"
CONFIGMAP_NAME="cosmos-libero-files"

cd "${SCRIPT_DIR}"

# Optional: --instance-type <type>  e.g. g6e.xlarge (L40S), p4de.24xlarge (A100 80GB)
INSTANCE_TYPE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

pod_name() {
  kubectl -n "${NAMESPACE}" get pods \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep '^cosmos-libero-interactive' \
    | tail -n 1
}

if ! kubectl -n "${NAMESPACE}" get trainjob "${TRAINJOB_NAME}" >/dev/null 2>&1; then
  kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP_NAME}" \
    --from-file=main.sh \
    --dry-run=client -o yaml \
    | kubectl apply -f -

  kubectl apply -f volumeclaims.yaml

  TRAINJOB_MANIFEST="$(cat trainjob.yaml)"
  if [ -n "${INSTANCE_TYPE}" ]; then
    echo "Using instance type: ${INSTANCE_TYPE}"
    TRAINJOB_MANIFEST="$(echo "${TRAINJOB_MANIFEST}" | \
      sed "s/node-tier: gpus/node-tier: gpus\n          node.kubernetes.io\/instance-type: ${INSTANCE_TYPE}/")"
  fi
  echo "${TRAINJOB_MANIFEST}" | kubectl apply -f -
fi

for _ in $(seq 1 60); do
  POD_NAME="$(pod_name || true)"
  if [ -n "${POD_NAME}" ]; then
    break
  fi
  sleep 2
done

if [ -z "${POD_NAME:-}" ]; then
  echo "Timed out waiting for the pod to appear."
  exit 1
fi

kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD_NAME}" --timeout=15m
kubectl -n "${NAMESPACE}" exec -it "${POD_NAME}" -- bash -lc \
  'cd /data/cosmos-predict2 2>/dev/null || cd /data && exec bash'
