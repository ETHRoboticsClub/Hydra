#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="robot-learning"
TRAINJOB_NAME="cosmos-libero-interactive"
CONFIGMAP_NAME="cosmos-libero-files"

cd "${SCRIPT_DIR}"

# Nodepool map (from cluster Karpenter config):
#   gpus / node-tier: gpus → g6.xlarge        (1x L4,   16GB  RAM,  4 vCPU)
#   gpum / node-tier: gpum → g6e.xlarge        (1x L40S, 32GB  RAM,  4 vCPU)
#                          → g6e.2xlarge       (1x L40S, 64GB  RAM,  8 vCPU)
#   gpul / node-tier: gpul → g6e.12xlarge      (4x L40S, 384GB RAM, 48 vCPU)
#   h100 / node-tier: h100 → p5.4xlarge        (1x H100, 192GB RAM, 16 vCPU)
INSTANCE_TYPE=""
NODEPOOL=""
WANDB_KEY=""
FULLRUN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --nodepool)      NODEPOOL="$2";       shift 2 ;;
    --wandb-key)     WANDB_KEY="$2";      shift 2 ;;
    --fullrun)       FULLRUN=true;        shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Auto-infer nodepool from instance type
if [ -z "${NODEPOOL}" ] && [ -n "${INSTANCE_TYPE}" ]; then
  case "${INSTANCE_TYPE}" in
    g6e.xlarge|g6e.2xlarge) NODEPOOL="gpum" ;;
    g6e.12xlarge)            NODEPOOL="gpul" ;;
    p5.4xlarge)              NODEPOOL="h100" ;;
    *)                       NODEPOOL="gpus" ;;
  esac
fi

# Max resources per instance type (leaving ~10% for system overhead)
GPU_COUNT=1; CPU="3";  MEM="12Gi"   # defaults (g6.xlarge)
case "${INSTANCE_TYPE}" in
  g6e.xlarge)   GPU_COUNT=1; CPU="3";  MEM="26Gi"  ;;
  g6e.2xlarge)  GPU_COUNT=1; CPU="7";  MEM="54Gi"  ;;
  g6e.12xlarge) GPU_COUNT=4; CPU="44"; MEM="340Gi" ;;
  p5.4xlarge)   GPU_COUNT=1; CPU="14"; MEM="160Gi" ;;
esac

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
    echo "Using instance type: ${INSTANCE_TYPE} (nodepool: ${NODEPOOL}, GPUs: ${GPU_COUNT}, CPU: ${CPU}, RAM: ${MEM})"
    TRAINJOB_MANIFEST="$(echo "${TRAINJOB_MANIFEST}" | \
      sed "s/karpenter.sh\/nodepool: gpus/karpenter.sh\/nodepool: ${NODEPOOL}/" | \
      sed "s/node-tier: gpus/node-tier: ${NODEPOOL}\n          node.kubernetes.io\/instance-type: ${INSTANCE_TYPE}/" | \
      sed "s/nvidia\.com\/gpu: \"[0-9]*\"/nvidia.com\/gpu: \"${GPU_COUNT}\"/g" | \
      sed "s/cpu: \"[^\"]*\"/cpu: \"${CPU}\"/g" | \
      sed "s/memory: \"[^\"]*\"/memory: \"${MEM}\"/g")"
  fi
  # Inject env vars into the pod command
  ENV_INJECT="export NPROC=${GPU_COUNT}"
  [ "${FULLRUN}" = "true" ] && ENV_INJECT="${ENV_INJECT}; export FULLRUN=1"
  [ -n "${WANDB_KEY}" ]     && ENV_INJECT="${ENV_INJECT}; export WANDB_API_KEY=${WANDB_KEY}"
  TRAINJOB_MANIFEST="$(echo "${TRAINJOB_MANIFEST}" | \
    sed "s|true  # env-inject-placeholder|${ENV_INJECT}|")"
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

if [ "${FULLRUN}" = "true" ]; then
  echo "Full run started. Logs (also visible in k9s):"
  echo "  kubectl -n ${NAMESPACE} logs -f ${POD_NAME}"
  kubectl -n "${NAMESPACE}" logs -f "${POD_NAME}"
else
  EXEC_CMD='cd /data/cosmos-predict2 2>/dev/null || cd /data && exec bash'
  if [ -n "${WANDB_KEY}" ]; then
    EXEC_CMD="export WANDB_API_KEY=${WANDB_KEY}; ${EXEC_CMD}"
  fi
  kubectl -n "${NAMESPACE}" exec -it "${POD_NAME}" -- bash -lc "${EXEC_CMD}"
fi
