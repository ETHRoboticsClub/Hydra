#!/usr/bin/env bash
set -Eeuo pipefail

NAMESPACE="${NAMESPACE:-robot-learning}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-training-sa}"
UPLOAD_IMAGE="${UPLOAD_IMAGE:-public.ecr.aws/aws-cli/aws-cli:2}"
CHECKPOINT_PVC_NAME="${CHECKPOINT_PVC_NAME:-act-checkpoints}"
UPLOAD_REQUEST_CPU="${UPLOAD_REQUEST_CPU:-250m}"
UPLOAD_REQUEST_MEMORY="${UPLOAD_REQUEST_MEMORY:-512Mi}"
UPLOAD_LIMIT_CPU="${UPLOAD_LIMIT_CPU:-1}"
UPLOAD_LIMIT_MEMORY="${UPLOAD_LIMIT_MEMORY:-2Gi}"
WORKER_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/upload-checkpoint-to-s3-worker.sh"
CHECKPOINT_NAME="${1:-}"
S3_TARGET_DIR="${2:-}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
POD_NAME="act-checkpoint-upload-${RUN_ID}"
CONFIGMAP_NAME="${POD_NAME}-script"
SELECTED_NODE_NAME=""
KEEP_FAILED_POD="${KEEP_FAILED_POD:-1}"
KEEP_SUCCEEDED_POD="${KEEP_SUCCEEDED_POD:-1}"
PENDING_TIMEOUT_SECONDS="${PENDING_TIMEOUT_SECONDS:-180}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./robot-learning/upload-checkpoint-to-s3.sh <checkpoint-name> <s3-dir>

Examples:
  ./robot-learning/upload-checkpoint-to-s3.sh last act/run-001
  ./robot-learning/upload-checkpoint-to-s3.sh 0900 act/run-001

This command runs a short-lived Kubernetes pod that mounts the checkpoint PVC,
uploads the requested checkpoint to S3, streams progress logs, and reports the
final result locally.
EOF
}

log() {
  printf '[upload-checkpoint-to-s3] %s\n' "$*" >&2
}

die() {
  log "$*"
  exit 1
}

cleanup() {
  local exit_code=$?

  if [ "${exit_code}" -eq 0 ] && [ "${KEEP_SUCCEEDED_POD}" != "1" ]; then
    kubectl -n "${NAMESPACE}" delete pod "${POD_NAME}" --ignore-not-found >/dev/null 2>&1 || true
    kubectl -n "${NAMESPACE}" delete configmap "${CONFIGMAP_NAME}" --ignore-not-found >/dev/null 2>&1 || true
  elif [ "${exit_code}" -ne 0 ] && [ "${KEEP_FAILED_POD}" != "1" ]; then
    kubectl -n "${NAMESPACE}" delete pod "${POD_NAME}" --ignore-not-found >/dev/null 2>&1 || true
    kubectl -n "${NAMESPACE}" delete configmap "${CONFIGMAP_NAME}" --ignore-not-found >/dev/null 2>&1 || true
  else
    log "Keeping pod ${POD_NAME} for inspection."
    log "Inspect with: kubectl -n ${NAMESPACE} describe pod ${POD_NAME}"
    log "Logs with: kubectl -n ${NAMESPACE} logs ${POD_NAME} -c uploader"
    log "Delete with: kubectl -n ${NAMESPACE} delete pod ${POD_NAME} configmap/${CONFIGMAP_NAME}"
  fi

  return "${exit_code}"
}

require_args() {
  if [ -z "${CHECKPOINT_NAME}" ] || [ -z "${S3_TARGET_DIR}" ]; then
    usage
    exit 1
  fi

  if [[ "${S3_TARGET_DIR}" == s3://* ]]; then
    die "Pass an S3 directory like act/run-001, not a full s3:// path."
  fi
}

require_tools() {
  command -v kubectl >/dev/null 2>&1 || die "kubectl is required."
  [ -f "${WORKER_SCRIPT_PATH}" ] || die "Worker script not found at ${WORKER_SCRIPT_PATH}."
}

discover_selected_node() {
  SELECTED_NODE_NAME="$(
    kubectl -n "${NAMESPACE}" get pvc "${CHECKPOINT_PVC_NAME}" \
      -o jsonpath='{.metadata.annotations.volume\.kubernetes\.io/selected-node}' 2>/dev/null || true
  )"

  if [ -n "${SELECTED_NODE_NAME}" ]; then
    log "PVC ${CHECKPOINT_PVC_NAME} selected node: ${SELECTED_NODE_NAME}"
  else
    log "PVC ${CHECKPOINT_PVC_NAME} has no selected-node annotation. Using default scheduling."
  fi
}

create_configmap() {
  kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP_NAME}" \
    --from-file=upload-checkpoint-to-s3-worker.sh="${WORKER_SCRIPT_PATH}"
}

create_pod() {
  local pod_name="$1"
  local manifest
  manifest="$(mktemp)"

  local node_name_block=""
  if [ -n "${SELECTED_NODE_NAME}" ]; then
    node_name_block="  nodeName: ${SELECTED_NODE_NAME}"
  fi

  cat > "${manifest}" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  serviceAccountName: ${SERVICE_ACCOUNT_NAME}
${node_name_block}
  nodeSelector:
    karpenter.sh/nodepool: gpus
    node-tier: gpus
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  volumes:
    - name: checkpoints
      persistentVolumeClaim:
        claimName: ${CHECKPOINT_PVC_NAME}
    - name: worker-script
      configMap:
        name: ${CONFIGMAP_NAME}
        defaultMode: 0755
  containers:
    - name: uploader
      image: ${UPLOAD_IMAGE}
      resources:
        requests:
          cpu: ${UPLOAD_REQUEST_CPU}
          memory: ${UPLOAD_REQUEST_MEMORY}
        limits:
          cpu: ${UPLOAD_LIMIT_CPU}
          memory: ${UPLOAD_LIMIT_MEMORY}
      command:
        - /bin/sh
        - /job-files/upload-checkpoint-to-s3-worker.sh
        - ${CHECKPOINT_NAME}
        - ${S3_TARGET_DIR}
      volumeMounts:
        - name: checkpoints
          mountPath: /checkpoints
        - name: worker-script
          mountPath: /job-files
          readOnly: true
EOF

  kubectl create --validate=false -f "${manifest}"
  rm -f "${manifest}"
}

print_logs() {
  local pod_name="$1"
  kubectl -n "${NAMESPACE}" logs "${pod_name}" -c uploader --ignore-errors=true 2>&1 || true
}

stream_logs_background() {
  local pod_name="$1"
  kubectl -n "${NAMESPACE}" logs -f "${pod_name}" -c uploader --ignore-errors=true --pod-running-timeout=5s &
  echo $!
}

wait_for_pod_completion() {
  local pod_name="$1"
  local start_time
  start_time="$(date +%s)"
  local last_pending_reason=""
  local saw_running="0"
  local log_pid=""
  while true; do
    local phase=""
    phase="$(kubectl -n "${NAMESPACE}" get pod "${pod_name}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"

    case "${phase}" in
      Succeeded|Failed)
        if [ -n "${log_pid}" ]; then
          wait "${log_pid}" 2>/dev/null || true
        fi
        log "Final pod logs:"
        print_logs "${pod_name}" >&2
        return 0
        ;;
      Pending)
        log "Pod ${pod_name} is still pending."
        kubectl -n "${NAMESPACE}" get pod "${pod_name}" -o jsonpath='{range .status.conditions[*]}{.type}={.status}{" "}{end}' >&2 || true
        printf '\n' >&2
        local scheduler_message=""
        scheduler_message="$(
          kubectl -n "${NAMESPACE}" get events \
            --field-selector "involvedObject.kind=Pod,involvedObject.name=${pod_name}" \
            --sort-by=.lastTimestamp \
            -o jsonpath='{range .items[*]}{.reason}{": "}{.message}{"\n"}{end}' 2>/dev/null | tail -n 3
        )"
        if [ -n "${scheduler_message}" ] && [ "${scheduler_message}" != "${last_pending_reason}" ]; then
          log "Recent pod events:"
          printf '%s\n' "${scheduler_message}" >&2
          last_pending_reason="${scheduler_message}"
        fi
        local now
        now="$(date +%s)"
        if [ $((now - start_time)) -ge "${PENDING_TIMEOUT_SECONDS}" ]; then
          log "Pod ${pod_name} stayed pending for ${PENDING_TIMEOUT_SECONDS}s."
          kubectl -n "${NAMESPACE}" describe pod "${pod_name}" >&2 || true
          return 1
        fi
        ;;
      Running)
        if [ "${saw_running}" != "1" ]; then
          saw_running="1"
          log "Pod ${pod_name} is running. Attaching to logs."
          log_pid="$(stream_logs_background "${pod_name}")"
        fi
        ;;
      "")
        if [ "${saw_running}" = "1" ]; then
          if [ -n "${log_pid}" ]; then
            wait "${log_pid}" 2>/dev/null || true
          fi
          log "Pod ${pod_name} disappeared before a terminal status was observed."
          return 1
        fi
        log "Waiting for pod ${pod_name} to appear."
        ;;
      *)
        log "Pod ${pod_name} phase: ${phase}"
        ;;
    esac

    sleep 5
  done
}

report_result() {
  local pod_name="$1"
  local success_message="$2"
  local phase
  phase="$(kubectl -n "${NAMESPACE}" get pod "${pod_name}" -o jsonpath='{.status.phase}')"

  if [ "${phase}" = "Succeeded" ]; then
    log "${success_message}"
    return 0
  fi

  local exit_code=""
  exit_code="$(kubectl -n "${NAMESPACE}" get pod "${pod_name}" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)"
  log "Pod ${pod_name} failed. Phase=${phase}${exit_code:+, exitCode=${exit_code}}"
  kubectl -n "${NAMESPACE}" describe pod "${pod_name}" >&2 || true
  return 1
}

main() {
  require_args
  require_tools
  trap cleanup EXIT

  discover_selected_node
  log "Creating upload pod ${POD_NAME} in namespace ${NAMESPACE}"
  log "Pod name: ${POD_NAME}"
  log "Checkpoint: ${CHECKPOINT_NAME}"
  log "Destination: s3://ethrc-ml-data-916780037007/${S3_TARGET_DIR}"

  create_configmap
  create_pod "${POD_NAME}"
  wait_for_pod_completion "${POD_NAME}"
  report_result "${POD_NAME}" "Upload pod succeeded."
}

main "$@"
