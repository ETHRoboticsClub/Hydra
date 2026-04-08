#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/tmp/workspace"
PERSIST_ROOT="/persist"

echo "[cosmos-wrapper] Container bootstrap starting..."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

mkdir -p "${PERSIST_ROOT}/cosmos" "${PERSIST_ROOT}/hf_hub" "${PERSIST_ROOT}/xdg_cache" "${WORKDIR}"
cd "${WORKDIR}"

cat <<EOF
[cosmos-wrapper] Ready.
[cosmos-wrapper] Working directory (ephemeral): ${WORKDIR}
[cosmos-wrapper] Persistent volume: ${PERSIST_ROOT} (use ${PERSIST_ROOT}/cosmos for checkpoints; HF_HOME=${PERSIST_ROOT}/hf_hub)
[cosmos-wrapper] GPU pod will stay alive until the TrainJob is deleted.
EOF

exec sleep infinity
