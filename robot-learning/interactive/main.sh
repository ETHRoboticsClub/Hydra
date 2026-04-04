#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/tmp/workspace"

echo "[interactive] Container bootstrap starting..."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

cat <<EOF
[interactive] Ready.
[interactive] Working directory: ${WORKDIR}
[interactive] GPU pod will stay alive until the TrainJob is deleted.
EOF

exec sleep infinity
