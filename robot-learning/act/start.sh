#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

"${SCRIPT_DIR}/delete-job.sh"
"${REPO_ROOT}/lib/update_volumes.sh"
kubectl apply -f "${SCRIPT_DIR}/trainjob.yaml"
