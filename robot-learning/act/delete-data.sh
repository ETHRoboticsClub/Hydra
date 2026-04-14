#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

kubectl delete --ignore-not-found -f "${REPO_ROOT}/launch.d/volumeclaims.yaml"
