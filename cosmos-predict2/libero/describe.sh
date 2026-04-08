#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="robot-learning"

kubectl -n "${NAMESPACE}" get trainjob cosmos-libero-interactive
echo ""
kubectl -n "${NAMESPACE}" get pods | grep cosmos-libero || true
