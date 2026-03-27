#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="robot-learning"

kubectl -n "${NAMESPACE}" delete trainjob cosmos-libero-interactive --ignore-not-found
kubectl -n "${NAMESPACE}" delete configmap cosmos-libero-files --ignore-not-found
