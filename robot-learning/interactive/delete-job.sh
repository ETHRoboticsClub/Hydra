#!/usr/bin/env bash
set -euo pipefail

kubectl -n robot-learning delete --ignore-not-found trainjob interactive-gpu
kubectl -n robot-learning delete --ignore-not-found configmap interactive-gpu-files
