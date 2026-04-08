#!/usr/bin/env bash
set -euo pipefail

kubectl -n robot-learning delete --ignore-not-found trainjob cosmos-wrapper-interactive
kubectl -n robot-learning delete --ignore-not-found configmap cosmos-wrapper-files
