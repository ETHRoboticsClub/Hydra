#!/usr/bin/env bash
set -euo pipefail

./delete-job.sh
kubectl apply -f volumeclaims.yaml
kubectl apply -f trainjob.yaml
