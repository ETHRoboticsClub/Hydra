#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f volumeclaims.yaml
kubectl apply -f trainjob.yaml
