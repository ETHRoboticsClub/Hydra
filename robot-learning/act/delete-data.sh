#!/usr/bin/env bash
set -euo pipefail

kubectl delete --ignore-not-found -f volumeclaims.yaml
