#!/usr/bin/env bash
set -euo pipefail

kubectl -n robot-learning describe trainjob cosmos-wrapper-interactive
