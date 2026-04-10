#!/usr/bin/env bash
set -euxo pipefail

: "${TRAIN_GIT_URL:?TRAIN_GIT_URL is required}"
: "${TRAIN_GIT_REF:?TRAIN_GIT_REF is required}"
: "${TRAIN_SCRIPT_REL:?TRAIN_SCRIPT_REL is required}"
: "${TRAIN_CONFIG_REL:?TRAIN_CONFIG_REL is required}"

WORKDIR="${TRAIN_WORKDIR:-/tmp/train-workspace}"
DEPTH="${GIT_CLONE_DEPTH:-1}"

rm -rf "${WORKDIR}"
git clone --branch "${TRAIN_GIT_REF}" --single-branch --depth "${DEPTH}" "${TRAIN_GIT_URL}" "${WORKDIR}"
cd "${WORKDIR}"

export JOB_CONFIG="${WORKDIR}/${TRAIN_CONFIG_REL}"
export TRAIN_REPO_ROOT="${WORKDIR}"

if [[ -n "${TRAIN_DEVICE_CONFIG_REL:-}" ]]; then
  export DEVICE_CONFIG="${WORKDIR}/${TRAIN_DEVICE_CONFIG_REL}"
  if [[ ! -f "${DEVICE_CONFIG}" ]]; then
    printf '[pod-bootstrap] Device config not found: %s\n' "${DEVICE_CONFIG}" >&2
    exit 1
  fi
fi

if [[ ! -f "${JOB_CONFIG}" ]]; then
  printf '[pod-bootstrap] Train config not found: %s\n' "${JOB_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${WORKDIR}/${TRAIN_SCRIPT_REL}" ]]; then
  printf '[pod-bootstrap] Train script not found: %s\n' "${WORKDIR}/${TRAIN_SCRIPT_REL}" >&2
  exit 1
fi

exec bash "${WORKDIR}/${TRAIN_SCRIPT_REL}"
