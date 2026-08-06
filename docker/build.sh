#!/usr/bin/env bash
# Stages the lerobot fork (which lives outside this repo, at LEROBOT_FORK_SRC) into
# docker/_lerobot_fork_src/ so the Docker build context (re_vla/) is self-contained, then
# builds the libero_smolvla image.
#
# Why staging instead of a plain COPY from the host path: `docker build` can only COPY
# files inside its build context, and the context here is re_vla/ - the fork checkout at
# /home/eunju/research/lerobot is a sibling directory outside it. This install of Docker
# doesn't have the buildx plugin (no `--build-context` support for extra named build
# contexts), so staging a local copy is the simplest context-agnostic way to bridge that.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

LEROBOT_FORK_SRC="${LEROBOT_FORK_SRC:-/home/eunju/research/lerobot}"
STAGE_DIR="${SCRIPT_DIR}/_lerobot_fork_src"
IMAGE_TAG="${IMAGE_TAG:-libero_smolvla:latest}"

if [ ! -d "${LEROBOT_FORK_SRC}/src/lerobot" ]; then
    echo "error: ${LEROBOT_FORK_SRC} doesn't look like the lerobot fork (no src/lerobot/)." >&2
    echo "Set LEROBOT_FORK_SRC=/path/to/lerobot to override." >&2
    exit 1
fi

echo "Staging ${LEROBOT_FORK_SRC} -> ${STAGE_DIR} ..."
mkdir -p "${STAGE_DIR}"
rsync -a --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='docs/' \
    --exclude='tests/' \
    --exclude='docker/' \
    --exclude='outputs/' \
    --exclude='annotation/' \
    "${LEROBOT_FORK_SRC}/" "${STAGE_DIR}/"

echo "Building ${IMAGE_TAG} (context: ${REPO_ROOT}) ..."
docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"

echo "Done: ${IMAGE_TAG}"
