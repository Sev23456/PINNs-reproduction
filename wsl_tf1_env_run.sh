#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"

exec /home/noir/miniforge3/bin/conda run --no-capture-output -n pinns_tf1 "$@"
