#!/usr/bin/env bash
set -euo pipefail

VENV_ROOT="/home/noir/.venvs/pinns_tf_wsl"
SITE_PACKAGES="$VENV_ROOT/lib/python3.12/site-packages"

source "$VENV_ROOT/bin/activate"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$@"
