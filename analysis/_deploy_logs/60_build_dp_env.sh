#!/bin/bash
# Build the DP policy environment (conda env `dp`) — official path, but with
# torch cu128 instead of env.sh's torch 2.0.0+cu118, because this GPU is
# Blackwell sm_120 and cu118 has no kernels for it (DEPLOYMENT_NOTES Pitfall 8).
set -uo pipefail
ALI=https://mirrors.aliyun.com/pypi/simple/
CUF=https://mirrors.aliyun.com/pytorch-wheels/cu128/
PROXY=http://host.docker.internal:1081
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=180
DPPY=/home/wangrenpeng/miniconda3/envs/dp/bin/python

echo "### STEP 1: conda create -n dp python=3.10  $(date -Is)"
env http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY \
  /home/wangrenpeng/miniconda3/bin/conda create -n dp python=3.10 -y 2>&1 | tail -5
[ -x "$DPPY" ] || { echo "!!! conda env dp not created"; exit 1; }
"$DPPY" --version

echo "### STEP 2: torch cu128 (NOT cu118) $(date -Is)"
uv pip install --python "$DPPY" --index-url "$ALI" --find-links "$CUF" \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" 2>&1 | tail -4

echo "### STEP 3: DP deps (from policy/DP/env.sh list) $(date -Is)"
cd /home/wangrenpeng/bench2dex/Bench2Dex/policy/DP
uv pip install --python "$DPPY" --index-url "$ALI" -e . \
  numpy==1.26.4 numba==0.59.1 scipy==1.11.4 h5py==3.10.0 pillow \
  opencv-python==4.7.0.72 pyyaml tqdm dill einops hydra-core==1.2.0 \
  diffusers==0.20.2 huggingface-hub==0.19.4 pandas==2.0.3 zarr==2.16.1 \
  numcodecs==0.11.0 wandb 2>&1 | tail -6

echo "### STEP 4: verify $(date -Is)"
"$DPPY" - <<'PY'
import torch, numpy, hydra, diffusers, zarr, numba
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability(0))
a = torch.randn(1024, 1024, device="cuda"); print("cuda matmul:", (a @ a).sum().item())
print("numpy", numpy.__version__, "| hydra", hydra.__version__, "| diffusers", diffusers.__version__, "| zarr", zarr.__version__, "| numba", numba.__version__)
import sys; sys.path.insert(0, "/home/wangrenpeng/bench2dex/Bench2Dex/policy/DP")
import dp_model; print("dp_model import OK")
PY
echo "### DP ENV DONE $(date -Is)"
