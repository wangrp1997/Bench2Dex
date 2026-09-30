#!/bin/bash
# Install flash-attn==2.8.2 into the GR00T_n15 env (declared in policy/GR00T_n15/pyproject.toml:93,
# missed by my uv-based rebuild). Compiles from sdist against torch 2.7.0+cu128 / CUDA 12.8.
set -uo pipefail
export PATH=/home/wangrenpeng/.local/bin:/usr/local/cuda/bin:$PATH
export MAX_JOBS=8
export NVCC_THREADS=4
export CUDA_HOME=/usr/local/cuda
echo "=== start $(date -Is) ==="
cd /tmp
uv pip install --python /home/wangrenpeng/miniconda3/envs/GR00T_n15/bin/python \
  "flash-attn==2.8.2" --no-build-isolation 2>&1 | tail -40
echo "=== exit=$? $(date -Is) ==="
