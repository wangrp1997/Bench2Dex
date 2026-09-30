#!/bin/bash
set -o pipefail
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=180
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
echo "=== start $(date -Is) ==="
echo "uv pip install isaacsim[all,extscache]==5.1.0  (index=aliyun, extra=pypi.nvidia.com, NO proxy)"
time uv pip install --python "$ENVPY" \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://pypi.nvidia.com \
  "isaacsim[all,extscache]==5.1.0"
echo "=== exit=$? $(date -Is) ==="
