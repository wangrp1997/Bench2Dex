#!/bin/bash
# Build the GR00T N1.5 policy environment.
#
# Two deliberate deviations from policy/GR00T_n15/setup_env.sh:
#  1. PIP_INDEX_URL -> aliyun (the script defaults to Tsinghua; measured here:
#     aliyun 7.7 MB/s vs tuna 854 B/s for torch wheels).
#  2. The script pins torch==2.5.1 (cu124) via the [base] extra. cu124 has no
#     sm_120 kernels, so after the official install we upgrade to torch
#     cu128 -- same reason as DEPLOYMENT_NOTES Pitfall 8.
set -uo pipefail
REPO=/home/wangrenpeng/bench2dex/Bench2Dex
L=/home/wangrenpeng/bench2dex/_deploy_logs
GPP=/home/wangrenpeng/miniconda3/envs/GR00T_n15/bin/python
PROXY=http://host.docker.internal:1081

export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY
# keep the domestic mirrors off the proxy
export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16,mirrors.aliyun.com,.aliyun.com,pypi.tuna.tsinghua.edu.cn,.tuna.tsinghua.edu.cn"
export NO_PROXY=$no_proxy

cd "$REPO" || exit 1
echo "### STEP 1: official setup_env.sh --skip-model-download (model comes from the Bench2Dex ckpt) $(date -Is)"
bash policy/GR00T_n15/setup_env.sh --skip-model-download 2>&1 | tail -25

[ -x "$GPP" ] || { echo "!!! GR00T_n15 env python missing"; exit 1; }
echo "### STEP 2: python version"; "$GPP" --version

echo "### STEP 3: torch -> cu128 (official pin is 2.5.1/cu124, unusable on sm_120) $(date -Is)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
uv pip install --python "$GPP" --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" 2>&1 | tail -4

echo "### STEP 4: verify $(date -Is)"
"$GPP" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability(0))
a = torch.randn(1024, 1024, device="cuda"); print("cuda matmul:", (a @ a).sum().item())
PY
echo "### STEP 5: can the GR00T integration import? $(date -Is)"
cd "$REPO" || exit 1
"$GPP" -c "
import sys; sys.path.insert(0, '.')
try:
    import policy.GR00T_n15.deploy_policy as d
    print('GR00T deploy_policy import OK')
except Exception as e:
    print('GR00T import FAILED:', type(e).__name__, e)
" 2>&1 | tail -5
echo "### GR00T ENV DONE $(date -Is)"
