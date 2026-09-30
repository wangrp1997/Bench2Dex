#!/bin/bash
# Rebuild the GR00T env with uv (the official setup_env.sh uses plain pip, which
# was measured at ~0.4 MB/s here; uv with 32 concurrent downloads is far faster).
set -uo pipefail
REPO=/home/wangrenpeng/bench2dex/Bench2Dex
GPP=/home/wangrenpeng/miniconda3/envs/GR00T_n15/bin/python
ALI=https://mirrors.aliyun.com/pypi/simple/
CUF=https://mirrors.aliyun.com/pytorch-wheels/cu128/
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=300

[ -x "$GPP" ] || { echo "!!! GR00T_n15 env missing"; exit 1; }
echo "### python: $("$GPP" --version)"

echo "### STEP A: GR00T package [base] + modelscope via uv $(date -Is)"
cd "$REPO" || exit 1
uv pip install --python "$GPP" --index-url "$ALI" -e "policy/GR00T_n15[base]" modelscope 2>&1 | tail -8

echo "### STEP B: torch -> cu128 $(date -Is)"
uv pip install --python "$GPP" --index-url "$ALI" --find-links "$CUF" \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" 2>&1 | tail -4

echo "### STEP C: verify $(date -Is)"
"$GPP" -c "
import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))
a=torch.randn(1024,1024,device='cuda'); print('cuda matmul:', (a@a).sum().item())"
echo "### STEP D: GR00T integration import $(date -Is)"
cd "$REPO" && "$GPP" -c "
import sys; sys.path.insert(0,'.')
try:
    import policy.GR00T_n15.deploy_policy
    print('GR00T deploy_policy import OK')
except Exception as e:
    print('GR00T import FAILED:', type(e).__name__, str(e)[:300])" 2>&1 | tail -4
echo "### GR00T DONE $(date -Is)"
