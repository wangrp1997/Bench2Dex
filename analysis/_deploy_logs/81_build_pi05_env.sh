#!/bin/bash
# Build the pi0.5 (openpi) policy environment.
#
# The integration (policy/pi05/pi_model.py) uses openpi's JAX path
# (openpi.policies.policy_config), and pyproject pins jax[cuda12]==0.5.0.
# JAX that old ships CUDA 12.4/12.6 kernels with no sm_120 (Blackwell) support,
# so we install the repo deps first and then upgrade jax to a build that does.
set -uo pipefail
REPO=/home/wangrenpeng/bench2dex/Bench2Dex
L=/home/wangrenpeng/bench2dex/_deploy_logs
P5=/home/wangrenpeng/miniconda3/envs/pi05/bin/python
PROXY=http://host.docker.internal:1081
ALI=https://mirrors.aliyun.com/pypi/simple/

# domsetic mirrors must stay off the proxy; the jax cuda plugin wheels come from
# storage.googleapis.com and DO need it.
export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY
export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16,mirrors.aliyun.com,.aliyun.com"
export NO_PROXY=$no_proxy
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=300

echo "### STEP 1: conda create -n pi05 python=3.11 $(date -Is)"
env http_proxy=$PROXY https_proxy=$PROXY \
  /home/wangrenpeng/miniconda3/bin/conda create -n pi05 python=3.11 -y 2>&1 | tail -4
[ -x "$P5" ] || { echo "!!! pi05 env not created"; exit 1; }
"$P5" --version

echo "### STEP 2: openpi deps (this pulls jax 0.5.0 first) $(date -Is)"
cd "$REPO" || exit 1
uv pip install --python "$P5" --index-url "$ALI" -e policy/pi05 2>&1 | tail -8

echo "### STEP 3: upgrade jax to a build with sm_120 kernels $(date -Is)"
uv pip install --python "$P5" --index-url "$ALI" -U "jax[cuda12]" 2>&1 | tail -6

echo "### STEP 4: verify jax actually runs on the Blackwell GPU $(date -Is)"
"$P5" - <<'PY'
import jax, jax.numpy as jnp
print("jax", jax.__version__)
try:
    print("devices:", jax.devices())
    x = jnp.ones((1024, 1024)); y = (x @ x).sum()
    y.block_until_ready()
    print("jax GPU matmul sum:", float(y))
    print("JAX SM120 OK")
except Exception as e:
    print("JAX FAILED:", type(e).__name__, str(e)[:400])
PY

echo "### STEP 5: can the openpi/pi05 integration import? $(date -Is)"
"$P5" -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'policy/pi05')
try:
    from openpi.policies import policy_config
    from openpi.training import config
    print('openpi import OK')
    import pi_model
    print('pi_model import OK')
except Exception as e:
    print('pi05 import FAILED:', type(e).__name__, str(e)[:300])
" 2>&1 | tail -5
echo "### PI05 ENV DONE $(date -Is)"
