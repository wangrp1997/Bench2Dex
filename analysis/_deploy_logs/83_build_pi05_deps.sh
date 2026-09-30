#!/bin/bash
# pi05 deps: the conda env now exists; install openpi + upgrade jax for sm_120.
set -uo pipefail
REPO=/home/wangrenpeng/bench2dex/Bench2Dex
P5=/home/wangrenpeng/miniconda3/envs/pi05/bin/python
ALI=https://mirrors.aliyun.com/pypi/simple/
PROXY=http://host.docker.internal:1081
export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY
export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16,mirrors.aliyun.com,.aliyun.com"
export NO_PROXY=$no_proxy
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=300

echo "### STEP 1: openpi deps (pulls jax 0.5.0 first) $(date -Is)"
cd "$REPO" || exit 1
uv pip install --python "$P5" --index-url "$ALI" -e policy/pi05 2>&1 | tail -10

echo "### STEP 2: upgrade jax for Blackwell $(date -Is)"
uv pip install --python "$P5" --index-url "$ALI" -U "jax[cuda12]" 2>&1 | tail -6

echo "### STEP 3: jax on sm_120 $(date -Is)"
"$P5" - <<'PY'
import jax, jax.numpy as jnp
print("jax", jax.__version__)
try:
    print("devices:", jax.devices())
    x = jnp.ones((1024,1024)); y = (x@x).sum(); y.block_until_ready()
    print("jax GPU matmul sum:", float(y)); print("JAX SM120 OK")
except Exception as e:
    print("JAX FAILED:", type(e).__name__, str(e)[:400])
PY

echo "### STEP 4: integrations import $(date -Is)"
cd "$REPO" && "$P5" -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'policy/pi05')
try:
    from openpi.policies import policy_config
    from openpi.training import config
    print('openpi import OK')
    import pi_model; print('pi_model import OK')
except Exception as e:
    print('pi05 import FAILED:', type(e).__name__, str(e)[:300])" 2>&1 | tail -5
echo "### PI05 DONE $(date -Is)"
