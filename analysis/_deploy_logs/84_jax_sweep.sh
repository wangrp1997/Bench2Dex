#!/bin/bash
# Find a jax version that BOTH has sm_120 (Blackwell) kernels AND still exposes
# the old jax.experimental.layout.DeviceLocalLayout API that
# orbax-checkpoint==0.11.1 (pinned by openpi) imports.
set -uo pipefail
P5=/home/wangrenpeng/miniconda3/envs/pi05/bin/python
ALI=https://mirrors.aliyun.com/pypi/simple/
export http_proxy=http://host.docker.internal:1081 https_proxy=http://host.docker.internal:1081
export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,mirrors.aliyun.com,.aliyun.com"
export NO_PROXY=$no_proxy
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=300

for V in 0.9.0 0.8.0 0.7.2 0.7.0 0.6.2 0.6.0 0.5.3; do
  echo "=========== trying jax $V"
  uv pip install --python "$P5" --index-url "$ALI" -U "jax[cuda12]==$V" >/tmp/jaxinst.log 2>&1
  if ! grep -q "Installed\|Audited\|checked" /tmp/jaxinst.log && ! "$P5" -c "import jax" 2>/dev/null; then
      echo "   install failed:"; tail -3 /tmp/jaxinst.log; continue
  fi
  "$P5" - <<'PY'
import jax
try:
    from jax.experimental.layout import DeviceLocalLayout
    api = "API-OK"
except Exception:
    api = "API-MISSING"
try:
    import jax.numpy as jnp
    x = jnp.ones((512, 512)); y = (x @ x).sum(); y.block_until_ready()
    gpu = f"GPU-OK({float(y):.0f})"
except Exception as e:
    gpu = "GPU-FAIL:" + type(e).__name__
print(f"RESULT jax={jax.__version__} {api} {gpu}")
PY
done
echo "### SWEEP DONE $(date -Is)"
