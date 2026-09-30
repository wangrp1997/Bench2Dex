#!/bin/bash
set -euo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
ALI=https://mirrors.aliyun.com/pypi/simple/
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=180 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES

echo "### $(date -Is) waiting for isaacsim install to finish..."
for i in $(seq 1 720); do
  grep -q "=== exit=" "$L/02_isaacsim_install.log" 2>/dev/null && break
  sleep 30
done
if ! grep -q "=== exit=0" "$L/02_isaacsim_install.log"; then
  echo "!!! isaacsim install did not exit 0 — ABORT"; tail -40 "$L/02_isaacsim_install.log"; exit 1
fi
echo "### $(date -Is) isaacsim install OK, continuing"

echo "### STEP 5: torch cu128 $(date -Is)"
uv pip install --python "$ENVPY" --index-url "$ALI" \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128"

echo "### STEP 5b: verify torch (Pitfall 8) $(date -Is)"
"$ENVPY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(0))
a = torch.randn(2048, 2048, device="cuda"); print("matmul sum:", (a @ a).sum().item())
print("norm:", torch.randn(1000, device="cuda").norm().item())
PY

echo "### STEP 6: build tools $(date -Is)"
uv pip install --python "$ENVPY" --index-url "$ALI" "setuptools<81" wheel

echo "### STEP 7: Isaac Lab v2.3.2 $(date -Is)"
cd /home/wangrenpeng/bench2dex
if [ ! -d IsaacLab/.git ]; then
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab/source/isaaclab
uv pip install --python "$ENVPY" --index-url "$ALI" --no-build-isolation -e .

echo "### STEP 8: Bench2Dex requirements (pinocchio line dropped, Pitfall 6) $(date -Is)"
cd /home/wangrenpeng/bench2dex/Bench2Dex
grep -vE '^[[:space:]]*pinocchio' requirements.txt > /tmp/req.txt
uv pip install --python "$ENVPY" --index-url "$ALI" -r /tmp/req.txt ipython

echo "### SECTION 9 CHECKS $(date -Is)"
"$ENVPY" -c "import numpy; print('numpy', numpy.__version__)"
"$ENVPY" -c "import isaacsim; print('isaacsim ok')"
"$ENVPY" -c "import isaaclab; print('isaaclab', isaaclab.__version__)"
"$ENVPY" -c "import pinocchio; print('pinocchio', pinocchio.__version__, pinocchio.__file__)"
"$ENVPY" -c "import IPython, h5py, cv2, trimesh, dex_retargeting; print('ipython/h5py/cv2/trimesh/dex_retargeting OK')"
echo "### ENV DONE $(date -Is)"
