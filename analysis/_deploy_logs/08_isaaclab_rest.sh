#!/bin/bash
set -euo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
ALI=https://mirrors.aliyun.com/pypi/simple/
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=180 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH

echo "### STEP 7 (workaround): Isaac Lab v2.3.2 via codeload tarball, git clone stalls on github.com $(date -Is)"
cd /home/wangrenpeng/bench2dex
curl -fSL --retry 6 --retry-delay 3 --retry-all-errors -C - \
  -o /tmp/isaaclab-v2.3.2.tar.gz \
  https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/refs/tags/v2.3.2
ls -l /tmp/isaaclab-v2.3.2.tar.gz
tar tzf /tmp/isaaclab-v2.3.2.tar.gz > /dev/null
rm -rf /tmp/IsaacLab-2.3.2
tar xzf /tmp/isaaclab-v2.3.2.tar.gz -C /tmp
rm -rf /home/wangrenpeng/bench2dex/IsaacLab
mv /tmp/IsaacLab-2.3.2 /home/wangrenpeng/bench2dex/IsaacLab
echo "### extracted:"; ls /home/wangrenpeng/bench2dex/IsaacLab | head -8

echo "### STEP 7b: editable install of isaaclab $(date -Is)"
cd /home/wangrenpeng/bench2dex/IsaacLab/source/isaaclab
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
"$ENVPY" -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
echo "### ENV DONE $(date -Is)"
