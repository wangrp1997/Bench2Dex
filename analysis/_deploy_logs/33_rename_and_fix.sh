#!/bin/bash
# 1) finish the asset move (content already checksum-verified identical)
# 2) rename ~/Projects -> ~/bench2dex so both sides use the same top-level name
# 3) repair everything that hard-coded the old path
set -uo pipefail
OLD=/home/wangrenpeng/bench2dex
NEW=/home/wangrenpeng/bench2dex
DST=/mnt/public/datasets/bench2dex/dex2bench_dataset

echo "### 1) finish assets move $(date -Is)"
rm -rf "$OLD/dex2bench_dataset"
ln -s "$DST" "$OLD/dex2bench_dataset"
rm -f "$OLD/probe_w.txt"
ls -ld "$OLD"/{dex2bench_dataset,teleopdata,policy_ckpt}

echo "### 2) rename $OLD -> $NEW $(date -Is)"
if [ ! -e "$NEW" ]; then mv "$OLD" "$NEW"; else echo "!!! $NEW already exists, aborting rename"; exit 1; fi
ls -la "$NEW"

echo "### 3a) conda activate.d libGLU path $(date -Is)"
cat > /home/wangrenpeng/miniconda3/envs/bench2dex/etc/conda/activate.d/bench2dex_env.sh <<'EOF'
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
# libGLU.so.1 (needed by Isaac Sim's iray/RTX renderer) was fetched with
# `apt-get download libglu1-mesa` + `dpkg-deb -x` because there is no root here.
export LD_LIBRARY_PATH=/home/wangrenpeng/bench2dex/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
EOF
cat /home/wangrenpeng/miniconda3/envs/bench2dex/etc/conda/activate.d/bench2dex_env.sh

echo "### 3b) rewrite hard-coded paths in helper scripts $(date -Is)"
grep -rl "/home/wangrenpeng/bench2dex" "$NEW/_deploy_logs" 2>/dev/null | while read -r f; do
    sed -i 's|/home/wangrenpeng/bench2dex|/home/wangrenpeng/bench2dex|g' "$f"; echo "    patched $f"
done

echo "### 3c) re-install IsaacLab editable (its .pth pointed at the old path) $(date -Is)"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
cd "$NEW/IsaacLab/source/isaaclab" && \
  uv pip install --python /home/wangrenpeng/miniconda3/envs/bench2dex/bin/python \
     --index-url https://mirrors.aliyun.com/pypi/simple/ --no-build-isolation -e . 2>&1 | tail -5

echo "### done $(date -Is)"
