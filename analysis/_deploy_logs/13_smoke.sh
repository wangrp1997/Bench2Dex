#!/bin/bash
# Smoke test attempt 5: + local libGLU.so.1 for the RTX/iray renderer.
#
# Findings driving this version:
#  * Kit's local dependency solve fails: isaaclab.python.kit pins
#    isaacsim.asset.importer.urdf = 2.4.31 (exact), but Isaac Sim 5.1.0 ships 2.4.30.
#    Kit therefore MUST sync the remote extension registry to get 2.4.31.
#  * kit/default (ovextensionsprod.blob.core.windows.net) and kit/sdk ARE reachable
#    through the lab proxy -- the earlier "hang" was just a slow one-time index sync.
#    The index it already pulled is cached in ~/.local/share/ov/data/exts/v2/index.
#  * kit/community (dw290v42wisod.cloudfront.net) is NOT reachable at all, not even
#    through the proxy -> repoint just that one at a local empty dir so it cannot block.
set -o pipefail
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
cd /home/wangrenpeng/bench2dex/Bench2Dex
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/wangrenpeng/bench2dex/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export http_proxy=http://host.docker.internal:1081
export https_proxy=http://host.docker.internal:1081
export HTTP_PROXY=http://host.docker.internal:1081
export HTTPS_PROXY=http://host.docker.internal:1081
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
mkdir -p /tmp/empty_reg

KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg \
--/exts/omni.kit.registry.nucleus/registries/2/optional=true"
# NOTE: indices 0 (kit/default) and 1 (kit/sdk) are deliberately left pointing at
# their real URLs -- that is where isaacsim.asset.importer.urdf-2.4.31 comes from.

echo "=== smoke test (registry via proxy) start $(date -Is) ==="
"$ENVPY" run_policy.py \
    --policy-type ACT \
    --task scenes/26_canned_food_tray_line_arrangement.yaml \
    --ckpt-dir /home/wangrenpeng/bench2dex/policy_ckpt/26/multi_iiwa7_with_sharpa/act_active \
    --ckpt-name policy_best.ckpt \
    --robot-key multi_iiwa7_with_sharpa \
    --enable-rgb \
    --temporal-agg --temporal-agg-k 0.2 \
    --episode-steps 871 \
    --warmup-steps 60 \
    --num-episodes 2 \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir /home/wangrenpeng/bench2dex/teleopdata/dataset/26_canned_food_tray_line_arrangement/replay-generalization \
    --kit_args "$KIT_ARGS" \
    --headless
echo "=== smoke test exit=$? $(date -Is) ==="
