#!/bin/bash
# Smoke test attempt 3: Section 7 command + --num-episodes 2.
#
# FIX for the startup hang: omni.kit.registry.nucleus blocks on its three REMOTE
# extension registries (ovextensionsprod.blob.core.windows.net, dw290v42wisod.cloudfront.net).
# Those hosts are throttled/blackholed from this network, so Kit never gets past
# extension startup. The pip install already ships every extension in isaacsim/extscache,
# so we repoint all three registries at an empty local folder and mark them optional
# (documented in omni.kit.registry.nucleus/docs/Overview.md). Passed through
# IsaacLab's AppLauncher --kit_args.
set -o pipefail
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
cd /home/wangrenpeng/bench2dex/Bench2Dex
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
mkdir -p /tmp/empty_reg

KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/0/url=/tmp/empty_reg \
--/exts/omni.kit.registry.nucleus/registries/0/optional=true \
--/exts/omni.kit.registry.nucleus/registries/1/url=/tmp/empty_reg \
--/exts/omni.kit.registry.nucleus/registries/1/optional=true \
--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg \
--/exts/omni.kit.registry.nucleus/registries/2/optional=true"

echo "=== smoke test (offline registry) start $(date -Is) ==="
echo "kit_args: $KIT_ARGS"
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
