#!/bin/bash
# Smoke test attempt 2: same as Section 7 command + --num-episodes 2,
# but with the lab HTTP proxy exported so omni.kit.registry.nucleus can
# reach ovextensionsprod.blob.core.windows.net / dw290v42wisod.cloudfront.net
# (both hung when contacted directly from this network).
set -o pipefail
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
cd /home/wangrenpeng/bench2dex/Bench2Dex
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export http_proxy=http://host.docker.internal:1081
export https_proxy=http://host.docker.internal:1081
export HTTP_PROXY=http://host.docker.internal:1081
export HTTPS_PROXY=http://host.docker.internal:1081
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
echo "=== smoke test (proxy) start $(date -Is) ==="
echo "proxy: HTTP_PROXY=$HTTP_PROXY  NO_PROXY=$NO_PROXY"
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
    --headless
echo "=== smoke test exit=$? $(date -Is) ==="
