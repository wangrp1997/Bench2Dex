#!/bin/bash
# Section 7 "verified run command" with --num-episodes 2 (Pitfall 12/13 applied)
set -o pipefail
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
cd /home/wangrenpeng/bench2dex/Bench2Dex
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
echo "=== smoke test start $(date -Is) (Task 26 / Sharpa / ACT / profile none) ==="
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
