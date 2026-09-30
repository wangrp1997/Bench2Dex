#!/bin/bash
# Bench2Dex full evaluation: ACT x {26,32,73} x generalization-profile none, 50 episodes.
# Usage: 20_eval_act.sh <TASK_ID> <SCENE_STEM> <EPISODE_STEPS> <GPU_INDEX>
#
# Env carried over from the verified smoke run (13_smoke.sh):
#   * CUDA_VISIBLE_DEVICES   - pin to an idle GPU
#   * LD_LIBRARY_PATH        - local libGLU.so.1 (RTX/iray needs it, no root here)
#   * HTTP(S)_PROXY          - so omni.kit.registry.nucleus can sync kit/default + kit/sdk
#   * --kit_args             - kit/community (cloudfront) is unreachable -> local empty dir
#   * ACCEPT_EULA / OMNI_KIT_ACCEPT_EULA, PYTHONPATH unset
set -o pipefail
TASK_ID="$1"; SCENE="$2"; STEPS="$3"; GPU="${4:-2}"
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
ROOT=/home/wangrenpeng/bench2dex
cd "$ROOT/Bench2Dex" || exit 1
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$ROOT/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export http_proxy=http://host.docker.internal:1081 https_proxy=http://host.docker.internal:1081
export HTTP_PROXY=http://host.docker.internal:1081 HTTPS_PROXY=http://host.docker.internal:1081
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
mkdir -p /tmp/empty_reg
KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg --/exts/omni.kit.registry.nucleus/registries/2/optional=true"

echo "=== EVAL task=$TASK_ID scene=$SCENE steps=$STEPS gpu=$GPU start $(date -Is) ==="
"$ENVPY" run_policy.py \
    --policy-type ACT \
    --task "scenes/${SCENE%.yaml}.yaml" \
    --ckpt-dir "$ROOT/policy_ckpt/$TASK_ID/multi_iiwa7_with_sharpa/act_active" \
    --ckpt-name policy_best.ckpt \
    --robot-key multi_iiwa7_with_sharpa \
    --enable-rgb \
    --temporal-agg --temporal-agg-k 0.2 \
    --episode-steps "$STEPS" \
    --warmup-steps 60 \
    --num-episodes 50 \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir "$ROOT/teleopdata/dataset/$SCENE/replay-generalization" \
    --kit_args "$KIT_ARGS" \
    --headless
echo "=== EVAL task=$TASK_ID exit=$? $(date -Is) ==="
