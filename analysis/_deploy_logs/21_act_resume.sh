#!/bin/bash
# Resume an interrupted ACT evaluation from where it stopped, appending to the
# existing output dir.
# Usage: 21_act_resume.sh <TASK_ID> <SCENE_STEM> <EPISODE_STEPS> <START_EP> <NUM_EP> <OUT_DIR> [GPU]
set -o pipefail
TASK_ID="${1:?}"; SCENE="${2:?}"; STEPS="${3:?}"; START="${4:?}"; NUM="${5:?}"; OUT="${6:?}"; GPU="${7:-2}"
ROOT=/home/wangrenpeng/bench2dex
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
cd "$ROOT/Bench2Dex" || exit 1
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=$ROOT/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export http_proxy=http://host.docker.internal:1081
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=$no_proxy
mkdir -p /tmp/empty_reg
KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg --/exts/omni.kit.registry.nucleus/registries/2/optional=true"

echo "=== ACT RESUME task=$TASK_ID start_ep=$START n=$NUM gpu=$GPU $(date -Is) ==="
echo "    out=$OUT"
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
    --num-episodes "$NUM" \
    --start-episode "$START" \
    --output-dir "$OUT" \
    --append-output \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir "$ROOT/teleopdata/dataset/$SCENE/replay-generalization" \
    --kit_args "$KIT_ARGS" \
    --headless
echo "=== ACT RESUME task=$TASK_ID exit=$? $(date -Is) ==="
