#!/bin/bash
# GR00T eval WITH ROLLOUT RECORDING (--record-dir --record-all) for the failure-prediction study.
# Mirrors policy/GR00T_n15/eval_double_env.sh (whose sim-side env name is the
# hard-coded `dex2bench`; ours is `bench2dex`), keeping our four Isaac fixes.
# Usage: 72_eval_gr00t.sh <TASK_ID> <SCENE_STEM> <EPISODE_STEPS> <GPU> <PORT>
set -o pipefail
TASK_ID="${1:?TASK_ID}"; SCENE="${2:?SCENE}"; STEPS="${3:?STEPS}"
GPU="${4:-2}"; PORT="${5:-9200}"; EPISODES="${6:-50}"; START_EP="${7:-1}"

ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOGS=$ROOT/_deploy_logs
GPP=/home/wangrenpeng/miniconda3/envs/GR00T_n15/bin/python
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
MODEL_PATH=$ROOT/policy_ckpt/$TASK_ID/multi_iiwa7_with_sharpa/gr00t_n15
ANCHOR=$ROOT/teleopdata/dataset/$SCENE/replay-generalization
OUT=$ROOT/output/metric/gr00t_${TASK_ID}_$(date +%m%d_%H%M)
KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg --/exts/omni.kit.registry.nucleus/registries/2/optional=true"

cd "$REPO" || exit 1
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export LD_LIBRARY_PATH=$ROOT/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export http_proxy=http://host.docker.internal:1081
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=$no_proxy
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT" /tmp/empty_reg

[ -d "$MODEL_PATH" ] || { echo "!!! GR00T ckpt dir missing: $MODEL_PATH"; exit 1; }
[ -d "$ANCHOR" ] || { echo "!!! anchor missing: $ANCHOR"; exit 1; }

echo "=== GR00T EVAL task=$TASK_ID scene=$SCENE steps=$STEPS gpu=$GPU port=$PORT episodes=$EPISODES start_ep=$START_EP start $(date -Is) ==="
echo "    model_path=$MODEL_PATH"; echo "    out=$OUT"

setsid nohup "$GPP" script/policy_model_server.py \
    --host 127.0.0.1 --port "$PORT" \
    --config policy/GR00T_n15/deploy_policy.yml --overrides \
    --policy_name GR00T_n15 \
    --task_name "$SCENE" \
    --model_path "$MODEL_PATH" \
    --ckpt_dir "$MODEL_PATH" \
    --ckpt_name gr00t_n15 \
    --seed 100000000 \
    --use_active_dof true \
    --robot_key multi_iiwa7_with_sharpa \
    > "$LOGS/72_gr00t_server_$TASK_ID.log" 2>&1 &
SERVER_PID=$!
echo "    [server] pid=$SERVER_PID"
trap 'kill -9 "$SERVER_PID" 2>/dev/null' EXIT

_ready=0
for _i in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; _ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server died:"; tail -25 "$LOGS/72_gr00t_server_$TASK_ID.log"; exit 1; }
    sleep 5
done
[ "$_ready" = "1" ] || { echo "!!! server never came up"; tail -25 "$LOGS/72_gr00t_server_$TASK_ID.log"; exit 1; }
echo "    [server] up after $((_i*5))s"

"$B2DPY" run_policy.py \
    --policy-type REMOTE \
    --remote-host 127.0.0.1 --remote-port "$PORT" \
    --task "scenes/${SCENE%.yaml}.yaml" \
    --robot-key multi_iiwa7_with_sharpa \
    --ckpt-dir "$MODEL_PATH" \
    --ckpt-name gr00t_n15 \
    --enable-rgb \
    --episode-steps "$STEPS" \
    --warmup-steps 60 \
    --num-episodes "$EPISODES" \
    --start-episode "$START_EP" \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir "$ANCHOR" \
    --output-dir "$OUT" \
    --policy-display-name gr00t_n15 \
      --record-dir "$OUT/rollouts" \
      --record-all \
    --kit_args "$KIT_ARGS" \
    --headless
RC=$?
echo "=== GR00T EVAL task=$TASK_ID client_exit=$RC $(date -Is) ==="
echo "    results: $OUT/per_episode.jsonl"
exit $RC
