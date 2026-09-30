#!/bin/bash
# Resume an interrupted pi0.5 evaluation.
# Reuses an already-listening policy server on the port if present (the stalled
# client's server is usually still healthy), otherwise starts one.
# Usage: 78_pi05_resume.sh <TASK_ID> <SCENE> <STEPS> <START_EP> <NUM_EP> <OUT_DIR> <GPU> <PORT>
set -o pipefail
TASK_ID="${1:?}"; SCENE="${2:?}"; STEPS="${3:?}"; START="${4:?}"; NUM="${5:?}"; OUT="${6:?}"
GPU="${7:-2}"; PORT="${8:-9300}"

ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOGS=$ROOT/_deploy_logs
P5PY=/home/wangrenpeng/miniconda3/envs/pi05/bin/python
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
PI05_ROOT=$ROOT/policy_ckpt/$TASK_ID/multi_iiwa7_with_sharpa/pi05
TRAIN_CONFIG=pi05_base_dex2bench_full
ANCHOR=$ROOT/teleopdata/dataset/$SCENE/replay-generalization
KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg --/exts/omni.kit.registry.nucleus/registries/2/optional=true"
CKPT_DIR=$(dirname "$(find "$PI05_ROOT" -maxdepth 4 -type d -name assets 2>/dev/null | head -1)")

cd "$REPO" || exit 1
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export LD_LIBRARY_PATH=$ROOT/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export http_proxy=http://host.docker.internal:1081
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=$no_proxy
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
mkdir -p /tmp/empty_reg

echo "=== PI05 RESUME task=$TASK_ID start_ep=$START n=$NUM port=$PORT $(date -Is) ==="
echo "    ckpt_dir=$CKPT_DIR"
echo "    out=$OUT"

# reuse an already-running server on this port if there is one
SERVER_PID=""
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    exec 3>&-
    echo "    [server] reusing the server already listening on $PORT"
else
    setsid nohup "$P5PY" script/policy_model_server.py \
        --host 127.0.0.1 --port "$PORT" \
        --config policy/pi05/deploy_policy.yml --overrides \
        --policy_name pi05 --task_name "$SCENE" \
        --train_config_name "$TRAIN_CONFIG" --checkpoint_path "$CKPT_DIR" \
        --seed 100000000 --use_active_dof true --robot_key multi_iiwa7_with_sharpa \
        > "$LOGS/78_pi05_server_${TASK_ID}_resume.log" 2>&1 &
    SERVER_PID=$!
    _ready=0
    for _i in $(seq 1 120); do
        if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; _ready=1; break; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server died:"; tail -20 "$LOGS/78_pi05_server_${TASK_ID}_resume.log"; exit 1; }
        sleep 5
    done
    [ "$_ready" = "1" ] || { echo "!!! server never came up"; exit 1; }
    echo "    [server] started, up after $((_i*5))s"
fi

"$B2DPY" run_policy.py \
    --policy-type REMOTE --remote-host 127.0.0.1 --remote-port "$PORT" \
    --task "scenes/${SCENE%.yaml}.yaml" \
    --robot-key multi_iiwa7_with_sharpa \
    --ckpt-dir "$CKPT_DIR" --ckpt-name pi05 \
    --enable-rgb \
    --episode-steps "$STEPS" --warmup-steps 60 \
    --num-episodes "$NUM" --start-episode "$START" \
    --output-dir "$OUT" --append-output \
    --seed 100000000 --generalization-profile none \
    --anchor-dir "$ANCHOR" \
    --policy-display-name pi05 \
    --kit_args "$KIT_ARGS" \
    --headless
RC=$?
EPS=$(wc -l < "$OUT/per_episode.jsonl" 2>/dev/null || echo 0)
[ "$RC" = "0" ] && [ "$EPS" -eq 0 ] && { echo "!!! exit 0 with 0 episodes -- FAILURE"; RC=1; }
echo "=== PI05 RESUME task=$TASK_ID client_exit=$RC episodes=$EPS $(date -Is) ==="
exit $RC
