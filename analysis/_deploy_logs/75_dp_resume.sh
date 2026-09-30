#!/bin/bash
# Resume an interrupted DP evaluation (server/client) from where it stopped.
# Usage: 75_dp_resume.sh <TASK_ID> <SCENE> <STEPS> <START_EP> <NUM_EP> <OUT_DIR> <GPU> <PORT>
set -o pipefail
TASK_ID="${1:?}"; SCENE="${2:?}"; STEPS="${3:?}"; START="${4:?}"; NUM="${5:?}"; OUT="${6:?}"
GPU="${7:-2}"; PORT="${8:-9110}"

ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOGS=$ROOT/_deploy_logs
DPPY=/home/wangrenpeng/miniconda3/envs/dp/bin/python
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
CKPT=$ROOT/policy_ckpt/$TASK_ID/multi_iiwa7_with_sharpa/dp/checkpoints/bs512_ep300.ckpt
ANCHOR=$ROOT/teleopdata/dataset/$SCENE/replay-generalization
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
mkdir -p /tmp/empty_reg

echo "=== DP RESUME task=$TASK_ID start_ep=$START n=$NUM port=$PORT $(date -Is) ==="
echo "    out=$OUT"

# kill any leftover server for this task/port
for p in $(pgrep -f "policy_model_server"); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "port $PORT" && { echo "    killing stale server pid=$p"; kill -9 $p; }
done
sleep 2

setsid nohup "$DPPY" script/policy_model_server.py \
    --host 127.0.0.1 --port "$PORT" \
    --config policy/DP/deploy_policy.yml --overrides \
    --policy_name DP --task_name "$SCENE" \
    --checkpoint_path "$CKPT" \
    --training_config_path policy/DP/diffusion_policy/config/robot_dp_36_dex2scene_pretrained.yaml \
    --seed 100000000 --use_active_dof true --robot_key multi_iiwa7_with_sharpa \
    > "$LOGS/75_dp_server_${TASK_ID}_resume.log" 2>&1 &
SERVER_PID=$!
trap 'kill -9 "$SERVER_PID" 2>/dev/null' EXIT
_ready=0
for _i in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; _ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server died:"; tail -20 "$LOGS/75_dp_server_${TASK_ID}_resume.log"; exit 1; }
    sleep 5
done
[ "$_ready" = "1" ] || { echo "!!! server never came up"; exit 1; }
echo "    [server] up after $((_i*5))s"

"$B2DPY" run_policy.py \
    --policy-type REMOTE --remote-host 127.0.0.1 --remote-port "$PORT" \
    --task "scenes/${SCENE%.yaml}.yaml" \
    --robot-key multi_iiwa7_with_sharpa \
    --enable-rgb \
    --episode-steps "$STEPS" --warmup-steps 60 \
    --num-episodes "$NUM" --start-episode "$START" \
    --output-dir "$OUT" --append-output \
    --seed 100000000 --generalization-profile none \
    --anchor-dir "$ANCHOR" \
    --policy-display-name dp_bs512_ep300 \
    --kit_args "$KIT_ARGS" \
    --headless
RC=$?
echo "=== DP RESUME task=$TASK_ID client_exit=$RC $(date -Is) ==="
exit $RC
