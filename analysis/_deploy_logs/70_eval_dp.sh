#!/bin/bash
# Bench2Dex DP (Diffusion Policy) evaluation — split server/client launcher.
#
# Why not policy/DP/eval_double_env.sh:
#   it hard-codes `conda activate dp` for the server AND `conda activate dex2bench`
#   for the sim client. Our sim env is named `bench2dex`, so we replicate its
#   logic here with the correct env names (and keep our four Isaac fixes).
#
# Usage: 70_eval_dp.sh <TASK_ID> <SCENE_STEM> <EPISODE_STEPS> <GPU> <PORT>
set -o pipefail
TASK_ID="${1:?TASK_ID}"; SCENE="${2:?SCENE}"; STEPS="${3:?STEPS}"
GPU="${4:-2}"; PORT="${5:-9100}"

ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOGS=$ROOT/_deploy_logs
DPPY=/home/wangrenpeng/miniconda3/envs/dp/bin/python
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
CKPT=$ROOT/policy_ckpt/$TASK_ID/multi_iiwa7_with_sharpa/dp/checkpoints/bs512_ep300.ckpt
ANCHOR=$ROOT/teleopdata/dataset/$SCENE/replay-generalization
OUT=$ROOT/output/metric/dp_${TASK_ID}_$(date +%m%d_%H%M)
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

[ -f "$CKPT" ] || { echo "!!! DP checkpoint missing: $CKPT"; exit 1; }
[ -d "$ANCHOR" ] || { echo "!!! anchor missing: $ANCHOR"; exit 1; }

echo "=== DP EVAL task=$TASK_ID scene=$SCENE steps=$STEPS gpu=$GPU port=$PORT start $(date -Is) ==="
echo "    ckpt=$CKPT"
echo "    out=$OUT"

# ── 1) policy server, in the `dp` env ────────────────────────────────────────
setsid nohup "$DPPY" script/policy_model_server.py \
    --host 127.0.0.1 --port "$PORT" \
    --config policy/DP/deploy_policy.yml --overrides \
    --policy_name DP \
    --task_name "$SCENE" \
    --checkpoint_path "$CKPT" \
    --training_config_path policy/DP/diffusion_policy/config/robot_dp_36_dex2scene_pretrained.yaml \
    --seed 100000000 \
    --use_active_dof true \
    --robot_key multi_iiwa7_with_sharpa \
    > "$LOGS/70_dp_server_$TASK_ID.log" 2>&1 &
SERVER_PID=$!
echo "    [server] pid=$SERVER_PID -> $LOGS/70_dp_server_$TASK_ID.log"
trap 'kill -9 "$SERVER_PID" 2>/dev/null; pkill -9 -P "$SERVER_PID" 2>/dev/null' EXIT

# ── 2) wait for the server to listen ─────────────────────────────────────────
echo "    [server] waiting for port $PORT ..."
_ready=0
for _i in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&- ; _ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server died early:"; tail -20 "$LOGS/70_dp_server_$TASK_ID.log"; exit 1; }
    sleep 5
done
[ "$_ready" = "1" ] || { echo "!!! server never came up"; tail -20 "$LOGS/70_dp_server_$TASK_ID.log"; exit 1; }
echo "    [server] up after $((_i*5))s"

# ── 3) sim client, in the bench2dex (Isaac Sim) env ──────────────────────────
"$B2DPY" run_policy.py \
    --policy-type REMOTE \
    --remote-host 127.0.0.1 --remote-port "$PORT" \
    --task "scenes/${SCENE%.yaml}.yaml" \
    --robot-key multi_iiwa7_with_sharpa \
    --enable-rgb \
    --episode-steps "$STEPS" \
    --warmup-steps 60 \
    --num-episodes 50 \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir "$ANCHOR" \
    --output-dir "$OUT" \
    --policy-display-name dp_bs512_ep300 \
    --kit_args "$KIT_ARGS" \
    --headless
RC=$?
echo "=== DP EVAL task=$TASK_ID client_exit=$RC $(date -Is) ==="
echo "    results: $OUT/per_episode.jsonl"
exit $RC
