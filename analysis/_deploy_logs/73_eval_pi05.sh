#!/bin/bash
# Bench2Dex pi0.5 (openpi) evaluation — split server/client launcher.
# Mirrors policy/pi05/eval_double_env.sh (its sim-side env is hard-coded to
# `dex2bench`; ours is `bench2dex`), keeping our four Isaac fixes.
# Usage: 73_eval_pi05.sh <TASK_ID> <SCENE_STEM> <EPISODE_STEPS> <GPU> <PORT>
set -o pipefail
TASK_ID="${1:?TASK_ID}"; SCENE="${2:?SCENE}"; STEPS="${3:?STEPS}"
GPU="${4:-2}"; PORT="${5:-9300}"
TRAIN_CONFIG="${6:-pi05_base_dex2bench_full}"

ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOGS=$ROOT/_deploy_logs
P5PY=/home/wangrenpeng/miniconda3/envs/pi05/bin/python
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
PI05_ROOT=$ROOT/policy_ckpt/$TASK_ID/multi_iiwa7_with_sharpa/pi05
ANCHOR=$ROOT/teleopdata/dataset/$SCENE/replay-generalization
OUT=$ROOT/output/metric/pi05_${TASK_ID}_$(date +%m%d_%H%M)
KIT_ARGS="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg --/exts/omni.kit.registry.nucleus/registries/2/optional=true"

# pi_model.py wants the step directory that contains an `assets/` subdir.
CKPT_DIR=$(find "$PI05_ROOT" -maxdepth 4 -type d -name assets 2>/dev/null | head -1)
CKPT_DIR=$(dirname "$CKPT_DIR" 2>/dev/null)

cd "$REPO" || exit 1
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export LD_LIBRARY_PATH=$ROOT/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export http_proxy=http://host.docker.internal:1081
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=$no_proxy
export CUDA_VISIBLE_DEVICES="$GPU"
# JAX (openpi / pi0.5) preallocates 75% of the GPU by default. On this box that sent
# GPU 2 to 97 GB used / 179 MB free, and the Isaac Sim client then died with
# "[gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state" yet still
# exited 0. Force on-demand allocation so three servers + three sims can coexist.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
mkdir -p "$OUT" /tmp/empty_reg

[ -d "$PI05_ROOT" ] || { echo "!!! pi05 ckpt root missing: $PI05_ROOT"; exit 1; }
[ -n "$CKPT_DIR" ] && [ -d "$CKPT_DIR" ] || { echo "!!! could not locate a pi05 step dir with assets/ under $PI05_ROOT"; ls -R "$PI05_ROOT" 2>/dev/null | head -20; exit 1; }
[ -d "$ANCHOR" ] || { echo "!!! anchor missing: $ANCHOR"; exit 1; }

echo "=== PI05 EVAL task=$TASK_ID scene=$SCENE steps=$STEPS gpu=$GPU port=$PORT start $(date -Is) ==="
echo "    train_config=$TRAIN_CONFIG"
echo "    ckpt_dir=$CKPT_DIR"
echo "    out=$OUT"

setsid nohup "$P5PY" script/policy_model_server.py \
    --host 127.0.0.1 --port "$PORT" \
    --config policy/pi05/deploy_policy.yml --overrides \
    --policy_name pi05 \
    --task_name "$SCENE" \
    --train_config_name "$TRAIN_CONFIG" \
    --checkpoint_path "$CKPT_DIR" \
    --seed 100000000 \
    --use_active_dof true \
    --robot_key multi_iiwa7_with_sharpa \
    > "$LOGS/73_pi05_server_$TASK_ID.log" 2>&1 &
SERVER_PID=$!
echo "    [server] pid=$SERVER_PID"
trap 'kill -9 "$SERVER_PID" 2>/dev/null' EXIT

_ready=0
for _i in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; _ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server died:"; tail -25 "$LOGS/73_pi05_server_$TASK_ID.log"; exit 1; }
    sleep 5
done
[ "$_ready" = "1" ] || { echo "!!! server never came up"; tail -25 "$LOGS/73_pi05_server_$TASK_ID.log"; exit 1; }
echo "    [server] up after $((_i*5))s"

"$B2DPY" run_policy.py \
    --policy-type REMOTE \
    --remote-host 127.0.0.1 --remote-port "$PORT" \
    --task "scenes/${SCENE%.yaml}.yaml" \
    --robot-key multi_iiwa7_with_sharpa \
    --ckpt-dir "$CKPT_DIR" \
    --enable-rgb \
    --episode-steps "$STEPS" \
    --warmup-steps 60 \
    --num-episodes 50 \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir "$ANCHOR" \
    --output-dir "$OUT" \
    --policy-display-name pi05 \
    --kit_args "$KIT_ARGS" \
    --headless
RC=$?
# A silent zero-episode "success" happened once (JAX had exhausted the GPU, so
# Isaac Sim reported "CUDA being in bad state" and run_policy exited 0). Never
# report success when nothing was evaluated.
EPS=$(wc -l < "$OUT/per_episode.jsonl" 2>/dev/null || echo 0)
if [ "$RC" = "0" ] && [ "$EPS" -eq 0 ]; then
    echo "!!! client exited 0 but recorded 0 episodes -- treating as FAILURE"
    echo "    (last renderer messages:)"
    grep -E "bad state|out of memory|Texture creation failed|NGX CreateFeature" "$OUT/../$(basename "$OUT").log" 2>/dev/null | tail -3
    RC=1
fi
echo "=== PI05 EVAL task=$TASK_ID client_exit=$RC episodes=$EPS $(date -Is) ==="
echo "    results: $OUT/per_episode.jsonl"
exit $RC
