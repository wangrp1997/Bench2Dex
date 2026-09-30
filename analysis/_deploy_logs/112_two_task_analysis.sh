#!/bin/bash
# Wait on an explicit PID (robust against pgrep self-matching the watcher's own cmdline),
# then produce the two-task analyses.
set -uo pipefail
PID="${1:?usage: 112_two_task_analysis.sh <pid-to-wait-for>}"
ROOT=/home/wangrenpeng/bench2dex
L=$ROOT/_deploy_logs
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
export CUDA_VISIBLE_DEVICES=2

echo "=== waiting for pid $PID $(date -Is) ==="
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== pid $PID finished $(date -Is) ==="
echo "   task32 tactile rollouts: $(find /mnt/public/datasets/bench2dex/rollouts/32 -name '*.hdf5' | wc -l)"

rm -rf "$ROOT/data/rollout_labels/32_baking_tray_prep_with_tools"
"$B2DPY" "$L/101_extract_rollout_labels.py" \
    --dir /mnt/public/datasets/bench2dex/rollouts/32 \
    --scene 32_baking_tray_prep_with_tools > "$L/112_labels32.log" 2>&1
tail -2 "$L/112_labels32.log"

echo "=== TWO-TASK early warning $(date -Is) ==="
"$B2DPY" "$L/107_failure_early_warning.py" --tasks 26,32 --epochs 25 --folds 5 \
    --batch 8 --steps-per-epoch 12 --vision-encoder resnet18 --vision-res 160 \
    --modalities "tac,rgb,prop,tacstat" > "$L/112_two_task.log" 2>&1 || true
tail -16 "$L/112_two_task.log"

echo "=== ZERO-SHOT TRANSFER $(date -Is) ==="
"$B2DPY" "$L/111_transfer.py" --tasks 26,32 > "$L/112_transfer.log" 2>&1 || true
tail -8 "$L/112_transfer.log"

echo "=== ONLINE CONTINUAL LEARNING $(date -Is) ==="
"$B2DPY" "$L/108_online_continual.py" --strategies naive,reservoir,safety \
    --steps-per-ep 20 --buffer 600 --n-val 4 > "$L/112_cl.log" 2>&1 || true
tail -20 "$L/112_cl.log"
echo "=== DONE $(date -Is) ==="
