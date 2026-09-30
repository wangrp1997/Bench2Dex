#!/bin/bash
# As soon as the task-32 recording eval finishes (independent of task 73), build its
# tactile rollouts and produce an intermediate TWO-TASK result. The master chain later
# repeats this for three tasks; everything here is idempotent (106/101 skip existing).
set -uo pipefail
ROOT=/home/wangrenpeng/bench2dex
L=$ROOT/_deploy_logs
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
export CUDA_VISIBLE_DEVICES=2

echo "=== waiting for the task-32 eval to finish $(date -Is) ==="
while pgrep -f '105_eval32.lo[g]' >/dev/null 2>&1 || ! grep -q 'GR00T EVAL task=32 client_exit' "$L/105_eval32.log" 2>/dev/null; do
    sleep 60
done
echo "=== task 32 eval finished $(date -Is) ==="

RD=$(ls -dt "$ROOT"/output/metric/gr00t_32_* 2>/dev/null | head -1)
echo "   rollouts: $RD/rollouts  ($(find "$RD/rollouts" -name '*.hdf5' | wc -l) files)"
"$L/106_patch_and_replay.sh" 32 32_baking_tray_prep_with_tools "$RD/rollouts" 2 \
    > "$L/110_replay32.log" 2>&1
echo "   tactile rollouts: $(find /mnt/public/datasets/bench2dex/rollouts/32 -name '*.hdf5' | wc -l)"

rm -rf "$ROOT/data/rollout_labels/32_baking_tray_prep_with_tools"
"$B2DPY" "$L/101_extract_rollout_labels.py" \
    --dir /mnt/public/datasets/bench2dex/rollouts/32 \
    --scene 32_baking_tray_prep_with_tools > "$L/110_labels32.log" 2>&1
tail -2 "$L/110_labels32.log"

echo "=== TWO-TASK early warning $(date -Is) ==="
"$B2DPY" "$L/107_failure_early_warning.py" --tasks 26,32 --epochs 25 --folds 5 \
    --batch 8 --steps-per-epoch 12 --vision-encoder resnet18 --vision-res 160 \
    --modalities "tac,rgb,prop" > "$L/110_two_task.log" 2>&1
tail -14 "$L/110_two_task.log"

echo "=== TWO-TASK online continual learning $(date -Is) ==="
"$B2DPY" "$L/108_online_continual.py" --strategies naive,reservoir,safety \
    --steps-per-ep 20 --buffer 600 > "$L/110_two_task_cl.log" 2>&1
tail -16 "$L/110_two_task_cl.log"
echo "=== TWO-TASK CHAIN DONE $(date -Is) ==="
