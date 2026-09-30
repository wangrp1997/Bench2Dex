#!/bin/bash
# Master chain: after the recording evals finish, run the whole downstream pipeline
# unattended -- tactile replay, per-frame labels, early-warning model, continual learning.
set -uo pipefail
ROOT=/home/wangrenpeng/bench2dex
L=$ROOT/_deploy_logs
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
export CUDA_VISIBLE_DEVICES=2

scenes(){ case "$1" in
  26) echo 26_canned_food_tray_line_arrangement;;
  32) echo 32_baking_tray_prep_with_tools;;
  73) echo 73_jigsaw_puzzle_assembly;; esac; }

echo "=== waiting for the recording evals (105 queue) $(date -Is) ==="
while pgrep -f '105_eval_queu[e].sh' >/dev/null 2>&1; do sleep 60; done
echo "=== all recording evals finished $(date -Is) ==="

for T in 26 32 73; do
    SC=$(scenes "$T")
    RD=$(ls -dt "$ROOT"/output/metric/gr00t_${T}_* 2>/dev/null | head -1)
    if [ -z "$RD" ] || [ ! -d "$RD/rollouts" ]; then
        echo "!! no rollouts for task $T, skipping"; continue
    fi
    echo "--- task $T rollouts: $RD/rollouts"
    find "$RD/rollouts" -name '*.hdf5' | wc -l | sed 's/^/    recorded: /'

    # 1) patch metadata + attach tactile by replaying each rollout
    "$L/106_patch_and_replay.sh" "$T" "$SC" "$RD/rollouts" 3 \
        > "$L/109_replay_$T.log" 2>&1
    echo "    tactile rollouts: $(find /mnt/public/datasets/bench2dex/rollouts/$T -name '*.hdf5' 2>/dev/null | wc -l)"

    # 2) per-frame stage labels on the replayed rollouts
    rm -rf "$ROOT/data/rollout_labels/$SC"
    "$B2DPY" "$L/101_extract_rollout_labels.py" \
        --dir "/mnt/public/datasets/bench2dex/rollouts/$T" --scene "$SC" \
        > "$L/109_labels_$T.log" 2>&1
    tail -2 "$L/109_labels_$T.log" | sed 's/^/    /'
done

echo "=== online continual learning: no rarity $(date -Is) ==="
"$B2DPY" "$L/108_online_continual.py" --strategies naive,reservoir,safety \
    --steps-per-ep 20 --buffer 600 --stream-failure-rate 1.0 \
    > "$L/109_continual_full.log" 2>&1
tail -8 "$L/109_continual_full.log" | sed 's/^/    /'

echo "=== online continual learning: RARE failures (deployment reality) $(date -Is) ==="
"$B2DPY" "$L/108_online_continual.py" --strategies naive,reservoir,safety \
    --steps-per-ep 20 --buffer 600 --stream-failure-rate 0.15 \
    > "$L/109_continual_rare.log" 2>&1
tail -8 "$L/109_continual_rare.log" | sed 's/^/    /'

echo "=== early warning: 3 tasks $(date -Is) ==="
"$B2DPY" "$L/107_failure_early_warning.py" --tasks 26,32,73 --epochs 25 --folds 5 \
    --batch 8 --steps-per-epoch 12 --vision-encoder resnet18 --vision-res 160 \
    --modalities "tac,rgb,prop,tacstat" > "$L/109_ew3.log" 2>&1
tail -25 "$L/109_ew3.log" | sed 's/^/    /'

echo "=== zero-shot transfer: 3 tasks $(date -Is) ==="
"$B2DPY" "$L/111_transfer.py" --tasks 26,32,73 > "$L/109_transfer3.log" 2>&1
tail -12 "$L/109_transfer3.log" | sed 's/^/    /'

echo "=== MASTER CHAIN DONE $(date -Is) ==="
