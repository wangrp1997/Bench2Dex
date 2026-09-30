#!/bin/bash
# Chain: wait for the three ACT evaluations to finish, then start the three DP
# evaluations (split server/client) automatically.
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
REPO=/home/wangrenpeng/bench2dex/Bench2Dex
DPPY=/home/wangrenpeng/miniconda3/envs/dp/bin/python
ROOT=/home/wangrenpeng/bench2dex

echo "### chain: waiting for ACT sessions (eval26/eval32/eval73) to end $(date -Is)"
while tmux has-session -t eval26 2>/dev/null || tmux has-session -t eval32 2>/dev/null || tmux has-session -t eval73 2>/dev/null; do
    sleep 60
done
echo "### ACT all finished, preparing DP $(date -Is)"

# ── sanity gates: do not start DP unless everything is really in place ──────
[ -x "$DPPY" ] || { echo "!!! dp env python missing: $DPPY"; exit 1; }
"$DPPY" -c "import torch, hydra, dp_model" 2>/dev/null || {
    cd "$REPO" || exit 1
    "$DPPY" -c "import sys; sys.path.insert(0,'policy/DP'); import torch, hydra, dp_model" \
      || { echo "!!! dp env broken"; exit 1; }
}
for T in 26 32 73; do
    f="$ROOT/policy_ckpt/$T/multi_iiwa7_with_sharpa/dp/checkpoints/bs512_ep300.ckpt"
    [ -f "$f" ] || { echo "!!! DP ckpt missing for task $T: $f"; exit 1; }
    a="$ROOT/teleopdata/dataset/$( [ "$T" = 26 ] && echo 26_canned_food_tray_line_arrangement || { [ "$T" = 32 ] && echo 32_baking_tray_prep_with_tools || echo 73_jigsaw_puzzle_assembly; } )/replay-generalization"
    [ -d "$a" ] || { echo "!!! anchor missing for task $T: $a"; exit 1; }
done
echo "### gates passed, launching DP $(date -Is)"

SCENE_26=26_canned_food_tray_line_arrangement
SCENE_32=32_baking_tray_prep_with_tools
SCENE_73=73_jigsaw_puzzle_assembly
PORT=9110
for T in 26 32 73; do
    eval "scene=\$SCENE_$T"
    case "$T" in 26) steps=871;; 32) steps=852;; 73) steps=1213;; esac
    tmux new-session -d -s "dp$T" \
      "cd $REPO && $L/70_eval_dp.sh $T $scene $steps 2 $PORT 2>&1 | tee $L/70_eval_dp_$T.log"
    echo "### started tmux session dp$T (port=$PORT, scene=$scene, steps=$steps)"
    PORT=$((PORT+1))
    sleep 45
done
echo "### DP chain done $(date -Is)"
tmux ls
