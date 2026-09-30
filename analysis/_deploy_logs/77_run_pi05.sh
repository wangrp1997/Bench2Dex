#!/bin/bash
# Run the pi0.5 stage for tasks 26/32/73, then record + push.
# Uses the fixed 73_eval_pi05.sh (JAX on-demand GPU allocation; a zero-episode
# client exit is now a failure rather than a silent success).
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOG=$L/77_pi05.log
SSH="ssh -i /home/wangrenpeng/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
say(){ echo "$(date -Is) $*" | tee -a "$LOG"; }
scenes(){ case "$1" in 26) echo 26_canned_food_tray_line_arrangement;; 32) echo 32_baking_tray_prep_with_tools;; 73) echo 73_jigsaw_puzzle_assembly;; esac; }
steps(){ case "$1" in 26) echo 871;; 32) echo 852;; 73) echo 1213;; esac; }
alive(){ for T in 26 32 73; do tmux has-session -t "$1$T" 2>/dev/null && return 0; done; return 1; }

say "launching pi0.5 for tasks 26/32/73 (JAX preallocation disabled)"
PORT=9300
for T in 26 32 73; do
    tmux new-session -d -s "pi05$T" \
      "cd $REPO && $L/73_eval_pi05.sh $T $(scenes $T) $(steps $T) 2 $PORT 2>&1 | tee $L/77_pi05_$T.log"
    say "  launched pi05$T (port $PORT)"
    PORT=$((PORT+1))
    sleep 45
done
say "waiting for pi0.5 ..."
while alive pi05; do sleep 60; done
say "pi0.5 finished"
for T in 26 32 73; do
    d=$(ls -dt $ROOT/output/metric/pi05_${T}_* 2>/dev/null | head -1)
    say "  task $T: $(wc -l < $d/per_episode.jsonl 2>/dev/null || echo 0)/50"
done
python3 "$L/99_make_report.py" >>"$LOG" 2>&1
cp "$ROOT/RESULTS.md" "$REPO/RESULTS.md"
cd "$REPO" || exit 1
git add RESULTS.md
git diff --cached --quiet || { git -c core.pager=cat commit -q -m "results: pi0.5 baseline -- see RESULTS.md"; \
  timeout 180 env -u GIT_SSH_COMMAND GIT_TERMINAL_PROMPT=0 git -c core.sshCommand="$SSH" \
    push git@github.com:wangrp1997/Bench2Dex.git main >/tmp/p77 2>&1 \
  && say "pushed $(git rev-parse --short HEAD)" || say "PUSH FAILED $(tail -2 /tmp/p77 | tr '\n' ' ')"; }
say "PI05 STAGE COMPLETE"
