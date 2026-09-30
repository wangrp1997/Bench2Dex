#!/bin/bash
# Finish the overnight pipeline after the (self-inflicted) watchdog incident:
#   STAGE 1  resume DP for tasks 26/32/73 to 50 episodes
#   STAGE 2  run pi0.5 for tasks 26/32/73
# Records RESULTS.md and pushes after each stage.
#
# Deliberately contains NO automatic process killing: the two watchdog attempts
# both misfired (v1 was blind, v2 matched a stale same-task log from a *previous*
# family and killed the live process group). Stalls are left for the supervising
# agent to handle by hand.
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
LOG=$L/76_finish.log
SSH="ssh -i /home/wangrenpeng/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
say(){ echo "$(date -Is) $*" | tee -a "$LOG"; }

scenes(){ case "$1" in 26) echo 26_canned_food_tray_line_arrangement;; 32) echo 32_baking_tray_prep_with_tools;; 73) echo 73_jigsaw_puzzle_assembly;; esac; }
steps(){ case "$1" in 26) echo 871;; 32) echo 852;; 73) echo 1213;; esac; }

alive(){ for T in 26 32 73; do tmux has-session -t "$1$T" 2>/dev/null && return 0; done; return 1; }

push(){ # $1 label
    python3 "$L/99_make_report.py" >>"$LOG" 2>&1
    cp "$ROOT/RESULTS.md" "$REPO/RESULTS.md"
    cd "$REPO" || return 1
    git add RESULTS.md
    git diff --cached --quiet && { say "$1: nothing to push"; return 0; }
    git -c core.pager=cat commit -q -m "results: $1 -- see RESULTS.md" || true
    timeout 180 env -u GIT_SSH_COMMAND GIT_TERMINAL_PROMPT=0 \
      git -c core.sshCommand="$SSH" push git@github.com:wangrp1997/Bench2Dex.git main >/tmp/p76 2>&1
    [ $? -eq 0 ] && say "$1: pushed $(git rev-parse --short HEAD)" || say "$1: PUSH FAILED $(tail -2 /tmp/p76 | tr '\n' ' ')"
}

# ═══ STAGE 1: finish DP ════════════════════════════════════════════════════
say "STAGE 1: resuming DP"
PORT=9110
for T in 26 32 73; do
    d=$(ls -dt $ROOT/output/metric/dp_${T}_* 2>/dev/null | head -1)
    n=$(wc -l < "$d/per_episode.jsonl" 2>/dev/null || echo 0)
    start=$((n+1)); num=$((50-n))
    if [ "$num" -le 0 ]; then say "  task $T already complete ($n)"; PORT=$((PORT+1)); continue; fi
    say "  task $T: $n/50 -> resuming $start..50 ($num eps) on port $PORT"
    tmux new-session -d -s "dp$T" \
      "cd $REPO && $L/75_dp_resume.sh $T $(scenes $T) $(steps $T) $start $num $d 2 $PORT 2>&1 | tee $L/76_dp_$T.log"
    PORT=$((PORT+1))
    sleep 40
done
say "waiting for DP to finish ..."
while alive dp; do sleep 60; done
say "DP finished"
for T in 26 32 73; do
    d=$(ls -dt $ROOT/output/metric/dp_${T}_* 2>/dev/null | head -1)
    say "  task $T: $(wc -l < $d/per_episode.jsonl 2>/dev/null)/50"
done
push "DP baseline (completed)"

# ═══ STAGE 2: pi0.5 ════════════════════════════════════════════════════════
say "STAGE 2: launching pi0.5"
PORT=9300
for T in 26 32 73; do
    tmux new-session -d -s "pi05$T" \
      "cd $REPO && $L/73_eval_pi05.sh $T $(scenes $T) $(steps $T) 2 $PORT 2>&1 | tee $L/76_pi05_$T.log"
    say "  launched pi05$T on port $PORT"
    PORT=$((PORT+1))
    sleep 45
done
say "waiting for pi0.5 to finish ..."
while alive pi05; do sleep 60; done
say "pi0.5 finished"
for T in 26 32 73; do
    d=$(ls -dt $ROOT/output/metric/pi05_${T}_* 2>/dev/null | head -1)
    say "  task $T: $(wc -l < $d/per_episode.jsonl 2>/dev/null)/50"
done
push "pi0.5 baseline"
say "PIPELINE COMPLETE"
