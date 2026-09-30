#!/bin/bash
# After each algorithm family finishes, snapshot RESULTS.md into the repo and
# push once to github (over SSH -- the HTTPS route has no credentials here).
#
# A family counts as "finished" when none of its tmux sessions exist any more and
# at least one episode has been recorded (so a fast-failing stage such as GR00T
# still gets recorded rather than silently skipped).
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
PUSH_LOG=$L/96_pushes.log
SSH="ssh -i /home/wangrenpeng/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

say(){ echo "$(date -Is) $*" | tee -a "$PUSH_LOG"; }

family_done() {  # $1 = tmux prefix, $2 = metric dir prefix
    for T in 26 32 73; do
        tmux has-session -t "$1$T" 2>/dev/null && return 1
    done
    # at least one episode recorded somewhere for this family
    compgen -G "$ROOT/output/metric/$2"* >/dev/null 2>&1 || return 1
    return 0
}

push_once() {  # $1 = label
    local label="$1"
    python3 "$L/99_make_report.py" >>"$PUSH_LOG" 2>&1
    cp "$ROOT/RESULTS.md" "$REPO/RESULTS.md"
    cd "$REPO" || return 1
    git add RESULTS.md
    if git diff --cached --quiet; then say "$label: RESULTS.md unchanged, nothing to push"; return 0; fi
    git -c core.pager=cat commit -q -m "results: $label finished -- see RESULTS.md

Auto-recorded by the overnight pipeline. Numbers, LSCR statistics, deviations and
known blockers (GR00T experiment_cfg gap) are all in RESULTS.md." || true
    timeout 180 env -u GIT_SSH_COMMAND GIT_TERMINAL_PROMPT=0 \
      git -c core.sshCommand="$SSH" push git@github.com:wangrp1997/Bench2Dex.git main \
      >/tmp/pushout 2>&1
    if [ $? -eq 0 ]; then say "$label: pushed $(git rev-parse --short HEAD)"; else say "$label: PUSH FAILED -- $(tail -2 /tmp/pushout | tr '\n' ' ')"; fi
}

say "pusher started (ACT already pushed manually as 3655b01)"
done_gr00t=0; done_pi05=0; done_dp=0
while true; do
    sleep 120
    if [ "$done_dp" = 0 ] && family_done dp "dp_"; then
        say "DP family finished"; push_once "DP baseline"; done_dp=1
    fi
    if [ "$done_gr00t" = 0 ] && family_done gr00t "gr00t_"; then
        say "GR00T family finished (expected: blocked upstream)"; push_once "GR00T stage (blocked)"; done_gr00t=1
    fi
    if [ "$done_pi05" = 0 ] && family_done pi05 "pi05_"; then
        say "pi05 family finished"; push_once "pi0.5 baseline"; done_pi05=1
    fi
    if [ "$done_dp" = 1 ] && [ "$done_gr00t" = 1 ] && [ "$done_pi05" = 1 ]; then
        say "all families recorded; pusher exiting"; break
    fi
done
