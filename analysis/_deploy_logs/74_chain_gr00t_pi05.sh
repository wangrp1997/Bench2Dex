#!/bin/bash
# Stage 3+4 of the overnight pipeline: after the DP evaluations finish, run
# GR00T N1.5, then pi0.5, each on tasks 26/32/73 (none channel, 50 episodes).
# Every stage regenerates RESULTS.md; failures are recorded, not fatal.
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
PIPE_LOG=$L/74_pipeline.log
say(){ echo "### $(date -Is) $*" | tee -a "$PIPE_LOG"; }

# Wait for DP to actually START first (71_chain launches it after ACT), then for
# it to END. Checking only "session absent" would race: DP has not started yet.
say "chain started; waiting for DP to start ..."
for _i in $(seq 1 240); do
    if tmux has-session -t dp26 2>/dev/null || tmux has-session -t dp32 2>/dev/null || tmux has-session -t dp73 2>/dev/null; then
        say "DP detected as started"; break
    fi
    if ls -d "$ROOT"/output/metric/dp_26_* >/dev/null 2>&1; then
        say "DP output detected"; break
    fi
    sleep 60
done
say "waiting for DP sessions to end ..."
while tmux has-session -t dp26 2>/dev/null || tmux has-session -t dp32 2>/dev/null || tmux has-session -t dp73 2>/dev/null; do
    sleep 60
done
say "DP finished"
python3 $L/99_make_report.py >> "$PIPE_LOG" 2>&1

# ── helper: wait for a family's tmux sessions to end ───────────────────────
wait_family() {  # $1 = prefix
    while tmux has-session -t "$1"26 2>/dev/null || tmux has-session -t "$1"32 2>/dev/null || tmux has-session -t "$1"73 2>/dev/null; do
        sleep 60
    done
}

# ── helper: are the checkpoints for a family complete? ─────────────────────
weights_ready() {  # $1 = algo dir name
    local A="$1" bad=0
    for T in 26 32 73; do
        local d="$ROOT/policy_ckpt/$T/multi_iiwa7_with_sharpa/$A"
        if [ ! -d "$d" ] || [ -n "$(find "$d" -name '*.incomplete' 2>/dev/null)" ] || [ -z "$(find "$d" -type f 2>/dev/null)" ]; then
            echo "    weights not ready: $A task $T"
            bad=1
        fi
    done
    return $bad
}

launch_family() {  # $1 = tag(gr00t|pi05) $2 = script $3 = port base
    local TAG="$1" SCRIPT="$2" PORT="$3"
    local SCENE_26=26_canned_food_tray_line_arrangement
    local SCENE_32=32_baking_tray_prep_with_tools
    local SCENE_73=73_jigsaw_puzzle_assembly
    local STEPS_26=871 STEPS_32=852 STEPS_73=1213
    for T in 26 32 73; do
        eval "scene=\$SCENE_$T"; eval "steps=\$STEPS_$T"
        local port=$PORT
        tmux new-session -d -s "$TAG$T" \
          "cd $REPO && $SCRIPT $T $scene $steps 2 $port 2>&1 | tee $L/${TAG}_${T}_${port}.log"
        say "launched tmux $TAG$T (port=$port scene=$scene steps=$steps)"
        PORT=$((PORT+1))
        sleep 45
    done
}

# ═══ STAGE 3: GR00T N1.5 ═══════════════════════════════════════════════════
say "STAGE 3 GR00T: waiting up to 90 min for the weights to finish downloading"
for i in $(seq 1 90); do weights_ready gr00t_n15 && break; sleep 60; done
if weights_ready gr00t_n15; then
    say "GR00T weights ready"
    launch_family gr00t $L/72_eval_gr00t.sh 9200
    wait_family gr00t
    say "GR00T finished"
else
    say "GR00T weights never became ready -- SKIPPING GR00T"
fi
python3 $L/99_make_report.py >> "$PIPE_LOG" 2>&1

# ═══ STAGE 4: pi0.5 ════════════════════════════════════════════════════════
say "STAGE 4 pi05: waiting up to 120 min for the weights to finish downloading"
for i in $(seq 1 120); do weights_ready pi05 && break; sleep 60; done
if weights_ready pi05; then
    say "pi05 weights ready"
    launch_family pi05 $L/73_eval_pi05.sh 9300
    wait_family pi05
    say "pi05 finished"
else
    say "pi05 weights never became ready -- SKIPPING pi05"
fi
python3 $L/99_make_report.py >> "$PIPE_LOG" 2>&1

say "PIPELINE COMPLETE -- final report regenerated"
touch $L/PIPELINE_DONE
