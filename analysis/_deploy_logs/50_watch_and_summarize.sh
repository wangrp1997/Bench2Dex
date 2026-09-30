#!/bin/bash
# Wait for the three ACT evaluations (tmux sessions eval26/eval32/eval73) to end,
# then aggregate their per_episode.jsonl into one summary.
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
OUT=/home/wangrenpeng/bench2dex/output/metric
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python

echo "### watcher start $(date -Is)"
while tmux has-session -t eval26 2>/dev/null || tmux has-session -t eval32 2>/dev/null || tmux has-session -t eval73 2>/dev/null; do
    sleep 60
done
echo "### all three sessions ended $(date -Is)"

"$ENVPY" - <<'PY' | tee /home/wangrenpeng/bench2dex/_deploy_logs/50_act_summary.txt
import json, glob, os, statistics as st

TASKS = {
    "26_canned_food_tray_line_arrangement": "26",
    "32_baking_tray_prep_with_tools": "32",
    "73_jigsaw_puzzle_assembly": "73",
}
OUT = "/home/wangrenpeng/bench2dex/output/metric"
PAPER = {"26": 17, "32": None, "73": None}   # paper: task26 ACT None = 17/50

print("=" * 96)
print("Bench2Dex | ACT | multi_iiwa7_with_sharpa | generalization-profile none | 50 episodes/seed 100000000")
print("=" * 96)
for scene, tid in TASKS.items():
    dirs = sorted(glob.glob(f"{OUT}/{tid}_iiwa7_sharpa_*_none_*"), key=os.path.getmtime)
    if not dirs:
        print(f"[{tid}] no output dir yet"); continue
    f = os.path.join(dirs[-1], "per_episode.jsonl")
    if not os.path.exists(f):
        print(f"[{tid}] {dirs[-1]} has no per_episode.jsonl"); continue
    eps = [json.loads(l) for l in open(f) if l.strip()]
    n = len(eps)
    succ = sum(1 for e in eps if e.get("stable_success") or e.get("success"))
    lscr = [e.get("latched_stage_completion_rate") for e in eps if e.get("latched_stage_completion_rate") is not None]
    scr  = [e.get("stage_completion_rate") for e in eps if e.get("stage_completion_rate") is not None]
    hold = [e.get("success_hold_s", 0.0) for e in eps]
    print(f"\n[{tid}] {os.path.basename(dirs[-1])}")
    print(f"     episodes={n}   stable_success={succ}/{n} = {100.0*succ/max(n,1):.1f}%", end="")
    if PAPER.get(tid):
        print(f"    (paper: {PAPER[tid]}/50 = {100*PAPER[tid]/50:.0f}%)")
    else:
        print()
    if lscr:
        print(f"     LSCR      mean={st.mean(lscr):.3f}  median={st.median(lscr):.3f}  min={min(lscr):.2f}  max={max(lscr):.2f}")
    if scr:
        print(f"     stage_rate mean={st.mean(scr):.3f}")
    zeros = sum(1 for v in lscr if v == 0)
    print(f"     LSCR==0 episodes: {zeros}/{len(lscr)}   (all-zero would mean a Pitfall 12/13 bug)")
    print(f"     success_hold_s mean={st.mean(hold):.2f}")
print("\n" + "=" * 96)
print("per-episode files:")
for scene, tid in TASKS.items():
    for d in sorted(glob.glob(f"{OUT}/{tid}_iiwa7_sharpa_*_none_*")):
        print("   ", d)
PY
echo "### summary written $(date -Is)"
