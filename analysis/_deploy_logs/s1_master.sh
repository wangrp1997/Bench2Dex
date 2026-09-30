#!/bin/bash
# Master chain for the dual-modality study (simulation arm):
#   stage 1: run the policy with the patched recorder -> rollouts carrying REAL joint torques
#   stage 2: replay the same rollouts with --enable-tactile -> rollouts carrying tactile
#   stage 3: merge the torques into the replayed files (same qpos trajectory => same frames)
# Run sequentially to avoid the minute-resolution output-dir collision found earlier.
set -uo pipefail
ROOT=/home/wangrenpeng/bench2dex; L=$ROOT/_deploy_logs
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
LOG=$L/s1_master.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A SCENE=(
 [26]=26_canned_food_tray_line_arrangement
 [32]=32_baking_tray_prep_with_tools
 [73]=73_jigsaw_puzzle_assembly
)
PORT=9800
for T in 26 32 73; do
  say "=== stage1: policy rollouts with qeffort, task $T ==="
  bash $L/99_eval_gr00t_record.sh "$T" "${SCENE[$T]}" 871 2 "$PORT" 50 1 >> "$LOG" 2>&1
  say "task $T stage1 done rc=$?"
  PORT=$((PORT+1))
  sleep 20
done
say "=== stage1 全部完成 ==="
# collect the freshly written rollout dirs (newest per task)
"$B2DPY" - <<'PY' >> "$LOG" 2>&1
import glob, os, json
out={}
for t in ("26","32","73"):
    ds=[d for d in glob.glob(f"/home/wangrenpeng/bench2dex/output/metric/gr00t_{t}_*") if os.path.isdir(os.path.join(d,"rollouts"))]
    ds.sort(key=os.path.getmtime)
    out[t]=ds[-1] if ds else None
json.dump(out, open("/home/wangrenpeng/bench2dex/_deploy_logs/s1_dirs.json","w"), indent=1)
print(json.dumps(out, indent=1))
PY
say "=== stage2: replay with tactile ==="
"$B2DPY" - <<'PY' >> "$LOG" 2>&1
import json, subprocess, os
dirs=json.load(open("/home/wangrenpeng/bench2dex/_deploy_logs/s1_dirs.json"))
scenes={"26":"26_canned_food_tray_line_arrangement","32":"32_baking_tray_prep_with_tools","73":"73_jigsaw_puzzle_assembly"}
for t,d in dirs.items():
    if not d: print(f"task {t}: 没有 rollout 目录"); continue
    rd=os.path.join(d,"rollouts")
    print(f"replaying task {t}: {rd}")
    subprocess.run(["bash","/home/wangrenpeng/bench2dex/_deploy_logs/106_patch_and_replay.sh",
                    t, scenes[t], rd, "3"], check=False)
PY
say "=== 全部完成 ==="
