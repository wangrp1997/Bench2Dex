#!/bin/bash
# Wait for the running task-26 recording eval, then record rollouts for tasks 32 and 73
# with their correct per-task step budgets (matching the demonstration lengths).
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs

wait_done(){
  while pgrep -f '99_eval_gr00t_reco[r]d.sh' >/dev/null 2>&1; do sleep 60; done
}

run(){ # $1 task_id  $2 scene  $3 steps  $4 port
  echo "=== launching task $1 ($3 steps) $(date -Is) ==="
  "$L/99_eval_gr00t_record.sh" "$1" "$2" "$3" 2 "$4" 20 1 > "$L/105_eval$1.log" 2>&1
  echo "=== task $1 done rc=$? $(date -Is) ==="
}

wait_done
echo "task 26 recording eval finished $(date -Is)"
run 32 32_baking_tray_prep_with_tools 852 9400
wait_done
run 73 73_jigsaw_puzzle_assembly 1213 9500
echo "ALL RECORDING EVALS DONE $(date -Is)"
