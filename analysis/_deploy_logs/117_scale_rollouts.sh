#!/bin/bash
# Scale IN-DOMAIN rollout collection: 80 more episodes per task (21..100), so the failure
# predictor gets ~300 in-domain rollouts instead of 60. This is the clean test of whether
# the hand-crafted contact statistic can be beaten by a learned model given enough
# in-domain data (adding out-of-domain demos was shown to hurt).
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
run(){ # task scene steps port start num
  echo "=== task $1 episodes $5..$(( $5 + $6 - 1 )) $(date -Is) ==="
  "$L/99_eval_gr00t_record.sh" "$1" "$2" "$3" 2 "$4" "$6" "$5" > "$L/117_eval$1.log" 2>&1
  echo "=== task $1 done rc=$? $(date -Is) ==="
}
run 26 26_canned_food_tray_line_arrangement 871 9600 21 80
run 32 32_baking_tray_prep_with_tools       852 9601 21 80
run 73 73_jigsaw_puzzle_assembly           1213 9602 21 80
echo "ALL SCALED EVALS DONE $(date -Is)"
