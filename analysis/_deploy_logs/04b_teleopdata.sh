#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
for T in 26_canned_food_tray_line_arrangement 32_baking_tray_prep_with_tools 73_jigsaw_puzzle_assembly; do
  echo "=== anchor $T start $(date -Is) ==="
  $NOPROXY modelscope download --repo-type dataset Bench2Dex/teleopdata \
    --local_dir /home/wangrenpeng/bench2dex/teleopdata \
    --include "dataset/$T/replay-generalization/*" --max-workers 16
  echo "=== anchor $T exit=$? $(date -Is) ==="
done
