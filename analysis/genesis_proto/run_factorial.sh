#!/bin/bash
# 3 routes x 2 prediction targets x 3 seeds = 18 cells
set -uo pipefail
PY=/home/wangrenpeng/miniconda3/envs/genesis/bin/python
cd /home/wangrenpeng/bench2dex/genesis_proto
OUT=results; mkdir -p $OUT
for route in none aux test; do
  for target in raw task; do
    for seed in 0 1 2; do
      tag="${route}_${target}_s${seed}"
      [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; continue; }
      echo "[$(date +%H:%M:%S)] $tag"
      $PY p5_ppo_routing.py --route $route --target $target --iters 120 --horizon 64 \
          --seed $seed --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
      echo "   done $tag rc=$?"
    done
  done
done
echo "[$(date +%H:%M:%S)] 全部完成"
