#!/bin/bash
# Download exactly ONE episode per task so we can read meta/robot_key and build the
# task -> embodiment mapping. 27 tasks x ~180 MB ~= 4.9 GB (vs 19 GB per full task).
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
LOCAL=/mnt/public/datasets/bench2dex/teleopdata
TASKS="$@"
for T in $TASKS; do
    $NOPROXY modelscope download --repo-type dataset Bench2Dex/teleopdata \
      --local_dir "$LOCAL" \
      --include "dataset/$T/replay-generalization/episode_000000.hdf5" \
      --max-workers 4 >/dev/null 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 6 ]; do sleep 2; done
done
wait
echo "ALL DONE $(date -Is)"
