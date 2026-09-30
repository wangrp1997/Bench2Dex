#!/bin/bash
# Download the DP checkpoints for tasks 26/32/73 in PARALLEL.
# NOTE: policy_ckpt is a symlink to /mnt/public/datasets/bench2dex/policy_ckpt,
# so nothing lands on the /home volume.
export PATH=$HOME/.local/bin:$PATH
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
LOCAL=/home/wangrenpeng/bench2dex/policy_ckpt

for T in 26 32 73; do
  (
    echo "=== DP task $T start $(date -Is) ==="
    $NOPROXY modelscope download --repo-type model Bench2Dex/New_Policy \
      --local_dir "$LOCAL" \
      --include "$T/multi_iiwa7_with_sharpa/dp/*" --max-workers 16
    echo "=== DP task $T exit=$? $(date -Is) ==="
  ) &
done
wait
echo "=== ALL DP DOWNLOADS DONE $(date -Is) ==="
du -sh "$LOCAL"/*/multi_iiwa7_with_sharpa/dp 2>/dev/null
