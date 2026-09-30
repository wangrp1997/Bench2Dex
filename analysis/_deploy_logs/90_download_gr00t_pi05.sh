#!/bin/bash
# Download the GR00T_n15 and pi05 checkpoints for tasks 26/32/73 (6 parallel
# streams). Everything lands in /mnt/public/datasets/bench2dex/policy_ckpt via
# the ~/bench2dex/policy_ckpt symlink -- nothing on the /home volume.
export PATH=$HOME/.local/bin:$PATH
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
LOCAL=/home/wangrenpeng/bench2dex/policy_ckpt

for ALGO in gr00t_n15 pi05; do
  for T in 26 32 73; do
    (
      echo "=== $ALGO task $T start $(date -Is) ==="
      $NOPROXY modelscope download --repo-type model Bench2Dex/New_Policy \
        --local_dir "$LOCAL" \
        --include "$T/multi_iiwa7_with_sharpa/$ALGO/*" --max-workers 16
      echo "=== $ALGO task $T exit=$? $(date -Is) ==="
    ) &
    sleep 3
  done
done
wait
echo "=== ALL DONE $(date -Is) ==="
for ALGO in gr00t_n15 pi05; do for T in 26 32 73; do
  printf "  %-10s task %s: %s\n" "$ALGO" "$T" "$(du -sh $LOCAL/$T/multi_iiwa7_with_sharpa/$ALGO 2>/dev/null | cut -f1)"
done; done
