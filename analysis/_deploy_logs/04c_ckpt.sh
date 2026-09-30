#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
for T in 26 32 73; do
  echo "=== ckpt $T start $(date -Is) ==="
  $NOPROXY modelscope download --repo-type model Bench2Dex/New_Policy \
    --local_dir /home/wangrenpeng/bench2dex/policy_ckpt \
    --include "$T/multi_iiwa7_with_sharpa/act_active/*" --max-workers 16
  echo "=== ckpt $T exit=$? $(date -Is) ==="
done
