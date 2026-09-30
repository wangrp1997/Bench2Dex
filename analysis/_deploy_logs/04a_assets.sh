#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
echo "=== assets start $(date -Is) ==="
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
  modelscope download --repo-type dataset Bench2Dex/Bench2Dex \
  --local_dir /home/wangrenpeng/bench2dex/dex2bench_dataset --max-workers 32
echo "=== assets exit=$? $(date -Is) ==="
