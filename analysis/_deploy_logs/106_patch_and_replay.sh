#!/bin/bash
# Attach tactile to recorded policy rollouts:
#   1) patch the metadata replay.py needs (and the correct articulation joint order)
#   2) replay each rollout with --enable-tactile, N in parallel
#
# Usage: 106_patch_and_replay.sh <TASK_ID> <SCENE> <ROLLOUTS_DIR> [WORKERS]
set -uo pipefail
TASK_ID="${1:?}"; SCENE="${2:?}"; RD="${3:?}"; W="${4:-3}"
ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
L=$ROOT/_deploy_logs
B2DPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
DEST=/mnt/public/datasets/bench2dex/rollouts/${TASK_ID}
KIT="--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg --/exts/omni.kit.registry.nucleus/registries/2/optional=true"

export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export LD_LIBRARY_PATH=$ROOT/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=2
export http_proxy=http://host.docker.internal:1081
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
export NO_PROXY=$no_proxy
mkdir -p /tmp/empty_reg "$DEST"

echo "=== task $TASK_ID: patching meta in $RD $(date -Is) ==="
"$B2DPY" "$L/100_patch_rollout_meta.py" --dir "$RD" --scene "$SCENE" 2>&1 | tail -3

cd "$REPO" || exit 1
mapfile -t FILES < <(find "$RD" -name "*.hdf5" | sort)
echo "=== replaying ${#FILES[@]} rollouts with --enable-tactile, $W workers $(date -Is) ==="

replay_one() {
    local src="$1"
    local outcome; outcome=$(basename "$(dirname "$src")")
    local out="$DEST/$outcome/$(basename "$src")"
    mkdir -p "$DEST/$outcome"
    if [ -f "$out" ]; then echo "   skip (exists): $outcome/$(basename "$src")"; return 0; fi
    "$B2DPY" replay.py --hdf5 "$src" --enable-tactile --output "$out" \
        --kit_args "$KIT" --headless > "$L/106_replay_${TASK_ID}_$(basename "$src" .hdf5).log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "   !! replay failed rc=$rc: $outcome/$(basename "$src")"; fi
    return $rc
}
export -f replay_one
export B2DPY REPO L KIT DEST TASK_ID

printf '%s\n' "${FILES[@]}" | xargs -P "$W" -I{} bash -c 'replay_one "$@"' _ {}
echo "=== task $TASK_ID done $(date -Is) ==="
find "$DEST" -name "*.hdf5" | wc -l | sed 's/^/   rollouts with tactile: /'
