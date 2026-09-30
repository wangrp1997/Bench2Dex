#!/bin/bash
# Move the big Bench2Dex artefacts off /home onto the shared NFS share, then
# leave symlinks in ~/Projects so the hard-coded relative paths in the scene
# YAMLs (../../dex2bench_dataset/..., ../dex2bench_dataset/...) keep working.
#
# Safety: rsync first, verify with a dry-run itemised diff, and only delete the
# source after the copy is proven identical.
set -uo pipefail

SRC=/home/wangrenpeng/bench2dex
DST_D=/mnt/public/datasets/Bench2Dex
DST_M=/mnt/public/models/Bench2Dex
mkdir -p "$DST_D" "$DST_M"

move() {  # $1 = source abs path, $2 = dest abs path
    local s="$1" d="$2" name
    name=$(basename "$s")
    echo "### $(date -Is) moving $name"
    if [ -L "$s" ]; then echo "    already a symlink, skipping"; return 0; fi
    mkdir -p "$d"
    rsync -a --info=progress2 "$s/" "$d/" || { echo "!!! rsync failed for $name"; return 1; }
    local diff
    diff=$(rsync -an --itemize-changes "$s/" "$d/" | head -20)
    if [ -n "$diff" ]; then
        echo "!!! NOT IDENTICAL for $name, keeping source:"; echo "$diff"; return 1
    fi
    echo "    verified identical: $(find "$d" -type f | wc -l) files, $(du -sh "$d" | cut -f1)"
    rm -rf "$s"
    ln -s "$d" "$s"
    echo "    -> $s now points at $d"
}

move "$SRC/dex2bench_dataset" "$DST_D/dex2bench_dataset"
move "$SRC/teleopdata"        "$DST_D/teleopdata"
move "$SRC/policy_ckpt"       "$DST_M/policy_ckpt"

# Optional pointer so the group can find the project from /mnt/public/projects
if [ ! -e /mnt/public/projects/Bench2Dex ] && [ -d /mnt/public/projects ]; then
    ln -s /home/wangrenpeng/bench2dex/Bench2Dex /mnt/public/projects/Bench2Dex 2>/dev/null \
      && echo "### pointer created: /mnt/public/projects/Bench2Dex -> ~/Projects/Bench2Dex" \
      || echo "### (could not create the projects/ pointer, not fatal)"
fi

echo "### done $(date -Is)"
ls -la "$SRC" | grep -E "dex2bench_dataset|teleopdata|policy_ckpt"
