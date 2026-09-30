#!/bin/bash
# Move the big Bench2Dex artefacts off /home onto the shared NFS share, into ONE
# top-level project dir named `bench2dex`, then leave symlinks in ~/Projects so
# the hard-coded relative paths in the scene YAMLs keep working:
#   scenes/*.yaml -> ../../dex2bench_dataset/...
#   robots/*.py   -> ../dex2bench_dataset/...
#
# Safety: rsync first, verify with a dry-run itemised diff, delete the source
# only after the copy is proven identical. Run detached (setsid) so an aborted
# tool call cannot SIGTERM it.
set -uo pipefail

SRC=/home/wangrenpeng/bench2dex
DEST=/mnt/public/datasets/bench2dex
mkdir -p "$DEST"

# discard the abandoned previous-layout partial copies
rm -rf /mnt/public/datasets/Bench2Dex /mnt/public/models/Bench2Dex

move() {  # $1 = source abs path
    local s="$1" name d
    name=$(basename "$s"); d="$DEST/$name"
    echo "### $(date -Is) moving $name -> $d"
    if [ -L "$s" ]; then echo "    already a symlink, skipping"; return 0; fi
    rsync -a --info=progress2 "$s/" "$d/" || { echo "!!! rsync FAILED for $name"; return 1; }
    local diff
    diff=$(rsync -an --itemize-changes "$s/" "$d/" | head -20)
    if [ -n "$diff" ]; then
        echo "!!! NOT IDENTICAL for $name -- keeping source, not deleting"; echo "$diff"; return 1
    fi
    echo "    verified identical: $(find "$d" -type f | wc -l) files, $(du -sh "$d" | cut -f1)"
    rm -rf "$s"
    ln -s "$d" "$s"
    echo "    OK  $s -> $d"
}

move "$SRC/dex2bench_dataset"
move "$SRC/teleopdata"
move "$SRC/policy_ckpt"

echo "### done $(date -Is)"
ls -la "$SRC" | grep -E "dex2bench_dataset|teleopdata|policy_ckpt"
echo "### content of $DEST:"
du -sh "$DEST"/*
