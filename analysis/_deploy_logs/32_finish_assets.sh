#!/bin/bash
# Finalise the dex2bench_dataset move: the first rsync was interrupted, so some
# destination files kept a stale mtime (rsync reported only "t" differences, never
# "s"/"c"). Re-sync until the trees are byte-identical, then delete source + symlink.
set -uo pipefail
SRC=/home/wangrenpeng/bench2dex/dex2bench_dataset
DST=/mnt/public/datasets/bench2dex/dex2bench_dataset

echo "### $(date -Is) re-sync pass"
rsync -a --info=stats2 "$SRC/" "$DST/"
diff=$(rsync -an --itemize-changes "$SRC/" "$DST/")
if [ -n "$diff" ]; then
    echo "!!! still differs:"; echo "$diff" | head -10
    echo "!!! keeping source"; exit 1
fi
echo "### identical: $(find "$DST" -type f | wc -l) files"
rm -rf "$SRC"
ln -s "$DST" "$SRC"
echo "### OK  $SRC -> $DST"
ls -ld "$SRC"
echo "### done $(date -Is)"
