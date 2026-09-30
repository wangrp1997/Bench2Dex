#!/bin/bash
# Download N episodes for several tasks spanning DIFFERENT hand morphologies, for the
# cross-hand tactile representation study. Chosen for morphological/sensor diversity:
#   64 multi_panda_with_allegro          8 sites, 4-finger (no pinky)
#   09 multi_xarm7_with_leap             8 sites, different site ORDER (index first)
#   43 multi_ur5_shadow_hand             10 sites, abbreviated names (FFJ1/MFJ1/RFJ1/LFJ1)
#   21 multi_ur5_wuji                    10 sites, _J4 suffix
#   22 multi_panda_with_orca             10 sites, joint-name suffixes (dip/pip)
#   03 multi_xarm7_with_ability          10 sites, q2 suffix
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
LOCAL=/mnt/public/datasets/bench2dex/teleopdata
N=${N:-15}
for T in 64_sports_ball_cup_sort 09_cleaner_moisturizer_box_loading 43_fridge_fruit_shelf_sorting \
         21_condiment_box_loading 22_tool_box_loading 03_wine_glass_plate_balance; do
    INC=""
    for i in $(seq 0 $((N-1))); do
        f=$(printf "episode_%06d.hdf5" $i)
        INC="$INC --include dataset/$T/replay-generalization/$f"
    done
    # shellcheck disable=SC2086
    $NOPROXY modelscope download --repo-type dataset Bench2Dex/teleopdata \
      --local_dir "$LOCAL" $INC --max-workers 8 >/dev/null 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 3; done
done
wait
echo "ALL DONE $(date -Is)"
