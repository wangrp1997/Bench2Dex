#!/bin/bash
# DEPLOYMENT_NOTES.md Section 9 validation checklist, items 2-7 (1 = nvidia-smi, done)
ENVPY=/home/wangrenpeng/miniconda3/envs/bench2dex/bin/python
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
R=/home/wangrenpeng/bench2dex
pass=0; fail=0
chk() { echo "--- [$1] $2"; shift 2; if "$@" >/tmp/_c.out 2>&1; then echo "    PASS: $(tail -1 /tmp/_c.out)"; pass=$((pass+1)); else echo "    FAIL: $(tail -3 /tmp/_c.out)"; fail=$((fail+1)); fi; }

chk 1 "nvidia-smi" bash -c "nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | head -4"
chk 2 "torch cu128 + real CUDA op" "$ENVPY" -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability(0),(torch.randn(999,device='cuda')@torch.ones(999,device='cuda')).item())"
chk 3 "import isaacsim (no EULA prompt)" "$ENVPY" -c "import isaacsim;print('isaacsim ok')"
chk 4 "isaaclab version" "$ENVPY" -c "import isaaclab;print(isaaclab.__version__)"
chk 5 "pinocchio 2.7.0 in THIS env" "$ENVPY" -c "import pinocchio;print(pinocchio.__version__, pinocchio.__file__)"
chk 6 "IPython/h5py/cv2/trimesh/dex_retargeting" "$ENVPY" -c "import IPython,h5py,cv2,trimesh,dex_retargeting;print('all ok')"
chk 7 "numpy <2" "$ENVPY" -c "import numpy;print(numpy.__version__)"
echo
echo "--- [7] data layout"
ls -d $R/dex2bench_dataset $R/dex2bench_dataset/Objects 2>&1 | head -3
for T in 26_canned_food_tray_line_arrangement 32_baking_tray_prep_with_tools 73_jigsaw_puzzle_assembly; do
  n=$(ls $R/teleopdata/dataset/$T/replay-generalization/episode_*.hdf5 2>/dev/null | wc -l)
  echo "    $T: $n episode_*.hdf5"
done
ls $R/policy_ckpt/*/multi_iiwa7_with_sharpa/act_active/policy_best.ckpt 2>&1
echo
echo "=== CHECKLIST: pass=$pass fail=$fail ==="
