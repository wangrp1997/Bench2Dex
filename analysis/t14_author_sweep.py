"""T14: author a FAMILY of variants spanning the contact-amount range, for the sweep.

Two families, so that frames can be matched on contact amount while differing in the
relational condition (thumb opposition):

  A family (thumb opposition ABSENT): both thumbs opened; k fingers of the right hand kept
  B family (thumb opposition PRESENT): both thumbs kept;   k fingers of the right hand kept
  (the whole left hand opened in both, so the amount is controlled by the right hand only)

k = 1..4 gives four matched amount levels for the sweep.
"""
import h5py, numpy as np, shutil
from pathlib import Path

SRC = sorted(Path("/mnt/public/datasets/bench2dex/teleopdata/dataset/"
                  "26_canned_food_tray_line_arrangement/replay-generalization").glob("*.hdf5"))[0]
OUT = Path("/home/wangrenpeng/bench2dex/theory/variants")

L_THUMB = [18, 33, 50]
R_THUMB = [28, 48, 57]
L_F = {"index": [14, 29, 39], "middle": [15, 30, 40], "ring": [17, 32, 42], "pinky": [21, 41, 49]}
R_F = {"index": [24, 44, 51], "middle": [25, 45, 52], "ring": [27, 47, 54], "pinky": [36, 53, 56]}
L_ALLF = sum(L_F.values(), [])
# order in which right-hand fingers are added as k grows
R_ORDER = ["index", "middle", "ring", "pinky"]


def make(name, open_joints):
    dst = OUT / f"{name}.hdf5"
    shutil.copy(SRC, dst)
    with h5py.File(dst, "r+") as f:
        q = np.asarray(f["robot/qpos"][:], np.float32).copy()
        q[:, open_joints] = 0.0
        f["robot/qpos"][...] = q
        v = np.asarray(f["robot/qvel"][:], np.float32).copy()
        v[:, open_joints] = 0.0
        f["robot/qvel"][...] = v
    print(f"   生成 {name:10s} 打开 {len(open_joints):2d} 关节")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3, 4):
        kept = R_ORDER[:k]                       # right-hand fingers kept closed
        opened_R = sum((R_F[x] for x in R_ORDER[k:]), [])
        # A family: thumbs AWAY
        make(f"A_k{k}", L_THUMB + R_THUMB + L_ALLF + opened_R)
        # B family: thumbs KEPT
        make(f"B_k{k}", L_ALLF + opened_R)
