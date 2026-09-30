"""T10 (1): author controlled hand-configuration variants for the dissociation experiment.

replay.py drives the robot from robot/qpos AND makes all objects kinematic, so editing the
hand joints changes ONLY the hand while the object keeps the demo's exact pose. That gives a
controlled contact experiment with no policy and no physics drift.

Variants (Sharpa; flexion increases with closure, 0 = open):
  base     : unmodified demo
  A_noThumb: thumb joints -> 0 (thumb away). Fingers keep the demo grasp.
             -> HIGH contact amount, but thumb opposition = FALSE
  B_opp1   : thumb kept, all fingers except right_index -> 0
             -> LOW contact amount, but thumb opposition = TRUE
  B_opp2   : same but keep right_index AND right_middle
  B_opp3   : same but keep right_index, middle, ring
"""
import h5py, numpy as np, shutil, sys
from pathlib import Path

SRC = sorted(Path("/mnt/public/datasets/bench2dex/teleopdata/dataset/"
                  "26_canned_food_tray_line_arrangement/replay-generalization").glob("*.hdf5"))[0]
OUT = Path("/home/wangrenpeng/bench2dex/theory/variants")

L_THUMB = [18, 33, 50]
R_THUMB = [28, 48, 57]
L_FING = {"index": [14, 29, 39], "middle": [15, 30, 40], "ring": [17, 32, 42], "pinky": [21, 41, 49]}
R_FING = {"index": [24, 44, 51], "middle": [25, 45, 52], "ring": [27, 47, 54], "pinky": [36, 53, 56]}


def make(name, open_joints):
    """copy the demo, set the listed joint indices to 0 (fully open), keep qvel."""
    dst = OUT / f"{name}.hdf5"
    shutil.copy(SRC, dst)
    with h5py.File(dst, "r+") as f:
        q = np.asarray(f["robot/qpos"][:], np.float32).copy()
        before = q[:, open_joints].mean()
        q[:, open_joints] = 0.0
        f["robot/qpos"][...] = q
        v = np.asarray(f["robot/qvel"][:], np.float32).copy()
        v[:, open_joints] = 0.0
        f["robot/qvel"][...] = v
    print(f"   {name:12s} 打开 {len(open_joints):2d} 个关节 (均值 {before:+.3f} -> 0.000)")
    return dst


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"   源: {SRC.name}")
    make("base", [])
    # A: both thumbs away, fingers as in the demo
    make("A_noThumb", L_THUMB + R_THUMB)
    # B variants: thumbs kept, progressively more fingers opened
    make("B_opp1", L_FING["middle"] + L_FING["ring"] + L_FING["pinky"]
                  + R_FING["middle"] + R_FING["ring"] + R_FING["pinky"])
    make("B_opp2", L_FING["middle"] + L_FING["ring"] + L_FING["pinky"]
                  + R_FING["ring"] + R_FING["pinky"])
    make("B_opp3", L_FING["middle"] + L_FING["ring"] + L_FING["pinky"] + R_FING["pinky"])
