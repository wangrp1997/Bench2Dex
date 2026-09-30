#!/usr/bin/env python3
"""Make recorded policy rollouts replayable by the stock `replay.py`.

`run_policy.py --record-dir --record-all` writes qpos/objects/cameras but omits the
metadata that `replay.py` requires:
    meta/scene_file, meta/robot_key, meta/fps, meta/step_stride
This script copies the authoritative timing fields from an anchor demo episode (the
recorded fps/step_stride convention is identical: 20 Hz policy / 60 Hz physics) and
writes the scene/robot identity, in place.

Usage:
  python 100_patch_rollout_meta.py --dir <rollout_dir> --scene 26_canned_food_tray_line_arrangement \
      --robot multi_iiwa7_with_sharpa [--anchor <demo_episode.hdf5>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/wangrenpeng/bench2dex/Bench2Dex")
DATA = Path("/mnt/public/datasets/bench2dex/teleopdata/dataset")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--scene", required=True, help="scene stem, e.g. 26_canned_food_tray_line_arrangement")
    ap.add_argument("--robot", default="multi_iiwa7_with_sharpa")
    ap.add_argument("--anchor", default=None)
    args = ap.parse_args()

    import h5py
    sys.path.insert(0, str(REPO))
    from robots.active_dof_utils import get_active_dof_info
    adi = get_active_dof_info(args.robot)

    anchor = args.anchor
    if anchor is None:
        cands = sorted((DATA / args.scene / "replay-generalization").glob("episode_*.hdf5"))
        if not cands:
            print(f"   !! no anchor demo for {args.scene}"); return 1
        anchor = str(cands[0])

    anchor_names = None
    with h5py.File(anchor, "r") as f:
        timing = {k: f[f"meta/{k}"][()] for k in
                  ("fps", "step_stride", "physics_dt", "effective_fps") if f"meta/{k}" in f}
        if "robot" in f and "joint_names" in f["robot"]:
            anchor_names = [n.decode() if isinstance(n, bytes) else str(n)
                            for n in f["robot/joint_names"][:]]
    print(f"   anchor joint order: {len(anchor_names) if anchor_names else 0} names")
    scene_path = str(REPO / "scenes" / f"{args.scene}.yaml")
    print(f"   anchor timing: {timing}")

    files = sorted(Path(args.dir).glob("**/*.hdf5"))
    if not files:
        print(f"   !! no rollouts under {args.dir}"); return 1
    patched = 0
    for p in files:
        try:
            with h5py.File(p, "a") as f:
                if "robot" not in f or "qpos" not in f["robot"]:
                    print(f"   skip (no robot/qpos): {p.name}"); continue
                meta = f.require_group("meta")
                def put(k, v):
                    if k in meta:
                        del meta[k]
                    meta.create_dataset(k, data=v)
                for k, v in timing.items():
                    put(k, v)
                for k, v in (("robot_key", args.robot), ("scene_name", args.scene),
                             ("scene_file", scene_path)):
                    if k not in meta:
                        put(k, np.bytes_(v.encode()))
                # The recorder omits robot/joint_names, but replay.py needs it to build the
                # joint map. IMPORTANT: the recorded qpos follows the ARTICULATION's own joint
                # order, which is NOT `active_dof_utils.full_joint_names` (measured: the two
                # orders differ substantially, while demo and rollout first-frame qpos agree to
                # 0.0). Using full_joint_names here silently mislabels every joint and the
                # replayed robot lands in a nonsense configuration (observed: zero tactile
                # contact over a whole episode). So copy the order from an anchor demo of the
                # SAME robot, and verify the width.
                if "joint_names" not in f["robot"]:
                    if anchor_names is None or len(anchor_names) != f["robot/qpos"].shape[1]:
                        print(f"   !! {p.name}: no usable anchor joint order "
                              f"({None if anchor_names is None else len(anchor_names)} vs "
                              f"{f['robot/qpos'].shape[1]}), skipped")
                        continue
                    f["robot"].create_dataset(
                        "joint_names", data=np.asarray(anchor_names, dtype=object),
                        dtype=h5py.string_dtype())
                meta.attrs["patched_for_replay"] = True
            patched += 1
        except Exception as exc:  # noqa: BLE001
            print(f"   !! {p.name}: {type(exc).__name__}: {exc}")
    print(f"   patched {patched}/{len(files)} rollouts in {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
