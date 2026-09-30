#!/usr/bin/env python3
"""Retarget an episode onto another hand INSIDE the same arm family -- v2.

Why v1 failed (measured)
------------------------
`96_retarget_episode.py` used the repo's unified hand slots verbatim. The slot values
are raw joint angles in the SOURCE hand's convention, and each hand only populates some
of the 22 slots (rh56dfx 6, schunk 9, rh5dg2 13, wuji 20, shadow 22). Copying them onto
a higher-DoF target leaves the untouched joints at their initial (open) value, so the
hand never closes: replays produced **0 contact frames** (source: 16.7% / 31.3%).

v2 fix
------
Two changes, both aimed at the root cause (finger closure is not transferred):

1. **Per-slot range normalisation on each hand's own data.** For every hand we measure
   each slot's [min, max] over that hand's own episodes, so a value maps as
   `norm = (v - min_src) / (max_src - min_src)` then `v_dst = min_dst + norm * (max_dst - min_dst)`.
   This removes the convention/range mismatch between hands.
2. **Same-finger nearest-slot fill.** When the source hand has no joint for a slot the
   target needs, the normalised value is taken from the nearest available slot of the
   SAME finger (slots 0-3 thumb, 4-7 index, 8-11 middle, 12-15 ring, 16-19 pinky,
   20-21 wrist). This is what actually closes the target hand's fingers.

Arms keep the repo's IK path (`unified_to_joint_action`), which is exact within a family
(measured 0 IK failures).

Usage:
  python 97_retarget_v2.py --src <ep.hdf5> --dst-robot multi_ur5_wuji_with_flange \
      --out <out.hdf5> [--max-frames N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/wangrenpeng/bench2dex/Bench2Dex")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "policy" / "GR00T_XE"))

FINGER_OF_SLOT = ([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [-1] * 2)  # -1 = wrist/pad
DATA = Path("/mnt/public/datasets/bench2dex/teleopdata/dataset")


def load_slot_joint_maps() -> dict[str, dict[str, str]]:
    """hand_name -> {joint_name: [slot indices]} for right+left."""
    import yaml
    from xe_hand_mapping import robot_key_to_hand_name  # noqa: F401  (imported for parity)
    doc = yaml.safe_load(open(REPO / "policy" / "GR00T_XE" / "embodiment_mapping.yml"))
    hs = doc["hand_slots"]
    hand_names = sorted({k for side in ("right", "left") for e in hs[side]
                         for k in e if k not in ("slot", "desc")})
    out: dict[str, dict[str, int]] = {h: {} for h in hand_names}
    for side, off in (("right", 0), ("left", 22)):
        for e in hs[side]:
            slot = int(e["slot"]) + off
            for h in hand_names:
                jn = e.get(h)
                if jn:
                    out[h][jn] = slot
    return out


def slot_ranges(hand_name: str, slot_joint: dict[str, int], episodes: list[Path],
                max_frames_per_ep: int = 200) -> dict[int, tuple[float, float]]:
    """Per-slot [min,max] measured on this hand's OWN episodes."""
    import h5py
    acc: dict[int, list[np.ndarray]] = {}
    for p in episodes:
        try:
            with h5py.File(p, "r") as f:
                names = [n.decode() if isinstance(n, bytes) else str(n)
                         for n in f["robot/joint_names"][:]]
                idx = {n: i for i, n in enumerate(names)}
                q = np.asarray(f["robot/qpos"][:], np.float64)
                step = max(1, len(q) // max_frames_per_ep)
                q = q[::step]
                for jn, slot in slot_joint.items():
                    if jn in idx:
                        acc.setdefault(slot, []).append(q[:, idx[jn]])
        except Exception:
            continue
    out = {}
    for slot, arrs in acc.items():
        v = np.concatenate(arrs)
        out[slot] = (float(np.percentile(v, 1)), float(np.percentile(v, 99)))
    return out


def episodes_of(hand_name: str, slot_joint, limit: int = 4) -> list[Path]:
    """Find episodes whose robot_key maps to this hand name."""
    import h5py
    from xe_hand_mapping import robot_key_to_hand_name
    found = []
    for task_dir in sorted(DATA.glob("*/replay-generalization")):
        eps = sorted(task_dir.glob("episode_*.hdf5"))
        if not eps:
            continue
        try:
            with h5py.File(eps[0], "r") as f:
                rk = f["meta/robot_key"][()].decode()
            if robot_key_to_hand_name(rk) == hand_name:
                found.extend(eps[:2])
        except Exception:
            continue
        if len(found) >= limit:
            break
    return found[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst-robot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import h5py
    from ik_arm_converter import XEStateActionConverter, get_ik_failure_stats
    from robots.active_dof_utils import get_active_dof_info
    from xe_hand_mapping import robot_key_to_hand_name

    slot_joint = load_slot_joint_maps()

    with h5py.File(args.src, "r") as f:
        src_robot = f["meta/robot_key"][()].decode()
        qpos = np.asarray(f["robot/qpos"][:], np.float64)
        src_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["robot/joint_names"][:]]
    T = len(qpos) if not args.max_frames else min(len(qpos), args.max_frames)
    src_adi = get_active_dof_info(src_robot)
    dst_adi = get_active_dof_info(args.dst_robot)
    src_hand = robot_key_to_hand_name(src_robot)
    dst_hand = robot_key_to_hand_name(args.dst_robot)
    print(f"   src {src_robot}  hand={src_hand}  dof={src_adi.full_dof}")
    print(f"   dst {args.dst_robot}  hand={dst_hand}  dof={dst_adi.full_dof}")

    # --- per-slot ranges from each hand's OWN episodes ---
    sj_src = slot_joint.get(src_hand, {})
    sj_dst = slot_joint.get(dst_hand, {})
    rng_src = slot_ranges(src_hand, sj_src, episodes_of(src_hand, sj_src, 4))
    rng_dst = slot_ranges(dst_hand, sj_dst, episodes_of(dst_hand, sj_dst, 4))
    print(f"   slot ranges: src {len(rng_src)} slots, dst {len(rng_dst)} slots")
    print(f"      src covers slots {sorted(rng_src)}")
    print(f"      dst needs slots {sorted(rng_dst)}")

    # --- arm retargeting via the repo's converter (exact within a family) ---
    conv = XEStateActionConverter()
    before = dict(get_ik_failure_stats())
    q = np.zeros(dst_adi.full_dof, np.float64)
    arms = np.zeros((T, dst_adi.full_dof), np.float64)
    for t in range(T):
        u = conv.qpos_to_unified(qpos[t], src_robot)
        q = np.asarray(conv.unified_to_joint_action(u, args.dst_robot, q)).ravel()
        arms[t] = q
    after = get_ik_failure_stats()
    ik_fail = sum(v - before.get(k, 0) for k, v in after.items())
    print(f"   arm IK failures: {ik_fail}/{2*T}")

    # --- hand retargeting: normalised closure with same-finger nearest-slot fill ---
    dst_names = list(dst_adi.full_joint_names)
    didx = {n: i for i, n in enumerate(dst_names)}
    sidx = {n: i for i, n in enumerate(src_names)}
    out = arms.copy()
    filled = 0
    for jn, slot in sj_dst.items():
        if jn not in didx:
            continue
        finger = FINGER_OF_SLOT[slot % 22]
        # source value for this slot, else nearest available slot of the same finger
        use = slot if slot in rng_src and slot in sj_src.values() else None
        if use is None:
            cands = [s for s in rng_src if FINGER_OF_SLOT[s % 22] == finger]
            use = min(cands, key=lambda s: abs(s - slot)) if cands else None
        if use is None or jn not in sj_dst:
            continue
        # which source joint carries `use`
        src_joint = next((k for k, v in sj_src.items() if v == use), None)
        if src_joint is None or src_joint not in sidx:
            continue
        s_lo, s_hi = rng_src[use]
        d_lo, d_hi = rng_dst.get(slot, (0.0, 1.0))
        span_s = max(s_hi - s_lo, 1e-6)
        norm = np.clip((qpos[:T, sidx[src_joint]] - s_lo) / span_s, 0.0, 1.0)
        out[:, didx[jn]] = d_lo + norm * (d_hi - d_lo)
        filled += 1
    print(f"   hand joints driven from data: {filled}/{len(sj_dst)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.src, "r") as src, h5py.File(args.out, "w") as dst:
        for k in src.keys():
            if k != "robot":
                src.copy(k, dst)
        rg = dst.create_group("robot")
        rg.create_dataset("joint_names", data=np.array(dst_names, dtype=object),
                          dtype=h5py.special_dtype(vlen=str))
        rg.create_dataset("qpos", data=out.astype(np.float32), compression="gzip", compression_opts=4)
        rg.create_dataset("qvel", data=np.zeros_like(out, np.float32), compression="gzip", compression_opts=4)
        rg.create_dataset("qeffort", data=np.zeros_like(out, np.float32), compression="gzip", compression_opts=4)
        tg = rg.create_group("tactile")
        for s in ("tacmap", "contact_mask", "distance_along_normal_m"):
            tg.create_group(s)
        for key, val in (("meta/robot_key", args.dst_robot), ("meta/frame_count", T),
                         ("meta/action_dim", dst_adi.full_dof)):
            if key in dst:
                del dst[key]
            dst.create_dataset(key, data=val)
    print(f"   wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
