#!/usr/bin/env python3
"""Per-frame stage labels for RECORDED POLICY ROLLOUTS (successes and failures).

Same machinery as `90_extract_stage_labels.py` -- the scene's executable stage
predicates replayed through the benchmark's own `StageTracker` -- but adapted to the
rollout schema written by `utils/inference_recorder.py`, which has no `time/sim_step`
and no stored `metrics/episode/*`:
  * T is taken from `meta/frame_count`
  * the sim step is `i * POLICY_STRIDE` with POLICY_STRIDE = 3 (20 Hz policy / 60 Hz physics)
  * tactile is absent until the rollout has been through `replay.py --enable-tactile`,
    so contact counts are emitted only when the group exists

Output: <out>/<outcome>/<episode>.npz with chain_depth, lscr, current_lscr, final_lscr.

Usage:
  101_extract_rollout_labels.py --dir <rollouts_dir> --scene 26_canned_food_tray_line_arrangement
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/home/wangrenpeng/bench2dex/Bench2Dex")
POLICY_STRIDE = 3
sys.path.insert(0, str(REPO))


def load_stages(scene: str) -> list[dict]:
    doc = yaml.safe_load(open(REPO / "scenes" / f"{scene}.yaml"))
    return doc["metrics"]["stages"]


def object_states_at(f, i: int) -> dict:
    st = {}
    for oid in f["objects"].keys():
        g = f["objects"][oid]
        st[oid] = {
            "pose_world": np.asarray(g["pose_world"][i], dtype=np.float64),
            "lin_vel_world": np.asarray(g["lin_vel_world"][i], dtype=np.float64),
            "ang_vel_world": np.asarray(g["ang_vel_world"][i], dtype=np.float64),
        }
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="rollout dir, e.g. <OUT>/rollouts")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import h5py
    from benchmark.stage_tracker import StageTracker

    stages = load_stages(args.scene)
    print(f"   scene {args.scene}: {len(stages)} stages "
          f"{[s.get('id') for s in stages]}")
    root = Path(args.dir)
    out_root = Path(args.out) if args.out else Path("/home/wangrenpeng/bench2dex/data/rollout_labels")
    eps = sorted(root.glob("**/*.hdf5"))
    print(f"   {len(eps)} rollouts under {root}")
    done = 0
    for ep in eps:
        outcome = ep.parent.name            # 'success' or 'failure'
        outdir = out_root / args.scene / outcome
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / (ep.stem + ".npz")
        if out.exists():
            done += 1
            continue
        try:
            with h5py.File(ep, "r") as f:
                T = int(f["meta/frame_count"][()])
                have_tac = "tactile" in f.get("robot", {}) and \
                    len(f["robot/tactile"].get("contact_mask", {}).keys()) > 0
                tracker = StageTracker(stages, dt=1 / 60)
                lscr = np.zeros(T, np.float32)
                clscr = np.zeros(T, np.float32)
                dep = np.zeros(T, np.int16)
                for i in range(T):
                    r = tracker.update(object_states_at(f, i), sim_step=i * POLICY_STRIDE)
                    lscr[i] = r.latched_stage_completion_rate
                    clscr[i] = r.current_stage_completion_rate
                    dep[i] = r.latched_chain_depth
                extra = {}
                if have_tac:
                    cm = f["robot/tactile"]["contact_mask"]
                    sites = list(cm.keys())
                    con = np.zeros((T, len(sites)), np.int16)
                    for j, s in enumerate(sites):
                        a = np.asarray(cm[s][:], bool)
                        con[:, j] = a.reshape(len(a), -1).sum(1)
                    extra["contact"] = con
            np.savez_compressed(out, chain_depth=dep, lscr=lscr, current_lscr=clscr,
                                final_lscr=np.float32(lscr[-1]), outcome=outcome, **extra)
            done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"   !! {ep.name}: {type(exc).__name__}: {exc}")
    print(f"   wrote {done}/{len(eps)} label files -> {out_root / args.scene}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
