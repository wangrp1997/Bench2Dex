#!/usr/bin/env python3
"""Milestone-level labels for a SPEC-DRIVEN infeasibility certificate.

Reframing (replaces "will this episode fail?")
----------------------------------------------
The task's own executable stage predicates define a specification with a dependency order.
At every frame t let
    s(t) = the first stage, in that order, that is not yet LATCHED
and define the target
    y(t) = 1  iff  s(t) is never latched for the remainder of the episode

So the monitor's question is "will the CURRENTLY-NEXT specification milestone be achieved?",
which is
  * spec-anchored  -> the stage ids come from the scene YAML, so it applies to any of the
                      26 tasks rather than assuming a grasp-type failure;
  * per-frame      -> the target stage advances as milestones latch, unlike the constant
                      episode-level outcome label used before.

Output per rollout: npz with `latched` (T, n_stages) bool, `stage_ids`, `s_index` (T,),
`y_milestone` (T,), plus `n_stages`.

Usage: 119_milestone_labels.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/home/wangrenpeng/bench2dex/Bench2Dex")
ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
OUT = Path("/home/wangrenpeng/bench2dex/data/milestone_labels")
POLICY_STRIDE = 3
TASKS = {"26": "26_canned_food_tray_line_arrangement",
         "32": "32_baking_tray_prep_with_tools",
         "73": "73_jigsaw_puzzle_assembly"}
sys.path.insert(0, str(REPO))


def object_states_at(f, i):
    return {o: {"pose_world": np.asarray(f["objects"][o]["pose_world"][i], np.float64),
                "lin_vel_world": np.asarray(f["objects"][o]["lin_vel_world"][i], np.float64),
                "ang_vel_world": np.asarray(f["objects"][o]["ang_vel_world"][i], np.float64)}
            for o in f["objects"].keys()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max rollouts per outcome (0=all)")
    args = ap.parse_args()

    import h5py
    from benchmark.stage_tracker import StageTracker

    total = 0
    for tid, scene in TASKS.items():
        stages = yaml.safe_load(open(REPO / "scenes" / f"{scene}.yaml"))["metrics"]["stages"]
        ids = [s.get("id", f"stage_{i}") for i, s in enumerate(stages)]
        print(f"\n=== task {tid}: {len(ids)} stages {ids}")
        for outcome in ("success", "failure"):
            eps = sorted((ROLL / tid / outcome).glob("*.hdf5"))
            if args.limit:
                eps = eps[:args.limit]
            for p in eps:
                out = OUT / tid / outcome / (p.stem + ".npz")
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists():
                    total += 1
                    continue
                try:
                    with h5py.File(p, "r") as f:
                        T = int(f["meta/frame_count"][()])
                        tr = StageTracker(stages, dt=1 / 60)
                        lat = np.zeros((T, len(ids)), bool)
                        for i in range(T):
                            r = tr.update(object_states_at(f, i), sim_step=i * POLICY_STRIDE)
                            lat[i] = [bool(r.completed[sid]) for sid in ids]
                    # s(t): first not-yet-latched stage; y(t): that stage never latches later
                    final = lat[-1]
                    s_idx = np.full(T, len(ids), np.int64)      # n_stages == "all done"
                    y = np.zeros(T, np.int8)
                    for i in range(T):
                        rem = np.where(~lat[i])[0]
                        if len(rem) == 0:
                            s_idx[i] = len(ids)
                            y[i] = 0
                        else:
                            k = int(rem[0])
                            s_idx[i] = k
                            y[i] = 0 if final[k] else 1
                    np.savez_compressed(out, latched=lat, stage_ids=np.array(ids, dtype=object),
                                        s_index=s_idx, y_milestone=y, n_stages=len(ids))
                    total += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"   !! {p.name}: {type(exc).__name__}: {exc}")
        # quick label stats
        fs = sorted((OUT / tid).glob("*/*.npz"))
        if fs:
            pos = np.mean([np.load(p)["y_milestone"].mean() for p in fs])
            print(f"   {len(fs)} rollouts, mean y_milestone positive rate {pos:.3f}")
    print(f"\n   wrote {total} milestone label files -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
