#!/usr/bin/env python3
"""Rebuild the GR00T `experiment_cfg/metadata.json` that the released Bench2Dex
checkpoints are missing.

Why this is legitimate rather than fabrication
----------------------------------------------
`policy/GR00T_n15/scripts/gr00t_finetune.py` builds the training dataset with
`build_hdf5_dataset(...)`, and `Dex2BenchHDF5Dataset.__init__`
(`gr00t_hdf5_dataset.py:319`) calls `self._compute_statistics()` -- a streaming
pass over the same HDF5 episodes -- then `_build_metadata()` assembles a
`DatasetMetadata(statistics=...)`.  `src/gr00t/experiment/runner.py:76-97` writes
`{train_dataset.tag: train_dataset.metadata.model_dump(mode="json")}` to
`<output_dir>/experiment_cfg/metadata.json` **in TrainRunner.__init__**, i.e.
before a single training step runs.

So the metadata is a deterministic function of (data dir, camera map, transforms,
embodiment tag, active-DOF settings).  We reproduce exactly the arguments
`policy/GR00T_n15/train.sh` passes, using the repo's own code, and write the same
file the trainer would have written.

Caveat recorded in RESULTS.md: this is a RECONSTRUCTION. If the authors'
checkpoints were finetuned with a different data subset or config, the statistics
would differ and the numbers would be subtly off -- which is why we validate the
result against the paper's 17/12/10 rather than assuming it.

Usage: 80_rebuild_gr00t_experiment_cfg.py [task_id ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/home/wangrenpeng/bench2dex/Bench2Dex")
GR = REPO / "policy" / "GR00T_n15"
CKPT_ROOT = Path("/home/wangrenpeng/bench2dex/policy_ckpt")
DATA_ROOT = Path("/home/wangrenpeng/bench2dex/teleopdata/dataset")

sys.path.insert(0, str(GR))
sys.path.insert(0, str(GR / "src"))

# exactly train.sh:289-294 (CAMERA_MODE=4cam)
CAMERA_MAP = {
    "stereo_left": "cam_stereo_left",
    "stereo_right": "cam_stereo_right",
    "right_wrist": "cam_wrist_right",
    "left_wrist": "cam_wrist_left",
}
DATA_CONFIG = "policy.GR00T_n15.gr00t_dex2bench_config:Dex2BenchGR00TDataConfig"
EMBODIMENT_TAG = "new_embodiment"        # train.sh:328
ROBOT_KEY = "multi_iiwa7_with_sharpa"


def scene_prompt(task: str) -> str:
    """train.sh:174-183 -- `description:` from scenes/<task>.yaml."""
    y = REPO / "scenes" / f"{task}.yaml"
    if y.is_file():
        for line in y.read_text().splitlines():
            if line.startswith("description:"):
                return line.split(":", 1)[1].strip().strip('"')
    return "perform task: " + task.replace("_", " ")


def find_dataset_dir(task: str) -> Path:
    """train.sh:150-154 -- <DATA_ROOT>/<TASK_NUM>_*/replay-generalization.

    TASK_NUM is just the numeric prefix ("26"), not the full scene name.
    """
    num = task.split("_")[0]
    cands = sorted(DATA_ROOT.glob(f"{num}_*/replay-generalization"))
    if not cands:
        raise SystemExit(f"no replay-generalization dir for task {task}")
    return cands[0]


def main(tasks: list[str]) -> int:
    from gr00t.experiment.data_config import load_data_config
    from gr00t_hdf5_dataset import build_hdf5_dataset

    rc = 0
    for task in tasks:
        t0 = time.time()
        ds_dir = find_dataset_dir(task)
        num = task.split("_")[0]
        ckpt = CKPT_ROOT / num / "multi_iiwa7_with_sharpa" / "gr00t_n15"
        print(f"\n=== task {task}")
        print(f"    data   : {ds_dir}")
        print(f"    ckpt   : {ckpt}")
        print(f"    prompt : {scene_prompt(task)[:90]}")
        if not ckpt.is_dir():
            print("    !! checkpoint dir missing, skipping")
            rc = 1
            continue
        try:
            cfg = load_data_config(DATA_CONFIG)
            ds = build_hdf5_dataset(
                input_dir=ds_dir,
                camera_map=CAMERA_MAP,
                modality_configs=cfg.modality_config(),
                transforms=cfg.transform(),
                embodiment_tag=EMBODIMENT_TAG,
                prompt=scene_prompt(task),
                use_active_dof=True,          # train.sh:317 --hdf5-use-active-dof
                robot_key=ROBOT_KEY,
                state_dim=None,
                action_dim=None,
                truncate_at_homing=True,      # train.sh:342 --hdf5-truncate-at-homing
            )
        except Exception as exc:              # noqa: BLE001
            import traceback
            print(f"    !! dataset construction FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            rc = 1
            continue

        meta = {ds.tag: ds.metadata.model_dump(mode="json")}
        out_dir = ckpt / "experiment_cfg"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "metadata.json"
        with open(out, "w") as fh:
            json.dump(meta, fh, indent=4)

        stat = ds.metadata.statistics
        st_keys = sorted((stat.state if isinstance(stat.state, dict) else stat.state.model_dump()).keys())
        ac_keys = sorted((stat.action if isinstance(stat.action, dict) else stat.action.model_dump()).keys())
        print(f"    tag   : {ds.tag}")
        print(f"    samples: {len(ds)}")
        print(f"    wrote : {out}  ({out.stat().st_size} bytes)  in {time.time()-t0:.0f}s")
        print(f"    state keys ({len(st_keys)}): {st_keys[:6]}")
        print(f"    action keys ({len(ac_keys)}): {ac_keys[:6]}")
    return rc


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        args = ["26_canned_food_tray_line_arrangement",
                "32_baking_tray_prep_with_tools",
                "73_jigsaw_puzzle_assembly"]
    # allow short forms 26/32/73
    full = []
    for a in args:
        if a in ("26", "32", "73"):
            full.append({"26": "26_canned_food_tray_line_arrangement",
                         "32": "32_baking_tray_prep_with_tools",
                         "73": "73_jigsaw_puzzle_assembly"}[a])
        else:
            full.append(a)
    sys.exit(main(full))
