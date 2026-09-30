#!/usr/bin/env python3
"""Build compact per-frame features for the modality probes.

Raw size is prohibitive: 300 episodes x ~700 frames x 10 tactile sites x 240x240
uint8 ~= 1.2e11 bytes. So we subsample to every STRIDE-th frame and downsample:
  tac  : tacmap  240x240 -> 24x24  per site  (uint8)
  rgb  : cam_chest 480x640x3 -> 64x64 grayscale (uint8)
  prop : qpos   58-dim (float32)

Labels come from `data/stage_labels/<task>/episode_XXXXXX.npz` (chain_depth per
frame = the simulator's own latched chain depth, i.e. the contact phase).

Output: data/probe_features/<task>/episode_XXXXXX.npz
  tac [nF,10,24,24] u8 | rgb [nF,64,64] u8 | prop [nF,58] f32 | phase [nF] i16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

DATA = Path("/mnt/public/datasets/bench2dex/teleopdata/dataset")
LABELS = Path("/home/wangrenpeng/bench2dex/data/stage_labels")
OUT = Path("/home/wangrenpeng/bench2dex/data/probe_features")
STRIDE = 5
TAC_GRID = 24
RGB_GRID = 64


def downsample_u8(img: np.ndarray, g: int) -> np.ndarray:
    """Block-max downsample a 2-D uint8 image to g x g."""
    h, w = img.shape
    bh, bw = h // g, w // g
    if bh == 0 or bw == 0:
        return np.zeros((g, g), np.uint8)
    crop = img[: bh * g, : bw * g].reshape(g, bh, g, bw)
    return crop.max(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=[
        "26_canned_food_tray_line_arrangement",
        "32_baking_tray_prep_with_tools",
        "73_jigsaw_puzzle_assembly"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import h5py

    for task in args.tasks:
        lab_dir = LABELS / task
        if not lab_dir.is_dir():
            print(f"!! no labels for {task}, run 90_extract_stage_labels.py first")
            continue
        eps = sorted(lab_dir.glob("episode_*.npz"))
        if args.limit:
            eps = eps[: args.limit]
        outdir = OUT / task
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {task}: {len(eps)} episodes")
        done = 0
        for lp in eps:
            out = outdir / lp.name
            if out.exists():
                done += 1
                continue
            h5 = DATA / task / "replay-generalization" / (lp.stem + ".hdf5")
            if not h5.exists():
                print(f"   !! missing hdf5 for {lp.stem}")
                continue
            try:
                lab = np.load(lp)
                phase = lab["chain_depth"]
                with h5py.File(h5, "r") as f:
                    T = int(phase.shape[0])
                    idx = np.arange(0, T, STRIDE)
                    sites = list(f["robot"]["tactile"]["tacmap"].keys())
                    tac = np.zeros((len(idx), len(sites), TAC_GRID, TAC_GRID), np.uint8)
                    for si, s in enumerate(sites):
                        d = f["robot"]["tactile"]["tacmap"][s]        # (T,240,240) u8
                        for j, i in enumerate(idx):
                            tac[j, si] = downsample_u8(np.asarray(d[i]), TAC_GRID)
                    rgbd = f["cameras"]["cam_chest"]["rgb"]           # (T,) object
                    rgb = np.zeros((len(idx), RGB_GRID, RGB_GRID), np.uint8)
                    for j, i in enumerate(idx):
                        raw = np.asarray(rgbd[i])
                        if raw.ndim == 1:
                            # stored as JPEG bytes (starts FF D8 FF E0), not raw pixels
                            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                            im = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr is not None
                                  else np.zeros((480, 640), np.uint8))
                        elif raw.ndim == 3:
                            im = raw[..., :3].mean(axis=2).astype(np.uint8)
                        else:
                            im = raw.astype(np.uint8)
                        rgb[j] = downsample_u8(im, RGB_GRID)
                    prop = np.asarray(f["robot"]["qpos"][:], np.float32)[idx]
                np.savez_compressed(out, tac=tac, rgb=rgb, prop=prop,
                                    phase=np.asarray(phase[idx], np.int16))
                done += 1
                if done % 20 == 0:
                    print(f"   {done}/{len(eps)}")
            except Exception as exc:  # noqa: BLE001
                print(f"   !! {lp.stem}: {type(exc).__name__}: {exc}")
        print(f"   wrote {done} feature files -> {outdir}")
        sz = sum(p.stat().st_size for p in outdir.glob("*.npz"))
        print(f"   total {sz/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
