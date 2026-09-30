#!/usr/bin/env python3
"""Map per-hand tactile SITE NAMES onto a shared semantic slot space.

Motivation (measured, not assumed): the tactile forks in this repo embed sites by
POSITION -- `nn.Embedding(num_sites, D)` -- and hard-fail when the count differs
(`tactile_token_encoder.py:47`). Across the 12 Bench2Dex embodiments the counts are
8 (LEAP, Allegro) or 10 (the rest), the ORDER differs (LEAP lists index first,
Sharpa lists thumb first), and the suffixes are sensor-specific
(`elastomer` / `J4` / `dip` / `pip` / `touch` / `pad` / `FFJ1` / `q2` ...).

The only thing shared across hands is the SEMANTIC identity of a site:
    hand side (left/right) x finger role (thumb/index/middle/ring/pinky)
so we canonicalize onto that fixed 10-slot space and carry a presence mask for the
slots a given hand lacks (e.g. Allegro has no pinky).

Usage:
    python 94_site_slots.py            # validate against every hand we have on disk
    python 94_site_slots.py --json     # emit the mapping as JSON
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SLOTS = tuple(f"{s}_{f}" for s in SIDES for f in FINGERS)   # canonical 10-slot vocabulary
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}

# finger aliases. Order matters: check the more specific patterns first so that
# e.g. "middle" is not swallowed by "mf" or "little" not by "lf".
_FINGER_PATTERNS = [
    ("thumb", r"thumb|_th|thj|\bth\b"),
    ("index", r"index|fore|_ff|ffj|\bff\b"),
    ("middle", r"middle|_mf|mfj|\bmf\b"),
    ("ring", r"ring|_rf|rfj|\brf\b"),
    ("pinky", r"pinky|pinkie|little|_lf|lfj|\blf\b"),
]
# NOTE: no \b here -- underscore is a word char, so "right_thumb" has no boundary
_SIDE_PATTERN = re.compile(r"(?:^|_)(left|right)(?=_|$)", re.I)


def parse_site(name: str) -> tuple[str, str] | None:
    """'right_FFJ1' -> ('right','index');  'left_thumb_elastomer' -> ('left','thumb')."""
    s = str(name)
    ls = s.lower()
    m = _SIDE_PATTERN.search(ls)
    if not m:
        # fall back to a leading-token match (e.g. 'R_thumb', 'L_index')
        head = ls.split("_", 1)[0]
        side = {"l": "left", "r": "right", "lt": "left", "rt": "right"}.get(head)
        if side is None:
            return None
    else:
        side = m.group(1).lower()
    for finger, pat in _FINGER_PATTERNS:
        if re.search(pat, ls):
            return f"{side}_{finger}", finger
    return None


def slots_for(names) -> tuple[list[int], list[bool], list[str]]:
    """-> (slot indices present, presence mask over the 10 slots, unmapped names)."""
    idx, unmapped = [], []
    for n in names:
        p = parse_site(n)
        if p is None:
            unmapped.append(str(n))
        else:
            idx.append(SLOT_INDEX[p[0]])
    mask = [False] * len(SLOTS)
    for i in idx:
        mask[i] = True
    return idx, mask, unmapped


def hand_names_from_disk() -> dict[str, list[str]]:
    """Read site_names for every embodiment we have at least one episode of."""
    import h5py
    out: dict[str, list[str]] = {}
    pat = "/mnt/public/datasets/bench2dex/teleopdata/dataset/*/replay-generalization/episode_*.hdf5"
    for p in sorted(glob.glob(pat)):
        try:
            with h5py.File(p, "r") as f:
                rk = f["robot/tactile/meta"]["robot_key"][()].decode()
                if rk in out:
                    continue
                sn = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in f["robot/tactile/meta"]["site_names"][()]]
                out[rk] = sn
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    hands = hand_names_from_disk()
    if not hands:
        print("no episodes on disk")
        return 1

    mapping = {}
    bad = 0
    print(f"canonical slot vocabulary ({len(SLOTS)}): {list(SLOTS)}\n")
    for rk, names in sorted(hands.items()):
        idx, mask, unmapped = slots_for(names)
        n_present = sum(mask)
        dup = len(idx) != len(set(idx))
        ok = (not unmapped) and (not dup) and n_present == len(idx)
        bad += 0 if ok else 1
        mapping[rk] = {"names": names, "slots": [SLOTS[i] for i in idx],
                       "mask": mask, "unmapped": unmapped}
        tag = "OK " if ok else "!! "
        print(f"{tag}{rk[:44]:46s} n={len(names):2d} -> slots={n_present:2d}"
              f"{'  DUP!' if dup else ''}{'  UNMAPPED=' + str(unmapped) if unmapped else ''}")
        print(f"      {[SLOTS[i] for i in idx]}")
    print(f"\n{len(hands) - bad}/{len(hands)} hands map cleanly onto the shared 10-slot space")
    if args.json:
        print(json.dumps(mapping, indent=2, ensure_ascii=False))
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
