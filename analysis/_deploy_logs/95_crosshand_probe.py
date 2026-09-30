#!/usr/bin/env python3
"""Cross-hand tactile representation probe: positional embedding vs semantic slots.

Why this probe exists
---------------------
Bench2Dex crosses neither tasks nor embodiments ("tasks and embodiments are not
factorially crossed"), so a POLICY trained on one hand cannot be evaluated on another
hand's task. The transferable object is therefore the REPRESENTATION, and that is what
this script measures.

Setup
-----
Input : every present site's tactile map of one hand at one frame (downsampled).
Target: the per-SLOT contact state (canonical 10 slots: side x finger role), i.e. which
        fingers of that hand are in contact. Target is hand-independent by construction
        because it is expressed in slot space, not in the dataset's site order.
Split : leave-one-hand-out. Train on N-1 hands, test on the held-out hand.

Two encoders share everything except how a site is identified:
  * positional : `nn.Embedding(num_sites)` (what the repo's tactile forks do) --
                 assumes a fixed count and a fixed order, and cannot be constructed for
                 a hand with a different count at all.
  * semantic   : `nn.Embedding(len(SLOTS)=10)` indexed by the site's SLOT, plus a presence
                 mask for slots the hand lacks -- one model serves 8- and 10-site hands.

Metric: per-slot contact F1 on the held-out hand.

Usage: 95_crosshand_probe.py [--epochs 12] [--stride 20]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wangrenpeng/bench2dex/_deploy_logs")
from site_slots import SLOTS, SLOT_INDEX, slots_for  # noqa: E402

DATA = "/mnt/public/datasets/bench2dex/teleopdata/dataset"
CACHE = Path("/home/wangrenpeng/bench2dex/data/crosshand_cache")
TAC_GRID = 16


def hand_of_episode(path: str) -> tuple[str, list[str]]:
    import h5py
    with h5py.File(path, "r") as f:
        rk = f["robot/tactile/meta"]["robot_key"][()].decode()
        sn = [s.decode() if isinstance(s, bytes) else str(s)
              for s in f["robot/tactile/meta"]["site_names"][()]]
    return rk, sn


def build_cache(stride: int, max_ep_per_hand: int) -> None:
    """One .npz per (hand, episode): maps [nF, 10, G, G] uint8 scattered into SLOTS,
    plus labels [nF, 10] uint8 (contact per slot) and mask [10] bool."""
    import h5py
    CACHE.mkdir(parents=True, exist_ok=True)
    eps = sorted(glob.glob(f"{DATA}/*/replay-generalization/episode_*.hdf5"))
    per_hand: dict[str, int] = {}
    done = 0
    for p in eps:
        task = os.path.basename(os.path.dirname(os.path.dirname(p)))
        try:
            rk, sn = hand_of_episode(p)
        except Exception:
            continue
        if per_hand.get(rk, 0) >= max_ep_per_hand:
            continue
        out = CACHE / f"{rk}__{task}__{Path(p).stem}.npz"
        if out.exists():
            per_hand[rk] = per_hand.get(rk, 0) + 1
            done += 1
            continue
        idx, mask, unmapped = slots_for(sn)
        if unmapped or len(idx) != len(sn):
            print(f"   !! {rk} has unmapped sites {unmapped}, skip")
            break
        try:
            with h5py.File(p, "r") as f:
                T = int(f["robot"]["tactile"]["tacmap"][sn[0]].shape[0])
                fr = np.arange(0, T, stride)
                maps = np.zeros((len(fr), len(SLOTS), TAC_GRID, TAC_GRID), np.uint8)
                lab = np.zeros((len(fr), len(SLOTS)), np.uint8)
                for si, s in enumerate(sn):
                    slot = idx[si]
                    d = f["robot/tactile"]["tacmap"][s]        # (T,240,240)
                    c = f["robot/tactile"]["contact_mask"][s]  # (T,240,240) bool
                    for j, i in enumerate(fr):
                        im = np.asarray(d[i])
                        bh, bw = im.shape[0] // TAC_GRID, im.shape[1] // TAC_GRID
                        maps[j, slot] = im[: bh * TAC_GRID, : bw * TAC_GRID] \
                            .reshape(TAC_GRID, bh, TAC_GRID, bw).max(axis=(1, 3))
                        lab[j, slot] = 1 if np.any(np.asarray(c[i])) else 0
            np.savez_compressed(out, maps=maps, labels=lab,
                                mask=np.array(mask, bool), hand=rk)
            per_hand[rk] = per_hand.get(rk, 0) + 1
            done += 1
            if done % 10 == 0:
                print(f"   cached {done} (hands so far: {len(per_hand)})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"   !! {Path(p).name}: {type(exc).__name__}: {exc}")
    print(f"   cache: {done} episodes over {len(per_hand)} hands -> {CACHE}")
    print("   per hand:", {k: v for k, v in sorted(per_hand.items())})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--max-ep-per-hand", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()

    if not list(CACHE.glob("*.npz")):
        print("building cache ...")
        build_cache(args.stride, args.max_ep_per_hand)
    if args.cache_only:
        return 0

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    files = sorted(CACHE.glob("*.npz"))
    hands = sorted({f.name.split("__")[0] for f in files})
    # drop hands whose tactile signal is dead/negligible in this dataset
    # (measured: multi_xarm7_with_leap has max=0 in tasks 09 and 62, ~1.5% contact in 51)
    DEAD = {"multi_xarm7_with_leap"}
    dropped = [h for h in hands if h in DEAD]
    hands = [h for h in hands if h not in DEAD]
    files = [f for f in files if f.name.split("__")[0] not in DEAD]
    if dropped:
        print(f"dropped (dead tactile): {dropped}")
    print(f"\n{len(files)} cached episodes over {len(hands)} hands:")
    for h in hands:
        ns = sum(1 for f in files if f.name.startswith(h + "__"))
        print(f"   {h[:46]:48s} {ns} episodes")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Enc(nn.Module):
        """Site maps -> per-slot contact logits. `mode` decides how a site is identified."""

        def __init__(self, mode: str, num_sites_hint: int = 10, d: int = 64):
            super().__init__()
            self.mode = mode
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1),
            )
            self.proj = nn.Linear(32, d)
            # positional: index by position in the dataset's site list
            # semantic  : index by canonical SLOT
            n_emb = len(SLOTS) if mode == "semantic" else max(num_sites_hint, 1)
            self.site_emb = nn.Embedding(n_emb, d)
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, maps, mask, site_idx, target):
            """maps [B,S,G,G] ; mask [B,S] ; site_idx [B,S] ; target [B] slot index.

            The target slot's OWN tactile map is removed from the input, so predicting
            whether that finger is in contact must use (a) the other fingers' maps and
            (b) knowledge of WHICH finger is being asked about. That is exactly what the
            site identifier has to encode, and it is what differs between a positional
            and a semantic scheme.
            """
            B, S, G, _ = maps.shape
            keep = mask.clone()
            keep.scatter_(1, target[:, None], False)          # drop the target site
            x = maps.float().unsqueeze(2) / 255.0
            x = self.cnn(x.reshape(B * S, 1, G, G)).reshape(B, S, -1)
            h = self.proj(x) + self.site_emb(site_idx)
            ctx = (h * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)
            # query the TARGET slot: its identity matters, its own map is gone
            tgt_h = self.site_emb(site_idx.gather(1, target[:, None]).squeeze(1))
            return self.head(tgt_h + ctx).squeeze(-1)          # [B] logit for the target

    def load_hand(h: str):
        """All tensors are padded to the FULL 10-slot space; `mask` marks which slots the
        hand actually has. That keeps shapes uniform across 8-site and 10-site hands --
        the whole point of the canonicalization."""
        fs = [f for f in files if f.name.startswith(h + "__")]
        M, L, K = [], [], []
        for f in fs:
            z = np.load(f, allow_pickle=True)
            m, lb, k = z["maps"], z["labels"], z["mask"]
            M.append(m); L.append(lb)
            K.append(np.repeat(k[None], len(m), 0))
        return np.concatenate(M), np.concatenate(L), np.concatenate(K)

    # slot -> original position in each hand's site list (for the positional baseline)
    slot_pos: dict[str, np.ndarray] = {}
    for h in hands:
        f0 = next(f for f in files if f.name.startswith(h + "__"))
        task = f0.name.split("__")[1]
        ep = f"{DATA}/{task}/replay-generalization/" + f0.name.split("__")[2].replace(".npz", ".hdf5")
        _, sn = hand_of_episode(ep)
        idx, mask, _ = slots_for(sn)
        pm = np.full(len(SLOTS), -1, np.int64)
        for position, slot in enumerate(idx):
            pm[slot] = position
        slot_pos[h] = pm

    def pos_ids(h: str, n: int) -> np.ndarray:
        """[n,10] site identifiers. positional: original list index; semantic: canonical slot."""
        pm = slot_pos[h]
        ids = np.zeros((n, len(SLOTS)), np.int64)
        for j in range(len(SLOTS)):
            ids[:, j] = j if mode == "semantic" else (pm[j] if pm[j] >= 0 else 0)
        return ids

    n_pos = int(max(slot_pos[h].max() for h in hands) + 1)

    results = {}
    for mode in ("positional", "semantic"):
        print(f"\n=== encoder: {mode}  (site-id table size = "
              f"{len(SLOTS) if mode == 'semantic' else n_pos})")
        per_hand_f1 = {}
        for held in hands:
            tr_hands = [h for h in hands if h != held]
            Ms, Ls, Ks, Is = [], [], [], []
            for h in tr_hands:
                m, lb, k = load_hand(h)
                Ms.append(m); Ls.append(lb); Ks.append(k); Is.append(pos_ids(h, len(m)))
            Mt = np.concatenate(Ms); Lt = np.concatenate(Ls)
            Kt = np.concatenate(Ks); It = np.concatenate(Is)
            Mv, Lv, Kv = load_hand(held)
            Iv = pos_ids(held, len(Mv))

            model = Enc(mode, num_sites_hint=n_pos).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

            def pick_target(K):
                """random present slot per sample"""
                tg = np.zeros(len(K), np.int64)
                for i in range(len(K)):
                    pres = np.where(K[i])[0]
                    tg[i] = pres[np.random.randint(len(pres))]
                return tg

            def batches(M, L, K, I, T, bs=64, shuffle=True):
                n = len(M)
                order = np.random.permutation(n) if shuffle else np.arange(n)
                for i in range(0, n, bs):
                    j = order[i:i + bs]
                    yield (torch.as_tensor(M[j], device=dev).float(),
                           torch.as_tensor(L[j], device=dev).float(),
                           torch.as_tensor(K[j], device=dev),
                           torch.as_tensor(I[j], device=dev).long(),
                           torch.as_tensor(T[j], device=dev).long())

            Tt = pick_target(Kt)
            for ep_i in range(args.epochs):
                model.train()
                for m, lb, k, ii, tg in batches(Mt, Lt, Kt, It, Tt):
                    logit = model(m, k, ii, tg)
                    y = lb.gather(1, tg[:, None]).squeeze(1)
                    loss = F.binary_cross_entropy_with_logits(logit, y)
                    opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            rows = []
            with torch.no_grad():
                for _ in range(6):                     # several target draws -> AUC estimate
                    Tv = pick_target(Kv)
                    for m, lb, k, ii, tg in batches(Mv, Lv, Kv, Iv, Tv, bs=256, shuffle=False):
                        pr = torch.sigmoid(model(m, k, ii, tg))
                        y = lb.gather(1, tg[:, None]).squeeze(1)
                        rows.append(torch.stack([pr, y], 1).cpu())
            R = torch.cat(rows).numpy()
            pos, neg = R[R[:, 1] > 0.5, 0], R[R[:, 1] <= 0.5, 0]
            if len(pos) == 0 or len(neg) == 0:
                auc = float("nan")
            else:
                auc = float((pos[:, None] > neg[None, :]).mean()
                            + 0.5 * (pos[:, None] == neg[None, :]).mean())
            base = float((R[:, 1] > 0.5).mean())
            per_hand_f1[held] = auc
            print(f"   held-out {held[:44]:44s} AUC={auc:.3f}  (target pos rate {base:.3f})", flush=True)
        results[mode] = per_hand_f1
        vals = [v for v in per_hand_f1.values() if v == v]
        print(f"   ** {mode}: mean AUC over held-out hands = {np.mean(vals):.3f}")

    print("\n================ SUMMARY ================")
    print(f"{'held-out hand':48s} {'positional':>11s} {'semantic':>10s}")
    for h in hands:
        print(f"{h[:46]:48s} {results['positional'][h]:11.3f} {results['semantic'][h]:10.3f}")
    print(f"{'MEAN':48s} {np.mean(list(results['positional'].values())):11.3f} "
          f"{np.mean(list(results['semantic'].values())):10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
