#!/usr/bin/env python3
"""Does touch know something about TASK PROGRESS that vision does not?  (path A)

Target
------
Per-frame task progress = the benchmark's own `lscr` (latched stage completion rate),
which takes 5 levels {0, .25, .5, .75, 1}. It is recomputed by us from the recorded
object states through the repo's executable stage predicates, and validated: the final
value is 1.0 for all 300 demonstrations, as it must be for a successful demo set
(the HDF5's stored `metrics/episode/*` are NOT reliable: task 73 stores 0.0 for all 100
episodes although the puzzle pieces demonstrably sit on target).

Why this target
---------------
`chain_depth` is NOT usable as a progress signal here: task 73's stages have no
dependencies, so it jumps 0->1 in the first 120 frames and is then constant while the
task is only 15% done. `lscr` instead changes 4x per episode with each level lasting
69-276 frames, which is what gives this probe statistical power.

The question is a modality comparison, not a stage classifier: train the SAME encoder
shape on tactile-only / RGB-only / proprio-only / fused windows and compare how much
task-state information each carries, SEPARATELY for frames near a progress transition
and frames far from one. Hypothesis under test:
    touch resolves TRANSITIONS, vision resolves APPROACH.

Encoder (ours)
--------------
Tactile is encoded as a SET over semantic finger slots (side x finger role) with a
presence mask -- not by positional embedding -- then pooled, then a GRU over the
temporal window. Sparse contact is kept by MAX pooling, not average pooling.

Usage: 98_modality_probe.py [--window 8] [--epochs 12] [--stride 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

FEAT = Path("/home/wangrenpeng/bench2dex/data/probe_features")
LAB = Path("/home/wangrenpeng/bench2dex/data/stage_labels")
TASKS = ["26_canned_food_tray_line_arrangement", "32_baking_tray_prep_with_tools",
         "73_jigsaw_puzzle_assembly"]
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
STRIDE = 5


def build() -> dict:
    T, R, P, Y, EP, TK, TR = [], [], [], [], [], [], []
    ep_id = 0
    for ti, task in enumerate(TASKS):
        for lp in sorted(LAB.glob(f"{task}/*.npz")):
            fp = FEAT / task / lp.name
            if not fp.exists():
                continue
            z = np.load(fp)
            lab = np.load(lp)
            lscr = lab["lscr"][::STRIDE].astype(np.float32)
            n = len(z["tac"])
            if len(lscr) != n:
                continue
            y = np.argmin(np.abs(lscr[:, None] - np.array(LEVELS)[None, :]), axis=1)
            # frames within +-1 sampled frame of an lscr change
            ch = np.zeros(n, bool)
            d = np.abs(np.diff(lscr, prepend=lscr[0]))
            tr = d > 1e-6
            for s in np.where(tr)[0]:
                ch[max(0, s - 1):min(n, s + 2)] = True
            T.append(z["tac"]); R.append(z["rgb"]); P.append(z["prop"])
            Y.append(y); EP.append(np.full(n, ep_id)); TK.append(np.full(n, ti)); TR.append(ch)
            ep_id += 1
    out = dict(tac=np.concatenate(T), rgb=np.concatenate(R), prop=np.concatenate(P),
               y=np.concatenate(Y), ep=np.concatenate(EP), task=np.concatenate(TK),
               trans=np.concatenate(TR))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--modalities", default="tac,rgb,prop,tac+rgb,all")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    D = build()
    n = len(D["y"])
    print(f"   samples: {n}   episodes: {int(D['ep'].max())+1}   window={args.window}")
    for i, t in enumerate(TASKS):
        m = D["task"] == i
        cnt = np.bincount(D["y"][m], minlength=5)
        print(f"   {t[:36]:38s} n={int(m.sum()):6d}  level counts={cnt.tolist()}")
    print(f"   frames near a transition: {int(D['trans'].sum())} ({100*D['trans'].mean():.1f}%)")

    # ---- episode-level split (no frame leakage between train and val) ----
    eps = np.unique(D["ep"])
    rng = np.random.RandomState(0)
    rng.shuffle(eps)
    nval = max(1, int(0.2 * len(eps)))
    val_eps = set(eps[:nval].tolist())
    is_val = np.isin(D["ep"], list(val_eps))
    print(f"   split: {len(eps)-nval} train / {nval} val episodes")

    # ---- window index (starts whose whole window stays inside one episode) ----
    W = args.window
    starts = []
    for e in eps:
        idx = np.where(D["ep"] == e)[0]
        if len(idx) < W:
            continue
        starts.extend(idx[W - 1::args.stride].tolist())
    starts = np.array(starts)
    starts = starts[(starts >= W - 1)]
    tr_idx = starts[~is_val[starts]]
    va_idx = starts[is_val[starts]]
    print(f"   windows: {len(tr_idx)} train / {len(va_idx)} val")

    tac_t = torch.as_tensor(D["tac"]); rgb_t = torch.as_tensor(D["rgb"])
    prop_t = torch.as_tensor(D["prop"]); y_t = torch.as_tensor(D["y"])
    trans_t = torch.as_tensor(D["trans"]); task_t = torch.as_tensor(D["task"])

    def gather(idx, dev):
        off = np.arange(-W + 1, 1)[None, :]                 # (1,W)
        ii = idx[:, None] + off                             # (B,W)
        tt = tac_t[ii].to(dev).float()                      # (B,W,10,24,24)
        rr = rgb_t[ii].to(dev).float()                      # (B,W,64,64)
        pp = prop_t[ii].to(dev)
        return tt, rr, pp, y_t[idx].to(dev), trans_t[idx].to(dev), task_t[idx].to(dev)

    class TacEnc(nn.Module):
        """OUR encoder: a SET over semantic finger slots, sparse-preserving, then time."""

        def __init__(self, d=96, n_slots=10):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1),          # MAX: contact is sparse, avg would erase it
            )
            self.proj = nn.Linear(32, d)
            self.slot = nn.Embedding(n_slots, d)  # semantic slot, NOT list position
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 5))

        def forward(self, x):                     # (B,W,10,24,24)
            B, W, S, G, _ = x.shape
            f = self.cnn(x.reshape(B * W * S, 1, G, G)).reshape(B * W, S, -1)
            f = self.proj(f) + self.slot.weight[None, :, :]
            f = f.max(1).values                   # set aggregation over slots
            h, _ = self.gru(f.reshape(B, W, -1))
            return self.head(h[:, -1])

    class RgbEnc(nn.Module):
        def __init__(self, d=96):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(64, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 5))

        def forward(self, x):                     # (B,W,64,64)
            B, W, G, _ = x.shape
            f = self.proj(self.cnn(x.reshape(B * W, 1, G, G)).reshape(B * W, -1))
            h, _ = self.gru(f.reshape(B, W, -1))
            return self.head(h[:, -1])

    class PropEnc(nn.Module):
        def __init__(self, d=96):
            super().__init__()
            self.proj = nn.Linear(58, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 5))

        def forward(self, x):                     # (B,W,58)
            f = torch.relu(self.proj(x))
            h, _ = self.gru(f)
            return self.head(h[:, -1])

    class Fusion(nn.Module):
        def __init__(self, parts, d=96):
            super().__init__()
            self.parts = nn.ModuleList(parts)
            self.head = nn.Sequential(nn.Linear(d * len(parts), d), nn.ReLU(), nn.Linear(d, 5))

        def forward(self, tt, rr, pp):
            outs = []
            for m in self.parts:
                if isinstance(m, TacEnc):
                    outs.append(m(tt))
                elif isinstance(m, RgbEnc):
                    outs.append(m(rr))
                else:
                    outs.append(m(pp))
            # average the logits of the branches (fusion by logit averaging)
            return sum(outs) / len(outs)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name in args.modalities.split(","):
        torch.manual_seed(0)
        parts = []
        if "tac" in name:
            parts.append(TacEnc().to(dev))
        if "rgb" in name:
            parts.append(RgbEnc().to(dev))
        if "prop" in name:
            parts.append(PropEnc().to(dev))
        model = Fusion(parts).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for ep in range(args.epochs):
            model.train()
            perm = np.random.permutation(tr_idx)
            for i in range(0, len(perm), args.batch):
                b = perm[i:i + args.batch]
                tt, rr, pp, yy, _, _ = gather(b, dev)
                loss = F.cross_entropy(model(tt, rr, pp), yy)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        model.eval()
        preds, ys, ts, trs = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(va_idx), args.batch):
                b = va_idx[i:i + args.batch]
                tt, rr, pp, yy, tr, tk = gather(b, dev)
                p = model(tt, rr, pp).argmax(1)
                preds.append(p.cpu()); ys.append(yy.cpu()); trs.append(tr.cpu()); ts.append(tk.cpu())
        P_ = torch.cat(preds).numpy(); Y_ = torch.cat(ys).numpy()
        TR_ = torch.cat(trs).numpy(); TK_ = torch.cat(ts).numpy()
        acc = float((P_ == Y_).mean())
        near = TR_; far = ~TR_
        results[name] = dict(
            acc=acc,
            acc_near=float((P_[near] == Y_[near]).mean()) if near.any() else float("nan"),
            acc_far=float((P_[far] == Y_[far]).mean()) if far.any() else float("nan"),
            n_near=int(near.sum()), n_far=int(far.sum()),
            per_task={TASKS[t][:2]: float((P_[TK_ == t] == Y_[TK_ == t]).mean())
                      for t in range(len(TASKS)) if (TK_ == t).any()},
        )
        print(f"   {name:10s} acc={acc:.3f}   near-transition={results[name]['acc_near']:.3f} "
              f"(n={results[name]['n_near']})   far={results[name]['acc_far']:.3f} "
              f"(n={results[name]['n_far']})")

    maj = float(np.bincount(D["y"][va_idx], minlength=5).max() / len(va_idx))
    print(f"\n   majority-class baseline = {maj:.3f}")
    print(f"\n   {'modality':12s} {'acc':>7s} {'near-trans':>11s} {'far':>7s}   per-task(26/32/73)")
    for k, v in results.items():
        pt = v["per_task"]
        print(f"   {k:12s} {v['acc']:7.3f} {v['acc_near']:11.3f} {v['acc_far']:7.3f}   "
              + "/".join(f"{pt.get(t, float('nan')):.2f}" for t in ("26", "32", "73")))
    out = Path("/home/wangrenpeng/bench2dex/data/modality_probe_results.json")
    out.write_text(json.dumps(dict(results=results, majority=maj), indent=2))
    print(f"\n   saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
