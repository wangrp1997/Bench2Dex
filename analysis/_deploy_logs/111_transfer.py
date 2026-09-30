#!/usr/bin/env python3
"""ZERO-SHOT CROSS-TASK TRANSFER for tactile failure prediction.

Why
---
On a single task with ~20 rollouts, a trivial contact statistic already reaches
epAUC 0.96-0.99, so a learned encoder that merely matches it buys nothing. The place a
learned model can actually win is TRANSFER: train the safety predictor on task A and
apply it unchanged to task B, where a task-specific statistic/threshold has no reason to
hold. This script measures exactly that.

Protocol
--------
For every ordered task pair (A -> B): train our tactile encoder with EPISODE-level
supervision on all of A's episodes, then score B's held-out episodes. Compare against the
hand-crafted contact statistic (needs no training, so it is a strong free baseline).
AUC is threshold-free, so this comparison is fair to both.

Usage: 111_transfer.py --tasks 26,32,73
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wangrenpeng/bench2dex/_deploy_logs")
from importlib.machinery import SourceFileLoader

EW = SourceFileLoader("ew", "/home/wangrenpeng/bench2dex/_deploy_logs/107_failure_early_warning.py").load_module()
TASKS = EW.TASKS


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def ep_auc_from_windows(pred, ep, lab, frac, bucket=None):
    """one score per episode (optionally restricted to a time bucket) -> AUC over episodes."""
    m = np.ones(len(pred), bool) if bucket is None else bucket
    sc, lb = [], []
    for e in np.unique(ep[m]):
        mm = m & (ep == e)
        if mm.sum() == 0:
            continue
        sc.append(np.asarray(pred)[mm].mean())
        lb.append(lab[mm][0])
    return auc(sc, lb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="26,32,73")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-ep", type=int, default=60)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    D = {}
    for tid in args.tasks.split(","):
        d = EW.load_task(tid, TASKS[tid], args.stride, args.max_ep)
        if d is None:
            continue
        D[tid] = d
        print(f"   task {tid}: {len(np.unique(d['ep']))} episodes, {len(d['y'])} frames, "
              f"failure rate {d['y'].mean():.2f}", flush=True)
    if len(D) < 2:
        print("   need >= 2 tasks with tactile rollouts"); return 1

    W = args.window
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self, d=64, n_slots=10):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1))
            self.proj = nn.Linear(32, d)
            self.slot = nn.Embedding(n_slots, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            x = x / 255.0
            B, W_, S, G, _ = x.shape
            f = self.cnn(x.reshape(B * W_ * S, 1, G, G)).reshape(B * W_, S, -1)
            f = (self.proj(f) + self.slot.weight[None]).max(1).values
            h, _ = self.gru(f.reshape(B, W_, -1))
            return self.head(h[:, -1]).squeeze(-1)

    def prep(d, tid):
        """-> tac tensor, labels, episode ids, absolute frame index, window starts."""
        tac = torch.as_tensor(d["tac"])
        y = d["y"]; ep = d["ep"]; fr = d["frac"]
        starts = []
        for e in np.unique(ep):
            idx = np.where(ep == e)[0]
            if len(idx) >= W:
                starts.extend(idx[W - 1::max(1, args.stride)].tolist())
        return tac, y, ep, fr, np.array(starts, dtype=np.int64)

    P = {t: prep(D[t], t) for t in D}

    def windows(tid, idx):
        tac, y, ep, fr, _ = P[tid]
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        return tac[ii].to(dev).float(), y[idx], ep[idx], fr[idx]

    def contact_stat(tid, idx):
        tac, _, _, _ = None, None, None, None
        tac_t = P[tid][0]
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        c = (tac_t[ii].float() > 0).float()
        ns = (c.mean(dim=(3, 4)) > 0).float().mean(dim=1)   # mean # contacting sites
        return (-ns.numpy())                                # higher = more likely failure

    def train(tid):
        tac, y, ep, fr, starts = P[tid]
        tr_eps = np.unique(ep)
        tr_idx = {e: np.where(ep == e)[0] for e in tr_eps}
        tr_idx = {e: (g[g >= W - 1][W - 1::max(1, args.stride)]) for e, g in tr_idx.items()}
        tr_idx = {e: g for e, g in tr_idx.items() if len(g) > 0}
        keys = list(tr_idx.keys())
        lab = {e: float(y[tr_idx[e][0]]) for e in keys}
        torch.manual_seed(0)
        net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            net.train()
            for _ in range(args.steps_per_epoch):
                sel = np.random.choice(keys, min(args.batch, len(keys)), replace=False)
                logs, labs = [], []
                for e in sel:
                    g = tr_idx[e]
                    if len(g) > 32:
                        g = g[np.random.choice(len(g), 32, replace=False)]
                    t, _, _, _ = windows(tid, g)
                    logs.append(net(t).mean()); labs.append(lab[e])
                loss = F.binary_cross_entropy_with_logits(
                    torch.stack(logs), torch.tensor(labs, device=dev, dtype=torch.float32))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        return net

    def score(tid, net=None):
        tac, y, ep, fr, starts = P[tid]
        if net is None:
            return contact_stat(tid, starts), ep[starts], y[starts], fr[starts]
        outs = []
        with torch.no_grad():
            for i in range(0, len(starts), 256):
                t, _, _, _ = windows(tid, starts[i:i + 256])
                outs.append(torch.sigmoid(net(t)).cpu().numpy())
        return np.concatenate(outs), ep[starts], y[starts], fr[starts]

    b0 = lambda f: f < 100
    b1 = lambda f: (f >= 100) & (f < 200)

    rows = []
    print(f"\n   {'train->test':16s} {'learned':>9s} {'learned<200f':>13s} "
          f"{'contact-stat':>13s} {'stat<200f':>11s}")
    for a in D:
        net = train(a)
        for b in D:
            if a == b:
                continue
            ps, ep_, y_, fr_ = score(b, net)
            p_sc, _, _, _ = score(b, None)
            r = dict(train=a, test=b,
                     learned=ep_auc_from_windows(ps, ep_, y_, fr_),
                     learned_early=ep_auc_from_windows(ps, ep_, y_, fr_, b0(fr_) | b1(fr_)),
                     stat=ep_auc_from_windows(p_sc, ep_, y_, fr_),
                     stat_early=ep_auc_from_windows(p_sc, ep_, y_, fr_, b0(fr_) | b1(fr_)))
            rows.append(r)
            print(f"   {a + ' -> ' + b:16s} {r['learned']:9.3f} {r['learned_early']:13.3f} "
                  f"{r['stat']:13.3f} {r['stat_early']:11.3f}", flush=True)

    if rows:
        print(f"\n   mean over {len(rows)} transfers: "
              f"learned={np.mean([r['learned'] for r in rows]):.3f} "
              f"(early {np.mean([r['learned_early'] for r in rows]):.3f})  "
              f"contact-stat={np.mean([r['stat'] for r in rows]):.3f} "
              f"(early {np.mean([r['stat_early'] for r in rows]):.3f})")
    out = Path("/home/wangrenpeng/bench2dex/data/transfer.json")
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"   saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
