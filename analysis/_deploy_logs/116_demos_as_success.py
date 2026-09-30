#!/usr/bin/env python3
"""Can 300 demonstration episodes (all SUCCESS) rescue the learned failure predictor?

Why
---
The failure predictor currently trains on only 60 policy rollouts (20 per task, 15 success
/ 45 failure). The benchmark ships 100 teleoperated demonstrations per task -- 300 here --
which are ALL successful (final lscr = 1.0 for every one). Those are free, already on disk,
and already featurised. They are exactly the "what does success look like" data a safety
monitor wants, so this script adds them as the success class.

Protocol
--------
Evaluation is on POLICY ROLLOUTS only (the deployment target), 5-fold episode-level CV.
For fold k:
    train = ALL demonstrations  +  rollouts not in fold k
    test  = rollouts in fold k
A `rollouts_only` control uses the identical folds so the demo contribution is isolated.
The `feature` row is the hand-crafted contact statistic (needs no training) -- the baseline
the learned model has so far failed to beat.

Domain note: demonstrations are teleoperation, rollouts are policy attempts. Mixing them is
deliberate -- the monitor should learn that an expert-like contact pattern means safety --
but it is a domain shift and is reported as such.

Usage: 116_demos_as_success.py --epochs 25 --folds 5 [--stride 5] [--model-seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wangrenpeng/bench2dex/_deploy_logs")
from importlib.machinery import SourceFileLoader

EW = SourceFileLoader("ew", "/home/wangrenpeng/bench2dex/_deploy_logs/107_failure_early_warning.py").load_module()
TASKS = EW.TASKS
FEAT = Path("/home/wangrenpeng/bench2dex/data/probe_features")


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def ep_auc(pred, ep, y):
    sc, lb = [], []
    for e in np.unique(ep):
        m = ep == e
        sc.append(pred[m].mean()); lb.append(y[m][0])
    return auc(sc, lb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5, help="demo features are precomputed at stride 5")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--demo-weights", default="1.0",
                    help="comma list of per-demo sampling weights. With 300 demos (all "
                         "success) vs ~48 rollouts, the source itself predicts the label, so "
                         "the model can lower its loss by detecting the DOMAIN instead of the "
                         "failure. Lower weights make the domain cue uninformative while "
                         "keeping the contact-level signal intact (unlike feature "
                         "normalisation, which would remove the very signal being used).")
    ap.add_argument("--demo-weight", type=float, default=1.0)
    ap.add_argument("--max-ep", type=int, default=200)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = args.window

    # ---------------- demonstrations: every episode is a success ----------------
    demos = []
    for t in ("26", "32", "73"):
        fs = sorted(glob.glob(str(FEAT / TASKS[t] / "*.npz")))[:args.max_ep]
        for p in fs:
            z = np.load(p)
            demos.append((t, z["tac"], z["prop"]))
    print(f"   demos: {len(demos)} episodes (all success)")
    if not demos:
        print("   no demo features found"); return 1

    # ---------------- rollouts: the evaluation target ----------------
    roll = {}
    for t in ("26", "32", "73"):
        d = EW.load_task(t, TASKS[t], args.stride, args.max_ep)
        if d is None:
            continue
        roll[t] = d
        print(f"   rollouts task {t}: {len(np.unique(d['ep']))} episodes, "
              f"failure rate {d['y'].mean():.2f}")
    if not roll:
        print("   no rollouts"); return 1
    tids = list(roll)

    # ---------------- tensors: [demos | rollouts] ----------------
    tac_l, prop_l, y_l, ep_l, tk_l, src_l = [], [], [], [], [], []
    cur = 0
    for t, tac, prop in demos:
        n = len(tac)
        tac_l.append(tac); prop_l.append(prop)
        y_l.append(np.zeros(n, np.int64)); ep_l.append(np.full(n, cur))
        tk_l.append(np.full(n, tids.index(t))); src_l.append(np.full(n, 0))
        cur += 1
    n_demo_ep = cur
    for t in tids:
        d = roll[t]
        n = len(d["y"])
        tac_l.append(d["tac"]); prop_l.append(d["prop"])
        y_l.append(d["y"].astype(np.int64))
        # one episode id PER EPISODE -- d["ep"] holds per-episode ids starting at 0
        ep_l.append(d["ep"].astype(np.int64) + cur)
        tk_l.append(np.full(n, tids.index(t))); src_l.append(np.full(n, 1))
        cur += int(d["ep"].max()) + 1

    tac_t = torch.as_tensor(np.concatenate(tac_l))
    prop_t = torch.as_tensor(np.concatenate(prop_l))
    y_t = torch.as_tensor(np.concatenate(y_l))
    ep_t = torch.as_tensor(np.concatenate(ep_l))
    tk_t = torch.as_tensor(np.concatenate(tk_l))
    src_t = torch.as_tensor(np.concatenate(src_l))
    ep_np = ep_t.numpy()
    demo_eps = np.unique(ep_np[src_t.numpy() == 0])
    roll_eps = np.unique(ep_np[src_t.numpy() == 1])
    print(f"   total episodes: {len(np.unique(ep_np))} "
          f"({len(demo_eps)} demos + {len(roll_eps)} rollouts), frames {len(ep_np)}")

    # folds over ROLLOUT episodes only, one task at a time, deterministic
    folds = []
    for t in tids:
        e = ep_np[(src_t.numpy() == 1) & (tk_t.numpy() == tids.index(t))]
        e = np.unique(e)
        rng = np.random.RandomState(11)
        rng.shuffle(e)
        folds.append(np.array_split(e, args.folds))

    class Net(nn.Module):
        def __init__(self, d=64, n_tasks=0, n_slots=10, agg="mean"):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1))          # spatial: max keeps sparse contact
            self.proj = nn.Linear(32, d)
            self.slot = nn.Embedding(n_slots, d)
            self.task_emb = nn.Embedding(n_tasks, d) if n_tasks > 0 else None
            self.pproj = nn.Linear(58, 32)
            self.gru = nn.GRU(d + 32, d, batch_first=True)
            self.head = nn.Linear(d, 1)
            self.agg = agg                          # over SITES: count matters, so mean/sum

        def forward(self, x, pp, task=None):
            x = x / 255.0
            B, W_, S, G, _ = x.shape
            f = self.cnn(x.reshape(B * W_ * S, 1, G, G)).reshape(B * W_, S, -1)
            f = self.proj(f) + self.slot.weight[None]
            f = f.max(1).values if self.agg == "max" else \
                (f.mean(1) if self.agg == "mean" else f.sum(1))
            f = f.reshape(B, W_, -1)
            if self.task_emb is not None and task is not None:
                f = f + self.task_emb(task)[:, None, :]
            g = torch.relu(self.pproj(pp))
            h, _ = self.gru(torch.cat([f, g], -1))
            return self.head(h[:, -1]).squeeze(-1)

    def ep_windows(e):
        idx = np.where(ep_np == e)[0]
        return idx[idx >= W - 1][W - 1::max(1, args.stride)]

    def train_model(train_eps, n_tasks, agg, seed, demo_w=1.0):
        torch.manual_seed(seed)
        net = Net(n_tasks=n_tasks, agg=agg).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        tr = {int(e): ep_windows(e) for e in train_eps}
        tr = {e: g for e, g in tr.items() if len(g)}
        keys = list(tr)
        lab = {e: float(y_t[tr[e][0]]) for e in keys}
        tsk = {e: int(tk_t[tr[e][0]]) for e in keys}
        is_demo = {e: int(src_t[tr[e][0]]) == 0 for e in keys}
        w = np.array([demo_w if is_demo[e] else 1.0 for e in keys], float)
        probs = w / max(w.sum(), 1e-9)
        for _ in range(args.epochs):
            net.train()
            for _ in range(args.steps_per_epoch):
                sel = np.random.choice(keys, min(args.batch, len(keys)), replace=False, p=probs)
                logs, labs = [], []
                for e in sel:
                    g = tr[e]
                    if len(g) > 32:
                        g = g[np.random.choice(len(g), 32, replace=False)]
                    ii = g[:, None] + np.arange(-W + 1, 1)[None, :]
                    tv = torch.full((len(g),), tsk[e], device=dev, dtype=torch.long)
                    out = net(tac_t[ii].to(dev).float(), prop_t[ii].to(dev), tv if n_tasks else None)
                    logs.append(out.mean()); labs.append(lab[e])
                loss = F.binary_cross_entropy_with_logits(
                    torch.stack(logs), torch.tensor(labs, device=dev, dtype=torch.float32))
                opt.zero_grad(); loss.backward(); opt.step()
        return net

    def predict(net, eps, n_tasks):
        net.eval()
        idx = np.concatenate([ep_windows(e) for e in eps]) if len(eps) else np.zeros(0, int)
        if len(idx) == 0:
            return np.zeros(0), np.zeros(0, int), np.zeros(0)
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                ii = b[:, None] + np.arange(-W + 1, 1)[None, :]
                tv = tk_t[b].to(dev)
                outs.append(torch.sigmoid(
                    net(tac_t[ii].to(dev).float(), prop_t[ii].to(dev), tv if n_tasks else None)
                ).cpu().numpy())
        return np.concatenate(outs), ep_np[idx], y_t[idx].numpy()

    def contact_stat(eps):
        idx = np.concatenate([ep_windows(e) for e in eps]) if len(eps) else np.zeros(0, int)
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        c = (tac_t[ii].float() > 0).float()
        ns = (c.mean(dim=(3, 4)) > 0).float().mean(dim=1)
        return (-ns.numpy()), ep_np[idx], y_t[idx].numpy()

    ws = [float(x) for x in args.demo_weights.split(",")]
    variants = ["rollouts_only"] + [f"demos_w{w:g}" for w in ws] + ["feature"]
    acc = {v: {t: [] for t in tids} for v in variants}
    for k in range(args.folds):
        val_eps = np.concatenate([folds[i][k] for i in range(len(tids))])
        tr_roll = np.setdiff1d(roll_eps, val_eps)
        net = train_model(tr_roll, 0, "mean", args.model_seed)
        for i, t in enumerate(tids):
            p, e_, y_ = predict(net, folds[i][k], 0)
            acc["rollouts_only"][t].append((p, e_, y_))
        for w in ws:
            net = train_model(np.concatenate([demo_eps, tr_roll]), 0, "mean",
                              args.model_seed, demo_w=w)
            for i, t in enumerate(tids):
                p, e_, y_ = predict(net, folds[i][k], 0)
                acc[f"demos_w{w:g}"][t].append((p, e_, y_))
        for i, t in enumerate(tids):
            p, e_, y_ = contact_stat(folds[i][k])
            acc["feature"][t].append((p, e_, y_))
        print(f"   fold {k + 1}/{args.folds} done", flush=True)

    print(f"\n   {'variant':16s}" + "".join(f"{('task ' + t):>10s}" for t in tids) + f"{'mean':>10s}")
    table = {}
    for v in variants:
        row = {}
        for t in tids:
            ps = np.concatenate([a for a, _, _ in acc[v][t]])
            es = np.concatenate([a for _, a, _ in acc[v][t]])
            ys = np.concatenate([a for _, _, a in acc[v][t]])
            row[t] = ep_auc(ps, es, ys)
        table[v] = row
        print(f"   {v:16s}" + "".join(f"{row[t]:10.3f}" for t in tids)
              + f"{np.nanmean(list(row.values())):10.3f}")

    stat_mean = np.nanmean(list(table["feature"].values()))
    for v in variants:
        m = np.nanmean(list(table[v].values()))
        print(f"   {v:16s} mean={m:.3f}   vs statistic {stat_mean:.3f}   ({m - stat_mean:+.3f})")
    out = Path("/home/wangrenpeng/bench2dex/data/demos_as_success.json")
    out.write_text(json.dumps(table, indent=2, default=float))
    print(f"   saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
