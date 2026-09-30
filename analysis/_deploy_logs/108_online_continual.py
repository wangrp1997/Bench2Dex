#!/usr/bin/env python3
"""ONLINE CONTINUAL LEARNING for tactile failure prediction.

Setting
-------
Rollouts arrive as a stream, task by task (26 -> 32 -> 73). After each episode the model
takes a few online gradient steps; we then measure AUC on HELD-OUT episodes of every task
seen so far. This is the deployment reality: a safety predictor must keep adapting to a
new task without forgetting the previous ones.

Compared strategies -- identical except for the replay buffer
------------------------------------------------------------
  naive      : update only on the current episode (classic catastrophic forgetting)
  reservoir  : uniform replay over past windows
  safety     : OUR buffer. Safety events are rare and they are exactly what must not be
               forgotten, so weight stored windows by (failure + fingertip contact) and
               keep a uniform floor so the decision boundary does not collapse.

Metrics
-------
  per-task AUC trajectory   (adaptation speed and retention)
  final mean AUC
  forgetting = AUC(task) right after finishing it  -  AUC(task) at the end of the stream

Usage: 108_online_continual.py [--steps-per-ep 20] [--buffer 600]
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


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y > 0], p[y <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--steps-per-ep", type=int, default=20)
    ap.add_argument("--buffer", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-ep", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--strategies", default="naive,reservoir,safety")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="deterministic seed for which failure episodes enter the stream")
    ap.add_argument("--stream-failure-rate", type=float, default=1.0,
                    help="fraction of FAILURE episodes actually appearing in the stream. In "
                         "deployment failures are the rare event, so a monitor mostly sees "
                         "successes and can drift away from detecting the thing it exists to "
                         "detect. 1.0 = every failure is streamed (no rarity pressure). "
                         "Evaluation always uses the FULL held-out set.")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ---------------- load data, one concatenated store with per-task offsets -------------
    D, offs, cur = {}, {}, 0
    for tid, scene in TASKS.items():
        d = EW.load_task(tid, scene, args.stride, args.max_ep)
        if d is None:
            continue
        eps = np.unique(d["ep"])
        rng = np.random.RandomState(7)
        rng.shuffle(eps)
        val_eps = set(eps[: args.n_val].tolist())
        d["is_val"] = np.isin(d["ep"], list(val_eps))
        d["ep_local"] = d["ep"].copy()
        d["ep_global"] = d["ep"] + cur
        offs[tid] = cur
        cur += len(d["y"])
        D[tid] = d
        print(f"   task {tid}: {len(eps)} episodes, {len(d['y'])} frames, "
              f"failure rate {d['y'].mean():.2f}, held out {len(val_eps)}")
    if len(D) < 2:
        print("   need >= 2 tasks (run the recording evals for 32 and 73 first)"); return 1

    W = args.window
    tac_t = torch.as_tensor(np.concatenate([D[t]["tac"] for t in D]))
    rgb_t = torch.as_tensor(np.concatenate([D[t]["rgb"] for t in D]))
    prop_t = torch.as_tensor(np.concatenate([D[t]["prop"] for t in D]))
    y_t = torch.as_tensor(np.concatenate([D[t]["y"] for t in D]))
    ep_t = torch.as_tensor(np.concatenate([D[t]["ep_global"] for t in D]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def gather(gidx, dev_=None):
        dev_ = dev_ or dev
        ii = gidx[:, None] + np.arange(-W + 1, 1)[None, :]
        return (tac_t[ii].to(dev_).float(), prop_t[ii].to(dev_),
                y_t[gidx].to(dev_).float())

    def windows_for(tid, only_val: bool):
        d = D[tid]
        m = d["is_val"] if only_val else ~d["is_val"]
        out = []
        for e in np.unique(d["ep_local"][m]):
            idx = np.where(d["ep_local"] == e)[0]
            if len(idx) >= W:
                out.extend((idx[W - 1:] + offs[tid]).tolist())
        return np.array(out, dtype=np.int64)

    def evaluate(net, tid):
        idx = windows_for(tid, True)
        if len(idx) == 0:
            return float("nan")
        net.eval()
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                tt, pp, yy = gather(b)
                ps.append(torch.sigmoid(net(tt, pp)).cpu().numpy())
                ys.append(yy.cpu().numpy())
        return auc(np.concatenate(ps), np.concatenate(ys))

    class Net(nn.Module):
        """our tactile encoder: semantic-slot set, MAX-pooled (sparse contact), then time;
        proprio is concatenated per frame."""

        def __init__(self, d=64, n_slots=10):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1))
            self.proj = nn.Linear(32, d)
            self.slot = nn.Embedding(n_slots, d)
            self.pproj = nn.Linear(58, 32)
            self.gru = nn.GRU(d + 32, d, batch_first=True)
            self.head = nn.Linear(d, 1)

        def forward(self, tt, pp):
            B, W_, S, G, _ = tt.shape
            f = self.cnn(tt.reshape(B * W_ * S, 1, G, G)).reshape(B * W_, S, -1)
            f = (self.proj(f) + self.slot.weight[None]).max(1).values   # (B*W, d)
            f = f.reshape(B, W_, -1)                                    # back to (B,W,d)
            g = torch.relu(self.pproj(pp))                             # (B,W,32)
            h, _ = self.gru(torch.cat([f, g], -1))
            return self.head(h[:, -1]).squeeze(-1)

    class Buffer:
        def __init__(self, cap, strategy):
            self.cap, self.strategy = cap, strategy
            self.tac, self.pp, self.y, self.w = [], [], [], []

        def add(self, gidx, contact):
            for g in gidx:
                lo = g - W + 1
                self.tac.append(tac_t[lo:g + 1].clone())
                self.pp.append(prop_t[lo:g + 1].clone())
                self.y.append(y_t[g].clone())
                self.w.append(self._weight(float(y_t[g]), contact))
            self._trim()

        def _weight(self, y, contact):
            if self.strategy == "safety":
                # failure and real fingertip contact are what a safety predictor must retain
                return 1.0 + 3.0 * y + 2.0 * float(contact > 0.02)
            return 1.0

        def _trim(self):
            if len(self.tac) <= self.cap:
                return
            if self.strategy == "safety":
                keep = np.argsort(self.w)[::-1][: self.cap]
            else:                                   # reservoir: keep a uniform random subset
                keep = np.random.choice(len(self.tac), self.cap, replace=False)
            keep = sorted(keep.tolist())
            self.tac = [self.tac[i] for i in keep]; self.pp = [self.pp[i] for i in keep]
            self.y = [self.y[i] for i in keep]; self.w = [self.w[i] for i in keep]

        def sample(self, k):
            if not self.tac:
                return None
            n = len(self.tac)
            if self.strategy == "safety":
                pr = np.asarray(self.w, float)
                pr = pr / pr.sum()
                sel = np.random.choice(n, k, replace=True, p=pr)
            else:
                sel = np.random.randint(0, n, k)
            return (torch.stack([self.tac[i] for i in sel]),
                    torch.stack([self.pp[i] for i in sel]),
                    torch.stack([self.y[i] for i in sel]))

    results, summaries = {}, {}
    for strat in args.strategies.split(","):
        torch.manual_seed(0)
        net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        buf = Buffer(args.buffer, strat)
        hist = []
        for tid in D:
            train_eps = [e for e in np.unique(D[tid]["ep_local"])
                         if not D[tid]["is_val"][D[tid]["ep_local"] == e][0]]
            if args.stream_failure_rate < 1.0:
                # keep every success, but let only a fraction of the failures arrive
                # deterministic: `hash()` is randomised per process, which would make the
                # rare-failure result depend on luck rather than on the strategy
                _r = np.random.RandomState((args.seed * 1000 + list(D).index(tid)) % (2 ** 31))
                kept = []
                for e in train_eps:
                    lab = int(D[tid]["y"][D[tid]["ep_local"] == e][0])
                    if lab == 0 or _r.rand() < args.stream_failure_rate:
                        kept.append(e)
                train_eps = kept
            for ei, e in enumerate(train_eps):
                idx = np.where(D[tid]["ep_local"] == e)[0]
                if len(idx) < W:
                    continue
                gidx = idx[W - 1:] + offs[tid]
                net.train()
                for _ in range(args.steps_per_ep):
                    b = gidx[np.random.randint(0, len(gidx), min(args.batch, len(gidx)))]
                    tt, pp, yy = gather(b)
                    if strat != "naive":
                        s = buf.sample(max(1, args.batch // 2))
                        if s is not None:
                            tt = torch.cat([tt, s[0].to(dev)])
                            pp = torch.cat([pp, s[1].to(dev)])
                            yy = torch.cat([yy, s[2].to(dev)])
                    loss = F.binary_cross_entropy_with_logits(net(tt, pp), yy)
                    opt.zero_grad(); loss.backward(); opt.step()
                if strat != "naive" and len(gidx):
                    # gidx is an ARRAY of window-end indices, so slice by its bounds
                    lo = int(gidx.min()) - W + 1
                    hi = int(gidx.max()) + 1
                    contact = float(tac_t[lo:hi].float().mean()) / 255.0
                    buf.add(gidx, contact)
                if ei % args.eval_every == 0 or ei == len(train_eps) - 1:
                    row = {t: evaluate(net, t) for t in D}
                    hist.append({"task": tid, "ep": int(ei), **row})
                    print(f"   [{strat}] task {tid} ep {ei:3d}  " +
                          "  ".join(f"{t}={row[t]:.3f}" for t in D), flush=True)
        results[strat] = hist
        final = {t: hist[-1].get(t, float("nan")) for t in D}
        forget = {}
        for t in D:
            last_t = max(h["ep"] for h in hist if h["task"] == t)
            after = next(h[t] for h in hist if h["task"] == t and h["ep"] == last_t)
            forget[t] = after - final[t]
        summaries[strat] = {"final": final, "forgetting": forget}
        print(f"   [{strat}] final " + "  ".join(f"{t}={final[t]:.3f}" for t in D) +
              "   forgetting " + "  ".join(f"{t}={forget[t]:+.3f}" for t in forget), flush=True)

    out = Path("/home/wangrenpeng/bench2dex/data/online_continual.json")
    out.write_text(json.dumps({"histories": results, "summary": summaries}, indent=2,
                              default=float))
    print(f"\n   SUMMARY (final AUC / forgetting)")
    for s, v in summaries.items():
        print(f"   {s:10s} " + "  ".join(f"{t}:{v['final'][t]:.3f}/{v['forgetting'][t]:+.3f}"
                                         for t in v["final"]))
    print(f"   saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
