#!/usr/bin/env python3
"""Does the safety monitor need to KNOW THE TASK?

Motivation (measured, not assumed)
----------------------------------
Pooling tasks 26/32/73 into one tactile failure predictor gives per-task episode AUC
    task 26: 0.810   task 32: 0.840   task 73: 0.266   <- 73 is INVERTED
The reason is that the tactile signature of failure differs in STRENGTH by task:
failures touch less than successes, but the ratio is 0.50 (task 26), 0.62 (32), 0.86 (73).
A pooled boundary is dominated by 26/32 and therefore mis-scores 73.

Hypothesis under test
---------------------
Task conditioning should restore task 73 toward its own ceiling. Note the ceiling is set
by the FEATURES, not by the model: the hand-crafted contact statistic itself only reaches
  task 73: 0.547   (vs 0.738 / 0.800 on 26 / 32)
so conditioning cannot be expected to reach 0.7+; the informative question is how much of
the 0.266 -> 0.547 gap it closes.

Variants (identical encoder, identical folds)
---------------------------------------------
  pooled     : one model on all tasks, no task information      (= current baseline)
  cond       : same, plus a learned per-task embedding
  per_task   : three independent models (upper bound for conditioning)
  feature    : the hand-crafted contact statistic (no training; feature ceiling)

Metric: episode-level AUC per task, 5-fold episode-level CV, identical folds throughout.

Usage: 115_task_conditioning.py [--epochs 25]
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


def ep_auc(pred, ep, y):
    sc, lb = [], []
    for e in np.unique(ep):
        m = ep == e
        sc.append(pred[m].mean()); lb.append(y[m][0])
    return auc(sc, lb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-ep", type=int, default=60)
    ap.add_argument("--model-seed", type=int, default=0,
                    help="model-init seed. Task-level AUC on 20 episodes has a huge "
                         "run-to-run spread (measured 0.47-0.80 for the SAME architecture), "
                         "so any claim needs several seeds.")
    ap.add_argument("--slot-aggs", default="max,mean",
                    help="how to aggregate the per-site features. max discards HOW MANY "
                         "fingertips are in contact, which is the actual failure signal "
                         "(the hand-crafted statistic is exactly that count), so max is "
                         "expected to underperform.")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ---------------- load, keeping a per-task index ----------------
    D = {}
    for tid in ("26", "32", "73"):
        d = EW.load_task(tid, TASKS[tid], args.stride, args.max_ep)
        if d is None:
            print(f"   (no data for task {tid})"); continue
        D[tid] = d
        print(f"   task {tid}: {len(np.unique(d['ep']))} episodes, {len(d['y'])} frames, "
              f"failure rate {d['y'].mean():.2f}")
    if len(D) < 2:
        print("   need >= 2 tasks"); return 1
    tids = list(D)
    t2i = {t: i for i, t in enumerate(tids)}

    W = args.window
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # concatenate with global episode ids and a task label per frame
    tac, yy, ep, tk, offs = [], [], [], [], {}
    cur = 0
    for t in tids:
        d = D[t]
        offs[t] = cur
        tac.append(d["tac"]); yy.append(d["y"])
        ep.append(d["ep"] + cur); tk.append(np.full(len(d["y"]), t2i[t]))
        cur += len(d["y"])
    tac_t = torch.as_tensor(np.concatenate(tac))
    y_t = torch.as_tensor(np.concatenate(yy))
    ep_t = torch.as_tensor(np.concatenate(ep))
    tk_t = torch.as_tensor(np.concatenate(tk))

    # identical folds for every variant: shuffle each task's episodes, split into 5
    folds = {}
    for t in tids:
        e = np.unique(D[t]["ep"])
        rng = np.random.RandomState(11)
        rng.shuffle(e)
        folds[t] = np.array_split(e + offs[t], args.folds)

    def windows_of(eps):
        out = []
        for e in eps:
            idx = np.where(ep_t.numpy() == e)[0]
            if len(idx) >= W:
                out.extend(idx[W - 1::max(1, args.stride)].tolist())
        return np.array(out, dtype=np.int64)

    def gather(idx):
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        return tac_t[ii].to(dev).float(), y_t[idx].to(dev).float(), tk_t[idx].to(dev).long()

    class Net(nn.Module):
        def __init__(self, d=64, n_tasks=0, n_slots=10, agg="max"):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1))
            self.proj = nn.Linear(32, d)
            self.slot = nn.Embedding(n_slots, d)
            self.task_emb = nn.Embedding(n_tasks, d) if n_tasks > 0 else None
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, 1)
            self.agg = agg

        def forward(self, x, task=None):
            x = x / 255.0
            B, W_, S, G, _ = x.shape
            f = self.cnn(x.reshape(B * W_ * S, 1, G, G)).reshape(B * W_, S, -1)
            f = self.proj(f) + self.slot.weight[None]                       # (B*W, S, d)
            f = f.max(1).values if self.agg == "max" else \
                (f.mean(1) if self.agg == "mean" else f.sum(1))
            f = f.reshape(B, W_, -1)
            if self.task_emb is not None and task is not None:
                f = f + self.task_emb(task)[:, None, :]                    # per-task context
            h, _ = self.gru(f)
            return self.head(h[:, -1]).squeeze(-1)

    def train_model(train_eps, n_tasks, agg="max"):
        torch.manual_seed(args.model_seed)
        net = Net(n_tasks=n_tasks, agg=agg).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        tr = {}
        for e in train_eps:
            idx = np.where(ep_t.numpy() == e)[0]
            g = idx[idx >= W - 1][W - 1::max(1, args.stride)]
            if len(g):
                tr[int(e)] = g
        keys = list(tr)
        if not keys:
            return net
        lab = {e: float(y_t[tr[e][0]]) for e in keys}
        tsk = {e: int(tk_t[tr[e][0]]) for e in keys}
        for _ in range(args.epochs):
            net.train()
            for _ in range(args.steps_per_epoch):
                sel = np.random.choice(keys, min(args.batch, len(keys)), replace=False)
                logs, labs = [], []
                for e in sel:
                    g = tr[e]
                    if len(g) > 32:
                        g = g[np.random.choice(len(g), 32, replace=False)]
                    x, _, _ = gather(g)
                    tvec = torch.full((len(g),), tsk[e], device=dev, dtype=torch.long)
                    logs.append(net(x, tvec).mean()); labs.append(lab[e])
                loss = F.binary_cross_entropy_with_logits(
                    torch.stack(logs), torch.tensor(labs, device=dev, dtype=torch.float32))
                opt.zero_grad(); loss.backward(); opt.step()
        return net

    def predict(net, eps, n_tasks):
        net.eval()
        idx = windows_of(eps)
        if len(idx) == 0:
            return np.zeros(0), np.zeros(0, int), np.zeros(0)
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                x, _, tvec = gather(idx[i:i + 256])
                outs.append(torch.sigmoid(net(x, tvec if n_tasks else None)).cpu().numpy())
        return np.concatenate(outs), ep_t[idx].numpy(), y_t[idx].numpy()

    def contact_stat(eps):
        idx = windows_of(eps)
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        c = (tac_t[ii].float() > 0).float()
        ns = (c.mean(dim=(3, 4)) > 0).float().mean(dim=1)
        return (-ns.numpy()), ep_t[idx].numpy(), y_t[idx].numpy()

    # task index of each episode, needed to keep the per-task models task-pure
    ep_all = ep_t.numpy()
    tk_of_ep = {}
    for t in tids:
        d = D[t]
        for e in np.unique(d["ep"]):
            tk_of_ep[int(e) + offs[t]] = t2i[t]

    aggs = args.slot_aggs.split(",")
    variants = [f"pooled_{a}" for a in aggs] + [f"cond_{a}" for a in aggs] + \
               ["per_task", "feature"]
    results = {k: {} for k in variants}
    for k in range(args.folds):
        # validation = fold k of EVERY task, removed from training as a whole.
        # (The previous version removed only one task's fold at a time and concatenated,
        # which put the other tasks' validation folds straight back into training --
        # that leak made the pooled model score 0.95 by memorising the val episodes.)
        val_all = np.concatenate([folds[t][k] for t in tids])
        tr_all = np.setdiff1d(ep_all, val_all)
        for a in aggs:
            for name, ntask in ((f"pooled_{a}", 0), (f"cond_{a}", len(tids))):
                net = train_model(tr_all, ntask, agg=a)
                for t in tids:
                    p, e_, y_ = predict(net, folds[t][k], ntask)
                    results[name].setdefault(t, []).append((p, e_, y_))
        # ---- per task: trained ONLY on that task's other folds ----
        for t in tids:
            tr_t = np.array([e for e in ep_all if tk_of_ep[int(e)] == t2i[t]
                             and e not in set(folds[t][k].tolist())])
            net = train_model(tr_t, 0)
            p, e_, y_ = predict(net, folds[t][k], 0)
            results["per_task"].setdefault(t, []).append((p, e_, y_))
        # ---- feature ceiling ----
        for t in tids:
            p, e_, y_ = contact_stat(folds[t][k])
            results["feature"].setdefault(t, []).append((p, e_, y_))
        print(f"   fold {k + 1}/{args.folds} done", flush=True)

    print(f"\n   {'variant':10s}" + "".join(f"{('task ' + t):>10s}" for t in tids) + f"{'mean':>10s}")
    table = {}
    for name in variants:
        row = {}
        for t in tids:
            ps = np.concatenate([a for a, _, _ in results[name][t]])
            es = np.concatenate([a for _, a, _ in results[name][t]])
            ys = np.concatenate([a for _, _, a in results[name][t]])
            row[t] = ep_auc(ps, es, ys)
        table[name] = row
        print(f"   {name:10s}" + "".join(f"{row[t]:10.3f}" for t in tids)
              + f"{np.nanmean(list(row.values())):10.3f}")

    print("\n   task 73 across variants: " + "  ".join(
        f"{k}={table[k]['73']:.3f}" for k in variants))
    out = Path("/home/wangrenpeng/bench2dex/data/task_conditioning.json")
    out.write_text(json.dumps(table, indent=2, default=float))
    print(f"   saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
