#!/usr/bin/env python3
"""SPEC-DRIVEN MILESTONE INFEASIBILITY CERTIFICATE, with a calibrated false-alarm rate.

What is being certified
-----------------------
At frame t the task specification (its executable stage predicates, in dependency order)
says the next thing that must happen is stage s(t). The certificate is the claim

    "stage s(t) will NOT be achieved for the rest of this episode"

and it is issued when the monitor's score exceeds a threshold tau that is CALIBRATED so
that the false-certificate rate (certificates issued on frames where the stage DOES get
achieved) stays at or below alpha. That is what makes it a certificate rather than a guess.

This is task-agnostic: s(t) comes from the scene YAML, not from any assumption about
grasping or about a "contact deficit". 26/26 Bench2Dex tasks ship such predicates.

Reported per task
-----------------
  AUC                : ranking quality for the milestone target (episode-level CV)
  false-cert rate    : measured at the calibrated tau (target alpha)
  coverage           : fraction of truly-failing frames that get certified
  lead time          : in failing episodes, frames between the FIRST certificate and the end
                       (per episode) -- the quantity a safety system cares about

Baselines: the hand-crafted contact statistic, calibrated identically, and a
"certify everything" degenerate reference.

Usage: 120_milestone_certificate.py [--alpha 0.1] [--epochs 25] [--folds 5]
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
ML = Path("/home/wangrenpeng/bench2dex/data/milestone_labels")


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def calib_tau(scores, labels, alpha):
    """Smallest tau whose false-certificate rate on `labels` is <= alpha."""
    order = np.argsort(-scores)
    s, l = np.asarray(scores)[order], np.asarray(labels)[order]
    neg = l <= 0
    # sweep candidate thresholds from high score to low
    best = float("inf")
    fp = 0
    n_neg = max(int(neg.sum()), 1)
    for i in range(len(s)):
        if not neg[i]:
            fp += 1
        if fp / n_neg > alpha:
            break
        best = s[i]
    return best if np.isfinite(best) else float(s.min()) - 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--model-seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = args.window

    # ---------------- load tactile features (EW) + milestone labels ----------------
    D = {}
    for tid in ("26", "32", "73"):
        d = EW.load_task(tid, TASKS[tid], args.stride, 60)
        if d is None:
            continue
        # milestone labels must follow the SAME episode order load_task used
        ym, si = [], []
        import glob
        order = []
        for outcome in ("success", "failure"):
            for p in sorted(glob.glob(str(Path("/mnt/public/datasets/bench2dex/rollouts") / tid / outcome / "*.hdf5"))):
                order.append((outcome, Path(p).stem))
        for k, (outcome, stem) in enumerate(order):
            z = np.load(ML / tid / outcome / f"{stem}.npz", allow_pickle=True)
            n0 = len(z["y_milestone"])
            ym.append(np.asarray(z["y_milestone"], np.int64)[::args.stride])
            si.append(np.asarray(z["s_index"], np.int64)[::args.stride])
        # align lengths with the feature arrays
        n = len(d["y"])
        y_m = np.concatenate(ym)[:n]
        s_i = np.concatenate(si)[:n]
        if len(y_m) < n:
            y_m = np.pad(y_m, (0, n - len(y_m)))
            s_i = np.pad(s_i, (0, n - len(s_i)))
        d["y_milestone"] = y_m
        d["s_index"] = s_i
        D[tid] = d
        print(f"   task {tid}: {len(np.unique(d['ep']))} episodes, {n} frames, "
              f"milestone-failure rate {y_m.mean():.3f}")

    tids = list(D)
    tac_l, ym_l, ep_l, tk_l = [], [], [], []
    cur = 0
    for ti, t in enumerate(tids):
        d = D[t]
        tac_l.append(d["tac"]); ym_l.append(d["y_milestone"])
        ep_l.append(d["ep"] + cur); tk_l.append(np.full(len(d["y_milestone"]), ti))
        cur += int(d["ep"].max()) + 1
    tac_t = torch.as_tensor(np.concatenate(tac_l))
    ym_t = torch.as_tensor(np.concatenate(ym_l))
    ep_t = torch.as_tensor(np.concatenate(ep_l))
    tk_t = torch.as_tensor(np.concatenate(tk_l))
    ep_np, ym_np, tk_np = ep_t.numpy(), ym_t.numpy(), tk_t.numpy()

    folds = []
    for t in tids:
        e = np.unique(ep_np[tk_np == tids.index(t)])
        rng = np.random.RandomState(11); rng.shuffle(e)
        folds.append(np.array_split(e, args.folds))

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
            f = self.proj(f) + self.slot.weight[None]
            f = f.mean(1).reshape(B, W_, -1)          # mean over SITES: count matters
            h, _ = self.gru(f)
            return self.head(h[:, -1]).squeeze(-1)

    def ep_windows(e):
        idx = np.where(ep_np == e)[0]
        return idx[idx >= W - 1][W - 1::max(1, args.stride)]

    def train_model(eps, seed):
        torch.manual_seed(seed)
        net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        tr = {int(e): ep_windows(e) for e in eps}
        tr = {e: g for e, g in tr.items() if len(g)}
        keys = list(tr)
        for _ in range(args.epochs):
            net.train()
            for _ in range(args.steps_per_epoch):
                sel = np.random.choice(keys, min(args.batch, len(keys)), replace=False)
                xs, ys = [], []
                for e in sel:
                    g = tr[e]
                    if len(g) > 32:
                        g = g[np.random.choice(len(g), 32, replace=False)]
                    ii = g[:, None] + np.arange(-W + 1, 1)[None, :]
                    xs.append(tac_t[ii].to(dev).float()); ys.append(ym_t[g].to(dev).float())
                x = torch.cat(xs); y = torch.cat(ys)
                loss = F.binary_cross_entropy_with_logits(net(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
        return net

    def score(net, eps):
        net.eval()
        idx = np.concatenate([ep_windows(e) for e in eps]) if len(eps) else np.zeros(0, int)
        if len(idx) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0, int)
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                ii = b[:, None] + np.arange(-W + 1, 1)[None, :]
                outs.append(torch.sigmoid(net(tac_t[ii].to(dev).float())).cpu().numpy())
        return np.concatenate(outs), ym_np[idx], ep_np[idx]

    def stat_score(eps):
        """mean number of contacting SITES per frame, averaged over the window.
        NOTE: the site dimension must be SUMMED (it is a count); an earlier version
        forgot this and produced a (B, S) score, which silently changed the statistic."""
        idx = np.concatenate([ep_windows(e) for e in eps]) if len(eps) else np.zeros(0, int)
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        c = (tac_t[ii].float() > 0).float()              # (B,W,S,G,G)
        site_on = (c.mean(dim=(3, 4)) > 0).float()       # (B,W,S)
        n_sites = site_on.sum(dim=2)                     # (B,W)  <- count over sites
        ns = n_sites.mean(dim=1)                         # (B,)
        return (-ns.numpy()), ym_np[idx], ep_np[idx]

    # global index -> task, for the per-task breakdown
    tk_of = tk_np

    # A threshold calibrated on the SAME episodes the model was fit on does not transfer
    # (measured: val false-certificate rate 0.447 against a target of 0.15), so each fold
    # splits its training episodes into a fit part and a held-out CALIBRATION part.
    results = {"tactile": [], "contact_stat": []}
    taus = {"tactile": [], "contact_stat": []}
    for k in range(args.folds):
        val_eps = np.concatenate([folds[i][k] for i in range(len(tids))])
        tr_eps = np.setdiff1d(ep_np, val_eps)
        rng = np.random.RandomState(100 + k)
        tr_eps = rng.permutation(tr_eps)
        n_cal = max(len(tids), int(0.3 * len(tr_eps)))
        cal_eps, fit_eps = tr_eps[:n_cal], tr_eps[n_cal:]
        net = train_model(fit_eps, args.model_seed)
        for name, fit_fn, cal_fn in (
                ("tactile", lambda e: score(net, e), lambda e: score(net, e)),
                ("contact_stat", lambda e: stat_score(e), lambda e: stat_score(e))):
            cs, cl, _ = cal_fn(cal_eps)
            tau = calib_tau(cs, cl, args.alpha)
            taus[name].append(float(tau))
            sc, lb, ep_ = fit_fn(val_eps)
            results[name].append((sc, lb, ep_, tau))
        print(f"   fold {k+1}/{args.folds}  tau tactile={taus['tactile'][-1]:.3f} "
              f"stat={taus['contact_stat'][-1]:.3f}", flush=True)

    print(f"\n   calibrated at alpha={args.alpha} on a held-out calibration split")
    print(f"   {'monitor':14s} {'AUC':7s} {'falseCert':>10s} {'coverage':>9s} "
          f"{'lead_med':>9s} {'cert_eps':>9s} {'n_frames':>9s}")
    table = {}
    for name, rows in results.items():
        sc = np.concatenate([r[0] for r in rows])
        lb = np.concatenate([r[1] for r in rows])
        ep = np.concatenate([r[2] for r in rows])
        cert = np.concatenate([r[0] >= r[3] for r in rows])
        neg, pos = lb <= 0, lb > 0
        row = {"auc": auc(sc, lb),
               "false_cert_rate": float(cert[neg].mean()) if neg.any() else float("nan"),
               "coverage": float(cert[pos].mean()) if pos.any() else float("nan"),
               "tau_mean": float(np.mean(taus[name]))}
        # lead time: within each failing episode, the number of FRAMES between the FIRST
        # certificate and the episode end. Positive by construction.
        leads, n_cert_eps, n_fail_eps = [], 0, 0
        for e in np.unique(ep):
            m = ep == e
            if not (lb[m] > 0).any():
                continue
            n_fail_eps += 1
            c = np.where(cert[m])[0]
            if len(c) == 0:
                continue
            n_cert_eps += 1
            leads.append(int(m.sum() - c[0]))
        row["lead_median"] = float(np.median(leads)) if leads else float("nan")
        row["cert_eps"] = f"{n_cert_eps}/{n_fail_eps}"
        row["n_frames"] = int(len(sc))
        table[name] = row
        print(f"   {name:14s} {row['auc']:7.3f} {row['false_cert_rate']:10.3f} "
              f"{row['coverage']:9.3f} {row['lead_median']:9.1f} {row['cert_eps']:>9s} "
              f"{row['n_frames']:9d}")

    out = Path("/home/wangrenpeng/bench2dex/data/milestone_certificate.json")
    out.write_text(json.dumps({"alpha": args.alpha, "taus": taus, "table": table},
                              indent=2, default=float))
    print(f"\n   saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
