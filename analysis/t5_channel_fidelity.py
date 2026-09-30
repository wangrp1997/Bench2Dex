"""T5': is the CHANNEL the bottleneck for tactile monitoring?

Two channels are recorded for every rollout:
  A  tacmap                    : 8-bit image, what the sensor actually emits
  B  distance_along_normal_m   : full-precision penetration depth (metres)

DATA-5 showed these are NOT the same quantity (fit slope 0.5-10, offset up to 7.4 mm,
residual RMS up to 2.2 mm). If the realised channel is the binding constraint, then a
monitor given channel B should clearly beat one given channel A -- and the one-line
statistic should improve too. If B ~ A, the channel is not the bottleneck and the task is.

Same encoder, same folds, same episode-level objective in every arm.
Arms: tacmap-learned, depth-learned, tacmap-stat (contact-site count), depth-stat (mean depth).
"""
import sys, glob, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/wangrenpeng/bench2dex/_deploy_logs")
from importlib.machinery import SourceFileLoader
EW = SourceFileLoader("ew", "/home/wangrenpeng/bench2dex/_deploy_logs/107_failure_early_warning.py").load_module()

ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
LAB = Path("/home/wangrenpeng/bench2dex/data/rollout_labels")
G = 24
W = 6


def load(task_id, scene, stride=4):
    import h5py
    T, D, Y, EP = [], [], [], []
    ep = 0
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task_id / outcome).glob("*.hdf5")):
            lab = LAB / scene / outcome / (p.stem + ".npz")
            if not lab.exists():
                continue
            try:
                with h5py.File(p, "r") as f:
                    n = int(f["meta/frame_count"][()])
                    tm = f["robot/tactile"]["tacmap"]
                    dm = f["robot/tactile"]["distance_along_normal_m"]
                    sites = list(tm.keys())
                    idx = np.arange(0, n, stride)
                    tac = np.zeros((len(idx), len(sites), G, G), np.uint8)
                    dep = np.zeros((len(idx), len(sites), G, G), np.float32)
                    for j, i in enumerate(idx):
                        for si, s in enumerate(sites):
                            a = np.asarray(tm[s][i], np.uint8)
                            bh, bw = a.shape[0] // G, a.shape[1] // G
                            tac[j, si] = a[:bh*G, :bw*G].reshape(G, bh, G, bw).max(axis=(1, 3))
                            b = np.asarray(dm[s][i], np.float32)
                            dep[j, si] = b[:bh*G, :bw*G].reshape(G, bh, G, bw).mean(axis=(1, 3))
            except Exception as exc:
                print(f"   !! {p.name}: {type(exc).__name__}")
                continue
            T.append(tac); D.append(dep)
            Y.append(np.full(len(idx), 1 if outcome == "failure" else 0, np.int64))
            EP.append(np.full(len(idx), ep)); ep += 1
    return (np.concatenate(T), np.concatenate(D), np.concatenate(Y), np.concatenate(EP))


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [("26", "26_canned_food_tray_line_arrangement"),
             ("32", "32_baking_tray_prep_with_tools"),
             ("73", "73_jigsaw_puzzle_assembly")]
    D = {}
    for tid, sc in tasks:
        d = load(tid, sc)
        D[tid] = d
        print(f"   task {tid}: {len(np.unique(d[3]))} eps, {len(d[2])} frames, "
              f"failrate {d[2].mean():.2f}")

    def auc(s, l):
        s, l = np.asarray(s, float), np.asarray(l)
        p, n = s[l > 0], s[l <= 0]
        if len(p) == 0 or len(n) == 0:
            return float("nan")
        return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())

    class Net(nn.Module):
        def __init__(s_, cin=1, d=64, ns=10):
            super().__init__()
            s_.cnn = nn.Sequential(nn.Conv2d(cin, 16, 3, 2, 1), nn.ReLU(),
                                   nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveMaxPool2d(1))
            s_.proj = nn.Linear(32, d); s_.slot = nn.Embedding(ns, d)
            s_.gru = nn.GRU(d, d, batch_first=True); s_.head = nn.Linear(d, 1)

        def forward(s_, x):
            B, W_, S, G_, _ = x.shape
            f = s_.cnn(x.reshape(B*W_*S, x.shape[-3] if False else 1, G_, G_)).reshape(B*W_, S, -1)
            f = (s_.proj(f) + s_.slot.weight[None]).mean(1).reshape(B, W_, -1)
            h, _ = s_.gru(f)
            return s_.head(h[:, -1]).squeeze(-1)

    def run(channel):
        out = {}
        for tid, _ in tasks:
            tac, dep, y, ep = D[tid]
            X = (tac.astype(np.float32) / 255.0) if channel == "tacmap" else dep
            X = np.stack([X], axis=0)  # placeholder to keep shapes obvious
            X = X[0]
            eps = np.unique(ep)
            rng = np.random.RandomState(11); rng.shuffle(eps)
            folds = np.array_split(eps, 5)
            P, Y, E = [], [], []
            for k in range(5):
                val = folds[k]; tr = np.setdiff1d(eps, val)
                def wins(es):
                    o = []
                    for e in es:
                        i = np.where(ep == e)[0]
                        if len(i) >= W:
                            o.extend(i[W-1::4].tolist())
                    return np.array(o, dtype=np.int64)
                tri, vai = wins(tr), wins(val)
                torch.manual_seed(0)
                net = Net().to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
                byep = {int(e): wins([e]) for e in tr}
                byep = {e: g for e, g in byep.items() if len(g)}
                keys = list(byep); lab = {e: float(y[byep[e][0]]) for e in keys}
                for _ in range(25):
                    net.train()
                    for _ in range(12):
                        sel = np.random.choice(keys, min(8, len(keys)), replace=False)
                        xs, ys = [], []
                        for e in sel:
                            g = byep[e]
                            if len(g) > 32:
                                g = g[np.random.choice(len(g), 32, replace=False)]
                            ii = g[:, None] + np.arange(-W+1, 1)[None, :]
                            xs.append(torch.as_tensor(X[ii], device=dev).float())
                            ys.append(torch.full((len(g),), lab[e], device=dev))
                        loss = F.binary_cross_entropy_with_logits(net(torch.cat(xs)), torch.cat(ys))
                        opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    outs = []
                    for i in range(0, len(vai), 256):
                        b = vai[i:i+256]
                        ii = b[:, None] + np.arange(-W+1, 1)[None, :]
                        outs.append(torch.sigmoid(net(torch.as_tensor(X[ii], device=dev).float())).cpu().numpy())
                P.append(np.concatenate(outs)); Y.append(y[vai]); E.append(ep[vai])
            P, Y, E = np.concatenate(P), np.concatenate(Y), np.concatenate(E)
            sc, lb = [], []
            for e in np.unique(E):
                m = E == e; sc.append(P[m].mean()); lb.append(Y[m][0])
            out[tid] = auc(sc, lb)
        return out

    def stat(channel):
        out = {}
        for tid, _ in tasks:
            tac, dep, y, ep = D[tid]
            if channel == "tacmap":
                s = (tac > 0).reshape(len(tac), tac.shape[1], -1).any(2).sum(1).astype(float)
            else:
                s = dep.reshape(len(dep), dep.shape[1], -1).max(2).max(1).astype(float)
            sc, lb = [], []
            for e in np.unique(ep):
                m = ep == e; sc.append(-s[m].mean()); lb.append(y[m][0])
            out[tid] = auc(sc, lb)
        return out

    res = {}
    for name, fn in (("tacmap-learned", lambda: run("tacmap")),
                     ("depth-learned", lambda: run("depth")),
                     ("tacmap-stat", lambda: stat("tacmap")),
                     ("depth-stat", lambda: stat("depth"))):
        r = fn(); res[name] = r
        print(f"   {name:16s} " + "  ".join(f"{t}={r[t]:.3f}" for t in r) +
              f"   mean={np.nanmean(list(r.values())):.3f}", flush=True)
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/t5_results.json", "w"), indent=1)
    print("\n   saved theory/t5_results.json")


if __name__ == "__main__":
    main()
