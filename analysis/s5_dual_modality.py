"""S5: the dual-modality comparison (tactile array vs force-related channel), reusable.

Data sources
  --source rollouts : /mnt/public/datasets/bench2dex/rollouts/<task>/{success,failure}
                      real policy rollouts, real success/failure labels, tactile + qeffort
                      (qeffort present only after the recorder patch + merge)
  --source variants : theory/variants_out/*.hdf5  (smoke-test source; both channels present,
                      labels are synthetic A/B, so results are a CODE CHECK ONLY)

Modalities, identical frames, identical labels, identical episode-level CV:
  amount  : hand-written contact-amount statistic (baseline)
  torque  : the force-related channel: hand joint efforts (44 joints)
  array   : per-site tactile depth, max-pooled to G x G
  resid   : the array AFTER linearly removing everything the torque channel explains
            -> isolates what the tactile carries beyond the low-dimensional force channel
  fused   : torque + array
"""
import argparse, glob, h5py, json, os
import numpy as np
from pathlib import Path

HAND0, HAND1 = 14, 58          # Sharpa hand joints (arms occupy 0-13)
ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
VAR = Path("/home/wangrenpeng/bench2dex/theory/variants_out")


def _dp(h, G):
    h = h.reshape(240, 240)
    bh, bw = 240 // G, 240 // G
    return h[:bh*G, :bw*G].reshape(G, bh, G, bw).max(axis=(1, 3))


def load(source, tasks, G=20, stride=8, max_eps=None):
    TQ, AR, AM, Y, EID = [], [], [], [], []
    if source == "rollouts":
        items = []
        for t in tasks:
            for oc in ("success", "failure"):
                for p in sorted((ROLL / t / oc).glob("*.hdf5")):
                    items.append((t, oc, p))
    else:
        items = []
        for p in sorted(VAR.glob("*.hdf5")):
            if p.stem in ("base",):
                continue
            items.append(("var", "B" if p.stem.startswith("B") else "A", p))
    ep = 0
    for t, oc, p in items:
        if max_eps is not None and ep >= max_eps:
            break
        try:
            with h5py.File(p, "r") as f:
                if "qeffort" not in f["robot"] or "tactile" not in f["robot"]:
                    print(f"     skip {p.name}: 缺 qeffort 或 tactile"); continue
                dm = f["robot/tactile"]["distance_along_normal_m"]
                sites = list(dm.keys())
                qe = np.asarray(f["robot/qeffort"][:], np.float32)[:, HAND0:HAND1]
                n = int(f["meta/frame_count"][()]) if "meta/frame_count" in f else qe.shape[0]
                n = min(n, qe.shape[0])
                idx = np.arange(0, n, stride)
                ar = np.zeros((len(idx), len(sites), G, G), np.uint8)
                am = np.zeros(len(idx), np.float32)
                for j, s in enumerate(sites):
                    for ii, i in enumerate(idx):
                        h = np.asarray(dm[s][i], np.float32)
                        q = (_dp(h, G) / 0.015 * 255.0).clip(0, 255).astype(np.uint8)
                        ar[ii, j] = q
                        am[ii] += float((h > 0).sum())
        except Exception as e:
            print(f"     !! {p.name}: {type(e).__name__}"); continue
        TQ.append(qe[idx]); AR.append(ar); AM.append(am)
        # label depends on the source: policy rollouts -> failure; variants -> the B family
        # (thumb kept), which is what the A/B contrast is about. Getting this wrong silently
        # produced an all-zero label vector and AUC = nan in the first self-test.
        if source == "rollouts":
            lab = 1 if oc == "failure" else 0
        else:
            lab = 1 if oc == "B" else 0
        Y.append(np.full(len(idx), lab, np.int64))
        EID.append(np.full(len(idx), ep)); ep += 1
    if not TQ:
        raise SystemExit("   没有可用数据")
    return (np.concatenate(TQ), np.concatenate(AR).astype(np.float32)/255.0,
            np.concatenate(AM), np.concatenate(Y), np.concatenate(EID))


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def ori(a): return max(a, 1-a) if np.isfinite(a) else float("nan")


def ep_auc(sc, y, eid):
    a, b = [], []
    for e in np.unique(eid):
        m = eid == e; a.append(np.asarray(sc)[m].mean()); b.append(y[m][0])
    return ori(auc(a, b))


def run(kind, X, y, eid, dev, seed=0, ep_of_frame=None):
    import torch, torch.nn as nn, torch.nn.functional as F
    eps = np.unique(eid); rng = np.random.RandomState(seed); rng.shuffle(eps)
    oof = np.zeros(len(y))
    for va in np.array_split(eps, 5):
        tr = np.setdiff1d(eps, va)
        mt, mv = np.isin(eid, tr), np.isin(eid, va)
        torch.manual_seed(seed)
        if kind == "torque":
            net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 64),
                                nn.ReLU(), nn.Linear(64, 1)).to(dev)
        elif kind == "array":
            net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                                nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(48, 1)).to(dev)
        else:
            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.c = nn.Sequential(nn.Conv2d(X[1].shape[1], 24, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                                        nn.AdaptiveMaxPool2d(1), nn.Flatten())
                    s.m = nn.Sequential(nn.Linear(X[0].shape[1], 64), nn.ReLU())
                    s.h = nn.Sequential(nn.Linear(112, 64), nn.ReLU(), nn.Linear(64, 1))
                def forward(s, a, b):
                    return s.h(torch.cat([s.c(a), s.m(b)], -1)).squeeze(-1)
            net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        yep = np.array([y[eid == e][0] for e in tr]); pos = {e: i for i, e in enumerate(tr)}
        yt = torch.as_tensor([yep[pos[e]] for e in eid[mt]], device=dev).float()
        if kind == "fused":
            At = torch.as_tensor(X[1][mt], device=dev).float(); Bt = torch.as_tensor(X[0][mt], device=dev).float()
            Av = torch.as_tensor(X[1][mv], device=dev).float(); Bv = torch.as_tensor(X[0][mv], device=dev).float()
        else:
            Xt = torch.as_tensor(X[mt], device=dev).float(); Xv = torch.as_tensor(X[mv], device=dev).float()
        ntr = len(yt)
        for _ in range(25):
            perm = torch.randperm(ntr, device=dev)
            for b in range(0, ntr, 128):
                s = perm[b:b+128]
                out = net(At[s], Bt[s]) if kind == "fused" else net(Xt[s]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(out, yt[s])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = net(Av, Bv) if kind == "fused" else net(Xv).squeeze(-1)
            oof[mv] = torch.sigmoid(o).cpu().numpy()
    aa, bb = [], []
    for e in np.unique(eid):
        m = eid == e; aa.append(oof[m].mean()); bb.append(y[m][0])
    return ori(auc(aa, bb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rollouts", choices=["rollouts", "variants"])
    ap.add_argument("--tasks", default="26,32,73")
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--out", default="/home/wangrenpeng/bench2dex/theory/s5_results.json")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    TQ, AR, AM, Y, EID = load(a.source, a.tasks.split(","), a.grid, a.stride, a.max_eps)
    print(f"   载入: {len(Y)} 帧, {len(np.unique(EID))} 集, 失败率 {Y.mean():.2f}")
    A = np.concatenate([TQ, np.ones((len(TQ), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, AR.reshape(len(AR), -1), rcond=None)
    resid = (AR.reshape(len(AR), -1) - A @ coef).reshape(AR.shape)
    ARc = AR.transpose(0, 2, 3, 1); RSc = resid.transpose(0, 2, 3, 1)
    res = {"source": a.source, "n_frames": int(len(Y)), "n_eps": int(len(np.unique(EID))),
           "fail_rate": float(Y.mean()),
           "auc_amount": ep_auc(-AM, Y, EID),
           "auc_torque": run("torque", TQ, Y, EID, dev),
           "auc_array": run("array", ARc, Y, EID, dev),
           "auc_array_resid": run("array", RSc, Y, EID, dev),
           "auc_fused": run("fused", (TQ, ARc), Y, EID, dev)}
    for k, v in res.items():
        if k.startswith("auc_"):
            print(f"     {k:18s} {v:.3f}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"   saved {a.out}")


if __name__ == "__main__":
    main()
