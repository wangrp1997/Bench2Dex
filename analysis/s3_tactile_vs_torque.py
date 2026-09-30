"""S3: the dual-modality comparison in SIMULATION, done properly.

Earlier attempts (S1/S2) synthesised a "force" modality as a projection of the tactile field,
which is circular. The simulator in fact provides a genuinely independent, physics-computed
force-related channel: robot/qeffort, the joint torques. The hand's 44 joints are the analogue
of the low-dimensional force channel that real hands have (joint-torque / current sensing),
while the tactile array is the high-dimensional local one -- the same contrast structure as
3-axis force vs array on the user's real fingertips.

Modalities, same frames, same real failure labels, same episode-level CV:
    torque  : 44 hand-joint efforts
    array   : per-site depth, max-pooled to 20x20
    fused   : both
plus the hand-written contact-amount statistic, and a residual test that asks whether the
tactile carries information BEYOND what the torque channel already provides.
"""
import h5py, numpy as np, json
from pathlib import Path

ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
SENS = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor"
G, HAND0, HAND1 = 20, 14, 58
TASKS = ["26", "32", "73"]


def load(task, stride=4):
    TQ, AR, AM, Y, EID = [], [], [], [], []
    ep = 0
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task / outcome).glob("*.hdf5")):
            try:
                with h5py.File(p, "r") as f:
                    dm = f["robot/tactile/distance_along_normal_m"]
                    qe = np.asarray(f["robot/qeffort"][:], np.float32)[:, HAND0:HAND1]
                    sites = list(dm.keys())
                    n = int(f["meta/frame_count"][()]); idx = np.arange(0, n, stride)
                    ar = np.zeros((len(idx), len(sites), G, G), np.float32)
                    am = np.zeros(len(idx), np.float32)
                    for j, s in enumerate(sites):
                        for ii, i in enumerate(idx):
                            h = np.asarray(dm[s][i], np.float32).reshape(240, 240)
                            bh, bw = 240 // G, 240 // G
                            ar[ii, j] = h[:bh*G, :bw*G].reshape(G, bh, G, bw).max(axis=(1, 3))
                            am[ii] += float((h > 0).sum())
            except Exception as e:
                print(f"     !! {p.name}: {type(e).__name__}"); continue
            TQ.append(qe[idx]); AR.append(ar); AM.append(am)
            Y.append(np.full(len(idx), 1 if outcome == "failure" else 0, np.int64))
            EID.append(np.full(len(idx), ep)); ep += 1
    return (np.concatenate(TQ), np.concatenate(AR), np.concatenate(AM),
            np.concatenate(Y), np.concatenate(EID))


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def ori(a): return max(a, 1-a) if np.isfinite(a) else float("nan")


def ep_auc(sc, y, eid):
    a, b = [], []
    for e in np.unique(eid):
        m = eid == e; a.append(sc[m].mean()); b.append(y[m][0])
    return ori(auc(a, b))


def run(kind, X, y, eid, dev):
    import torch, torch.nn as nn, torch.nn.functional as F
    eps = np.unique(eid); rng = np.random.RandomState(0); rng.shuffle(eps)
    oof = np.zeros(len(y))
    for va in np.array_split(eps, 5):
        tr = np.setdiff1d(eps, va)
        mt, mv = np.isin(eid, tr), np.isin(eid, va)
        torch.manual_seed(0)
        if kind == "torque":
            net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 64),
                                nn.ReLU(), nn.Linear(64, 1)).to(dev)
        elif kind == "array":
            net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                                nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(48, 1)).to(dev)
        else:  # fused
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
        for _ in range(25):
            perm = torch.randperm(len(yt), device=dev)
            for b in range(0, len(yt), 128):
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
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for t in TASKS:
        TQ, AR, AM, Y, EID = load(t)
        print(f"   task {t}: {len(Y)} 帧, {len(np.unique(EID))} 集")
        # residualise the array on the torque channel (linear), to ask what the tactile
        # carries BEYOND the low-dimensional force-related channel
        A = np.concatenate([TQ, np.ones((len(TQ), 1))], 1)
        coef, *_ = np.linalg.lstsq(A, AR.reshape(len(AR), -1), rcond=None)
        resid = (AR.reshape(len(AR), -1) - A @ coef).reshape(AR.shape)
        r = {"auc_amount": ep_auc(-AM, Y, EID),
             "auc_torque": run("torque", TQ, Y, EID, dev),
             "auc_array": run("array", AR.transpose(0, 2, 3, 1), Y, EID, dev),
             "auc_array_resid": run("array", resid.transpose(0, 2, 3, 1), Y, EID, dev),
             "auc_fused": run("fused", (TQ, AR.transpose(0, 2, 3, 1)), Y, EID, dev)}
        res[t] = r
        print(f"     量统计量         AUC={r['auc_amount']:.3f}")
        print(f"     力矩(44维,学习)  AUC={r['auc_torque']:.3f}")
        print(f"     阵列(学习)       AUC={r['auc_array']:.3f}")
        print(f"     阵列【去掉力矩可解释部分】(残差) AUC={r['auc_array_resid']:.3f}")
        print(f"     融合             AUC={r['auc_fused']:.3f}")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/s3_results.json", "w"), indent=1)
    print("\n   saved theory/s3_results.json")


if __name__ == "__main__":
    main()
