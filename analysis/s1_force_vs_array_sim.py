"""S1: the dual-modality experiment in SIMULATION, using the existing real rollouts.

Key structural fact: in simulation a 3-axis force sensor is a PROJECTION of the dense
tactile field,

        F_site = k * sum_j h_j * n_j          (h_j = penetration depth, n_j = surface normal)

so the force modality is a function of the array modality by construction. That makes the
simulation version a CLEAN information-theoretic test -- no hardware confound at all -- of
the claim that the coarse modality provably loses what the fine one carries.

Modalities compared, on the SAME frames and the SAME real failure labels:
    force : the 3-vectors of all 10 sites (30 numbers)
    array : the per-site depth maps, downsampled to 20x20 (4000 numbers)
plus the hand-written contact-amount statistic, and a matched-net-force stratification.
"""
import h5py, numpy as np, glob, os, json
from pathlib import Path

ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
SENS = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor"
G = 20
TASKS = ["26", "32", "73"]


def site_geometry():
    """per-site surface points and normals in metres, keyed as the HDF5 site names"""
    P4 = np.load(f"{SENS}/tactileSensor_map_4F_point.npy").astype(np.float64) * 1e-3
    N4 = np.load(f"{SENS}/tactileSensor_map_4F_normal.npy").astype(np.float64)
    PT = np.load(f"{SENS}/tactileSensor_map_TH_point.npy").astype(np.float64) * 1e-3
    NT = np.load(f"{SENS}/tactileSensor_map_TH_normal.npy").astype(np.float64)
    out = {}
    for site in ("left_index_elastomer", "left_middle_elastomer", "left_ring_elastomer",
                 "left_pinky_elastomer", "right_index_elastomer", "right_middle_elastomer",
                 "right_ring_elastomer", "right_pinky_elastomer"):
        out[site] = (P4.reshape(-1, 3), N4.reshape(-1, 3))
    for site in ("left_thumb_elastomer", "right_thumb_elastomer"):
        out[site] = (PT.reshape(-1, 3), NT.reshape(-1, 3))
    return out


def load(task, stride=4):
    geo = site_geometry()
    FOR, ARR, AMT, Y, EID = [], [], [], [], []
    ep = 0
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task / outcome).glob("*.hdf5")):
            try:
                with h5py.File(p, "r") as f:
                    dm = f["robot/tactile/distance_along_normal_m"]
                    sites = list(dm.keys())
                    n = int(f["meta/frame_count"][()])
                    idx = np.arange(0, n, stride)
                    fv = np.zeros((len(idx), len(sites) * 3), np.float32)
                    av = np.zeros((len(idx), len(sites), G, G), np.float32)
                    am = np.zeros(len(idx), np.float32)
                    for j, s in enumerate(sites):
                        P, N = geo[s]
                        for ii, i in enumerate(idx):
                            h = np.asarray(dm[s][i], np.float32).reshape(-1)
                            fv[ii, 3*j:3*j+3] = (h[:, None] * N).sum(0)      # net 3-axis force
                            am[ii] += (h > 0).sum()                          # amount proxy
                            bh, bw = 240 // G, 240 // G
                            av[ii, j] = h.reshape(240, 240)[:bh*G, :bw*G].reshape(
                                G, bh, G, bw).mean(axis=(1, 3))
            except Exception as e:
                print(f"     !! {p.name}: {type(e).__name__}")
                continue
            FOR.append(fv); ARR.append(av); AMT.append(am)
            Y.append(np.full(len(idx), 1 if outcome == "failure" else 0, np.int64))
            EID.append(np.full(len(idx), ep)); ep += 1
    return (np.concatenate(FOR), np.concatenate(ARR), np.concatenate(AMT),
            np.concatenate(Y), np.concatenate(EID))


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def ori(a):
    return max(a, 1-a) if np.isfinite(a) else float("nan")


def ep_auc(score, y, eid):
    sc, lb = [], []
    for e in np.unique(eid):
        m = eid == e
        sc.append(score[m].mean()); lb.append(y[m][0])
    return ori(auc(sc, lb))


def main():
    res = {}
    for t in TASKS:
        FOR, ARR, AMT, Y, EID = load(t)
        print(f"   task {t}: {len(Y)} 帧, {len(np.unique(EID))} 集, 失败率 {Y.mean():.2f}")
        # force magnitude as a scalar baseline
        fmag = np.linalg.norm(FOR.reshape(len(FOR), -1, 3), axis=2).sum(1)
        r = {"n_frames": int(len(Y)), "n_eps": int(len(np.unique(EID))),
             "auc_amount": ep_auc(-AMT, Y, EID),
             "auc_forcemag": ep_auc(-fmag, Y, EID)}
        # learned monitors: 5-fold CV over episodes, pooled per-episode score
        import torch, torch.nn as nn, torch.nn.functional as F
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        def run_cv(X, kind):
            eps = np.unique(EID); rng = np.random.RandomState(0); rng.shuffle(eps)
            oof = np.zeros(len(Y))
            for va in np.array_split(eps, 5):
                tr = np.setdiff1d(eps, va)
                mt = np.isin(EID, tr); mv = np.isin(EID, va)
                torch.manual_seed(0)
                if kind == "force":
                    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
                else:
                    net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                                        nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                                        nn.Linear(48, 1)).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
                ytr = Y[mt]; yep = np.array([Y[EID == e][0] for e in tr])
                epof = {e: i for i, e in enumerate(tr)}
                yt = torch.as_tensor([yep[epof[e]] for e in EID[mt]], device=dev).float()
                Xt = torch.as_tensor(X[mt], device=dev).float()
                for _ in range(25):
                    perm = torch.randperm(len(yt), device=dev)
                    for b in range(0, len(yt), 256):
                        s = perm[b:b+256]
                        loss = F.binary_cross_entropy_with_logits(net(Xt[s]).squeeze(-1), yt[s])
                        opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    oof[mv] = torch.sigmoid(net(torch.as_tensor(
                        X[mv], device=dev).float()).squeeze(-1)).cpu().numpy()
            sc, lb = [], []
            for e in np.unique(EID):
                m = EID == e; sc.append(oof[m].mean()); lb.append(Y[m][0])
            return ori(auc(sc, lb))

        r["auc_force_learned"] = run_cv(FOR.reshape(len(FOR), -1), "force")
        r["auc_array_learned"] = run_cv(ARR.transpose(0, 2, 3, 1), "array")
        res[t] = r
        print(f"     量统计量       AUC={r['auc_amount']:.3f}")
        print(f"     力幅值         AUC={r['auc_forcemag']:.3f}")
        print(f"     力(学习,30维)  AUC={r['auc_force_learned']:.3f}")
        print(f"     阵列(学习,20²) AUC={r['auc_array_learned']:.3f}")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/s1_results.json", "w"), indent=1)
    print("\n   saved theory/s1_results.json")


if __name__ == "__main__":
    main()
