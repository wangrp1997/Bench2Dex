"""S2 (decisive): the MATCHED-FORCE dissociation in simulation.

S1 showed array >= force in 3/3 tasks but the gaps were inside the noise band. The decisive
test is a CONSTRUCTED one, exactly parallel to T11 but with the force modality:

  - take the controlled variants A_kk (thumbs away) and B_kk (thumbs kept)
  - compute each site's net 3-axis force  F = k * sum_j h_j n_j  (the physical quantity a
    real F/T sensor measures)
  - SELECT frames from A and B whose net force MATCHES within a tolerance
  - then ask: can the force modality tell A from B?   (it should not: matched by construction)
              can the array modality tell A from B?   (it should: the distribution differs)

Because the force is matched by construction, a ~0.5 result CANNOT be dismissed as noise or
small-sample, which is what made S1 and T7/T8 inconclusive.
"""
import h5py, numpy as np, os, json

D = "/home/wangrenpeng/bench2dex/theory/variants_out"
SENS = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor"
G = 20


def geo():
    P4 = np.load(f"{SENS}/tactileSensor_map_4F_point.npy").astype(np.float64)
    N4 = np.load(f"{SENS}/tactileSensor_map_4F_normal.npy").astype(np.float64)
    PT = np.load(f"{SENS}/tactileSensor_map_TH_point.npy").astype(np.float64)
    NT = np.load(f"{SENS}/tactileSensor_map_TH_normal.npy").astype(np.float64)
    out = {}
    for s in ("left_index_elastomer", "left_middle_elastomer", "left_ring_elastomer",
              "left_pinky_elastomer", "right_index_elastomer", "right_middle_elastomer",
              "right_ring_elastomer", "right_pinky_elastomer"):
        out[s] = N4.reshape(-1, 3)
    for s in ("left_thumb_elastomer", "right_thumb_elastomer"):
        out[s] = NT.reshape(-1, 3)
    return out


def load(name, N):
    with h5py.File(os.path.join(D, name + ".hdf5"), "r") as f:
        dm = f["robot/tactile/distance_along_normal_m"]
        sites = list(dm.keys()); T = dm[sites[0]].shape[0]
        FV = np.zeros((T, len(sites) * 3), np.float32)
        FM = np.zeros(T, np.float32)
        AR = np.zeros((T, len(sites), G, G), np.float32)
        for j, s in enumerate(sites):
            for i in range(T):
                h = np.asarray(dm[s][i], np.float32).reshape(-1)
                fv = (h[:, None] * N[s]).sum(0)                 # net 3-axis force, sensor frame
                FV[i, 3*j:3*j+3] = fv
                FM[i] += float(np.linalg.norm(fv))
                bh, bw = 240 // G, 240 // G
                # FIX 2: MAX-pool, not mean-pool. Mean-pooling a 240x240 depth field down to
                # 20x20 averages the small contact patches away (an earlier run got array AUC
                # ~0.50 for exactly this reason); max-pooling keeps the peak depth.
                AR[i, j] = h.reshape(240, 240)[:bh*G, :bw*G].reshape(G, bh, G, bw).max(axis=(1, 3))
    return FV, FM, AR


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def ori(a):
    return max(a, 1-a) if np.isfinite(a) else float("nan")


def cv_auc(X, y, kind, seed=0):
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); sc = np.zeros(len(y))
    for va in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, va)
        torch.manual_seed(0)
        if kind == "force":
            net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
        else:
            net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                                nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(48, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        Xt = torch.as_tensor(X[tr], device=dev).float(); yt = torch.as_tensor(y[tr], device=dev).float()
        for _ in range(30):
            perm = torch.randperm(len(tr), device=dev)
            for b in range(0, len(tr), 64):
                s = perm[b:b+64]
                loss = F.binary_cross_entropy_with_logits(net(Xt[s]).squeeze(-1), yt[s])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sc[va] = torch.sigmoid(net(torch.as_tensor(X[va], device=dev).float()
                                       ).squeeze(-1)).cpu().numpy()
    return ori(auc(sc, y))


def main():
    N = geo()
    res = {}
    for k in (1, 2, 3):
        fa, fb = f"A_k{k}", f"B_k{k}"
        if not os.path.exists(os.path.join(D, fa + ".hdf5")):
            continue
        FVa, FMa, ARa = load(fa, N)
        FVb, FMb, ARb = load(fb, N)
        print(f"\n   {fa}: 净力 中位={np.median(FMa):.4f} 范围[{FMa.min():.4f},{FMa.max():.4f}]")
        print(f"   {fb}: 净力 中位={np.median(FMb):.4f} 范围[{FMb.min():.4f},{FMb.max():.4f}]")
        r = {"pair": f"{fa} vs {fb}"}
        for tol in (0.15, 0.30, 0.50):
            # FIX 1: the "force modality" is the full per-site 3-vector (one F/T per fingertip),
            # so we must match the WHOLE VECTOR, not just its magnitude. Matching only the
            # magnitude left the distribution inside the force features, which is why an earlier
            # run had the learned force model separating the classes at 0.88-0.98.
            scale = np.median(np.linalg.norm(FVa, axis=1)) + 1e-12
            used = set(); ia, ib = [], []
            for i in np.where(np.linalg.norm(FVa, axis=1) > 1e-9)[0]:
                d = np.linalg.norm(FVb - FVa[i], axis=1) / scale
                order = np.argsort(d)
                for j in order:
                    if j in used:
                        continue
                    if d[j] <= tol:
                        used.add(int(j)); ia.append(i); ib.append(int(j))
                    break
            if len(ia) < 25:
                print(f"     容差 {tol:.0%}: 匹配帧数不足 ({len(ia)})")
                continue
            ia, ib = np.array(ia), np.array(ib)
            m = min(len(ia), len(ib)); ia, ib = ia[:m], ib[:m]
            fv = np.concatenate([FVa[ia], FVb[ib]])
            ar = np.concatenate([ARa[ia], ARb[ib]]).transpose(0, 2, 3, 1)
            y = np.concatenate([np.zeros(m), np.ones(m)])
            fmatch = np.concatenate([FMa[ia], FMb[ib]])
            vec_res = np.linalg.norm(FVa[ia] - FVb[ib], axis=1) / scale
            rr = {"tol": tol, "n_per_class": int(m),
                  "force_vec_residual_median": float(np.median(vec_res)),
                  "auc_forcemag": ori(auc(fmatch, y)),
                  "auc_force_learned": cv_auc(fv, y, "force"),
                  "auc_array_learned": cv_auc(ar, y, "array")}
            r[f"tol{tol}"] = rr
            print(f"     容差 {tol:.0%}  n={m:4d}  力向量残差中位={rr['force_vec_residual_median']:.1%}")
            print(f"        力幅值  AUC={rr['auc_forcemag']:.3f}   "
                  f"力30维(学习) AUC={rr['auc_force_learned']:.3f}   "
                  f"阵列(学习) AUC={rr['auc_array_learned']:.3f}")
        res[fa] = r
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/s2_results.json", "w"), indent=1)
    print("\n   saved theory/s2_results.json")


if __name__ == "__main__":
    main()
