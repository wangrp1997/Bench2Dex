"""T11 (1): the matched-magnitude dissociation test.

Two hand configurations were authored by editing robot/qpos of the same kinematic replay, so
the OBJECT POSE IS IDENTICAL and only the hand differs:
    A_noThumb : both thumbs opened, fingers in the demo grasp  -> contact amount present,
                thumb opposition ALWAYS FALSE
    B_opp1    : thumbs kept, all but right index/middle opened -> thumb opposition PRESENT

Because replay is kinematic, this is a controlled probe. We now MATCH THE CONTACT AMOUNT:
pick frames from A and from B with the SAME number of contacting sites, and ask
    (a) the amount statistic           -> cannot separate them BY CONSTRUCTION (~0.5)
    (b) thumb-contact indicator        -> trivially 1.0 (it IS the relation)
    (c) a learned monitor on tactile   -> the real question:
        does the SENSOR carry the relational information that the amount statistic cannot?
If (c) >> 0.5 the relation is monitorable and the scalar is NOT sufficient - the theory's
positive prediction, and the matched-magnitude experiment the literature does not contain.
Frame-level, so n is in the thousands (unlike the 20-episode study).
"""
import h5py, numpy as np, json, os

D = "/home/wangrenpeng/bench2dex/theory/variants_out"
G = 24


def load(name):
    with h5py.File(os.path.join(D, name + ".hdf5"), "r") as f:
        tm = f["robot/tactile/tacmap"]
        sites = list(tm.keys())
        T = tm[sites[0]].shape[0]
        img = np.zeros((T, len(sites), G, G), np.uint8)
        on = np.zeros((T, len(sites)), bool)
        for j, s in enumerate(sites):
            a = np.asarray(tm[s][:], np.uint8)
            bh, bw = a.shape[1] // G, a.shape[2] // G
            img[:, j] = a[:, :bh*G, :bw*G].reshape(T, G, bh, G, bw).max(axis=(2, 4))
            on[:, j] = a.reshape(T, -1).max(1) > 0
    return img, on, sites


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def ori(a):
    return max(a, 1-a) if np.isfinite(a) else float("nan")


def main():
    imgA, onA, sites = load("A_noThumb")
    imgB, onB, _ = load("B_opp1")
    thr = np.array([("thumb" in s) for s in sites])
    amtA, amtB = onA.sum(1), onB.sum(1)
    thB = onB[:, thr].any(1)
    print(f"   sites={len(sites)}  拇指位点={list(np.array(sites)[thr])}")

    res = {}
    for k in (1, 2, 3):
        ia = np.where(amtA == k)[0]
        ib = np.where((amtB == k) & thB)[0]
        if len(ia) < 30 or len(ib) < 30:
            print(f"   量={k}: 帧数不足 (A={len(ia)}, B={len(ib)})")
            continue
        m = min(len(ia), len(ib))
        rng = np.random.RandomState(0)
        ia, ib = rng.choice(ia, m, replace=False), rng.choice(ib, m, replace=False)
        X = np.concatenate([imgA[ia], imgB[ib]]).astype(np.float32) / 255.0
        y = np.concatenate([np.zeros(m), np.ones(m)])
        amt = np.concatenate([amtA[ia], amtB[ib]]).astype(float)
        thumb = np.concatenate([onA[ia][:, thr].any(1), onB[ib][:, thr].any(1)]).astype(float)
        r = {"k": k, "n_per_class": int(m),
             "auc_amount": ori(auc(amt, y)),
             "auc_thumb": ori(auc(thumb, y))}
        # learned monitor on tactile, 5-fold CV over frames
        import torch, torch.nn as nn, torch.nn.functional as F
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
        scores = np.zeros(len(y))
        for fi in range(5):
            va = folds[fi]; tr = np.setdiff1d(idx, va)
            torch.manual_seed(0)
            net = nn.Sequential(
                nn.Conv2d(len(sites), 24, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(), nn.AdaptiveMaxPool2d(1),
                nn.Flatten(), nn.Linear(48, 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
            Xt = torch.as_tensor(X[tr], device=dev); yt = torch.as_tensor(y[tr], device=dev)
            for ep in range(40):
                perm = torch.randperm(len(tr), device=dev)
                for b in range(0, len(tr), 64):
                    s = perm[b:b+64]
                    loss = F.binary_cross_entropy_with_logits(
                        net(Xt[s]).squeeze(-1), yt[s])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                scores[va] = torch.sigmoid(
                    net(torch.as_tensor(X[va], device=dev)).squeeze(-1)).cpu().numpy()
        r["auc_learned"] = ori(auc(scores, y))
        res[k] = r
        print(f"   量={k}  每类 {m} 帧")
        print(f"      量统计量      AUC={r['auc_amount']:.3f}   <- 按构造应≈0.5")
        print(f"      拇指指示      AUC={r['auc_thumb']:.3f}   <- 平凡（就是那个关系）")
        print(f"      学习式监测器  AUC={r['auc_learned']:.3f}   <- ★ 关键：传感器是否携带关系信息")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/t11_results.json", "w"), indent=1)
    print("\n   saved theory/t11_results.json")


if __name__ == "__main__":
    main()
