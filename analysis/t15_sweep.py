"""T15 (1): the matched-magnitude SWEEP, with the control that makes it interpretable.

Two complementary tests, both on the controlled kinematic variants:

  TEST 1  matched RELATION, varying AMOUNT
          frames from the same family (thumb contact state held fixed) at different contact
          amounts. The amount statistic SHOULD separate them (~1.0). This is the control
          proving the statistic is not simply broken.

  TEST 2  matched AMOUNT, varying RELATION
          frames from family A (thumbs away) and family B (thumbs kept) with the SAME contact
          amount. The amount statistic CANNOT separate them by construction (~0.5), while a
          learned monitor should, if the sensor carries the relational information.

Together: the statistic tracks the amount perfectly and is blind to the relation.
"""
import h5py, numpy as np, json, os, itertools

D = "/home/wangrenpeng/bench2dex/theory/variants_out"
G = 24


def load(name):
    p = os.path.join(D, name + ".hdf5")
    with h5py.File(p, "r") as f:
        tm = f["robot/tactile/tacmap"]
        sites = list(tm.keys()); T = tm[sites[0]].shape[0]
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


def train_eval(Xtr, ytr, Xva, n_sites, seed=0):
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(n_sites, 24, 3, 2, 1), nn.ReLU(),
                        nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(), nn.AdaptiveMaxPool2d(1),
                        nn.Flatten(), nn.Linear(48, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.as_tensor(Xtr, device=dev); yt = torch.as_tensor(ytr, device=dev)
    for _ in range(30):
        perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), 64):
            s = perm[b:b+64]
            loss = F.binary_cross_entropy_with_logits(net(Xt[s]).squeeze(-1), yt[s])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.as_tensor(Xva, device=dev)).squeeze(-1)).cpu().numpy()


def cv_auc(X, y, n_sites, folds=5, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y)); sc = np.zeros(len(y))
    for f in np.array_split(idx, folds):
        tr = np.setdiff1d(idx, f)
        sc[f] = train_eval(X[tr], y[tr], X[f], n_sites, seed)
    return ori(auc(sc, y))


def main():
    fams = {}
    for fam in ("A", "B"):
        for k in (1, 2, 3, 4):
            n = f"{fam}_k{k}"
            if not os.path.exists(os.path.join(D, n + ".hdf5")):
                continue
            img, on, sites = load(n)
            fams[n] = dict(img=img, amt=on.sum(1),
                           thumb=on[:, np.array([("thumb" in s) for s in sites])].any(1),
                           sites=sites)
    print(f"   载入 {len(fams)} 个变体")
    ns = len(next(iter(fams.values()))["sites"])
    res = {"variants": {}, "test1": [], "test2": []}

    for n, d in fams.items():
        res["variants"][n] = dict(frames=int(len(d["amt"])), amt_mean=float(d["amt"].mean()),
                                  thumb_frac=float(d["thumb"].mean()),
                                  amt_hist={int(k): int((d["amt"] == k).sum())
                                            for k in range(0, 7)})
        print(f"   {n:7s} 量均值={d['amt'].mean():.2f}  拇指占比={d['thumb'].mean():.3f}  "
              f"量直方图={res['variants'][n]['amt_hist']}")

    # ---------------- TEST 1: matched relation, varying amount (within family B) ----------
    print("\n   ===== 检验 1：匹配关系（都有拇指接触），变化接触量 → 量统计量应≈1.0 =====")
    for k1, k2 in itertools.combinations([1, 2, 3, 4], 2):
        a, b = f"B_k{k1}", f"B_k{k2}"
        if a not in fams or b not in fams:
            continue
        da, db = fams[a], fams[b]
        ia = np.where(da["thumb"] & (da["amt"] == da["amt"][da["thumb"]].min()))[0]
        ib = np.where(db["thumb"] & (db["amt"] == db["amt"][db["thumb"]].max()))[0]
        if len(ia) < 25 or len(ib) < 25:
            continue
        m = min(len(ia), len(ib)); rng = np.random.RandomState(0)
        ia, ib = rng.choice(ia, m, replace=False), rng.choice(ib, m, replace=False)
        amt = np.concatenate([da["amt"][ia], db["amt"][ib]]).astype(float)
        X = np.concatenate([da["img"][ia], db["img"][ib]]).astype(np.float32)/255.0
        y = np.concatenate([np.zeros(m), np.ones(m)])
        r = dict(pair=f"{a}({da['amt'][ia].mean():.0f}) vs {b}({db['amt'][ib].mean():.0f})",
                 n_per_class=int(m), auc_amount=ori(auc(amt, y)),
                 auc_learned=cv_auc(X, y, ns))
        res["test1"].append(r)
        print(f"     {r['pair']:26s} n={m:3d}  量统计量 AUC={r['auc_amount']:.3f}  "
              f"学习 AUC={r['auc_learned']:.3f}")

    # ---------------- TEST 2: matched amount, varying relation ----------------------------
    print("\n   ===== 检验 2：匹配接触量，变化关系 → 量统计量应=0.5，学习应≈1.0 =====")
    for k in (1, 2, 3, 4):
        a, b = f"A_k{k}", f"B_k{k}"
        if a not in fams or b not in fams:
            continue
        da, db = fams[a], fams[b]
        for m_amt in (1, 2, 3):
            ia = np.where(da["amt"] == m_amt)[0]
            ib = np.where(db["thumb"] & (db["amt"] == m_amt))[0]
            if len(ia) < 25 or len(ib) < 25:
                continue
            m = min(len(ia), len(ib)); rng = np.random.RandomState(0)
            ia, ib = rng.choice(ia, m, replace=False), rng.choice(ib, m, replace=False)
            amt = np.concatenate([da["amt"][ia], db["amt"][ib]]).astype(float)
            X = np.concatenate([da["img"][ia], db["img"][ib]]).astype(np.float32)/255.0
            y = np.concatenate([np.zeros(m), np.ones(m)])
            r = dict(amount=m_amt, famA=a, famB=b, n_per_class=int(m),
                     auc_amount=ori(auc(amt, y)), auc_learned=cv_auc(X, y, ns))
            res["test2"].append(r)
            print(f"     量={m_amt}  {a} vs {b}  n={m:3d}  量统计量 AUC={r['auc_amount']:.3f}  "
                  f"学习 AUC={r['auc_learned']:.3f}")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/t15_results.json", "w"), indent=1)
    print("\n   saved theory/t15_results.json")


if __name__ == "__main__":
    main()
