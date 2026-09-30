"""T8: instantiate the monitor-sufficiency ceiling.

CORRECTED STATEMENT (a survey caught my original Proposition 1 being too strong):
    sufficiency does NOT imply AUC(f) <= AUC(c) for the RAW statistic c. It implies
        sup_f AUC(f) = AUC( P(Y|c) )
    i.e. the ceiling is the best TRANSFORM of c (its likelihood ratio), and if P(Y|c) is
    non-monotone (U-shaped risk) a learned model may legitimately beat raw AUC(c).

So the testable predictions become
    P1  AUC(P(Y|c))  >=  AUC(raw c)                     (transform can only help)
    P2  AUC(learned on full tactile)  <=  AUC(P(Y|c))   (learning cannot add beyond the
                                                         sufficient statistic) <-- THE new test
    P3  divergence-derived ceiling    >=  AUC(P(Y|c))
Arms, episode level, leave-one-out, per task and pooled (60 episodes):
    raw statistic c        = mean number of contacting sites
    transform P(Y|c)       = LOO logistic on c            (the sufficiency ceiling given c)
    learned (rich)         = LOO logistic on the full per-site contact-frequency vector
                             + the contact-count histogram
    ceiling(MI)            = Gaussian-style Phi(sqrt(2*MI)) using an estimated JSD
"""
import h5py, numpy as np, json
from pathlib import Path

ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
TASKS = ["26", "32", "73"]


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def ori(a):
    return max(a, 1.0 - a) if np.isfinite(a) else float("nan")


def feats(task):
    C, RICH, Y = [], [], []
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task / outcome).glob("*.hdf5")):
            with h5py.File(p, "r") as f:
                tm = f["robot/tactile"]["tacmap"]
                sites = list(tm.keys()); n = int(f["meta/frame_count"][()])
                on = np.zeros((n, len(sites)), bool)
                for j, s in enumerate(sites):
                    a = np.asarray(tm[s][:], np.uint8)
                    on[:, j] = a.reshape(n, -1).max(1) > 0
            cnt = on.sum(1)
            C.append([cnt.mean()])
            RICH.append(np.concatenate([on.mean(0),
                                        np.histogram(cnt, bins=np.arange(0, 9))[0] / max(n, 1)]))
            Y.append(1.0 if outcome == "failure" else 0.0)
    return np.array(C), np.array(RICH), np.array(Y)


def loo_logreg(X, y, l2=1.0):
    out = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xt = (X[tr] - mu) / sd; Xv = (X[[i]] - mu) / sd
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(300):
            pr = 1/(1+np.exp(-(Xt @ w + b)))
            g = Xt.T @ (pr - y[tr])/len(tr) + l2*w/len(tr)
            H = Xt.T @ ((pr*(1-pr))[:, None]*Xt)/len(tr) + l2*np.eye(X.shape[1])/len(tr)
            w -= np.linalg.solve(H + 1e-9*np.eye(len(w)), g)
            b -= (pr - y[tr]).mean()
        out[i] = float(Xv @ w + b)
    return out


def jsd_mi(x, y):
    """Balanced-prior MI = JSD(P_+ || P_-) in nats, estimated with 1-D histograms."""
    a, b = x[y > 0], x[y <= 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    lo, hi = np.percentile(x, [1, 99])
    bins = np.linspace(lo, hi + 1e-9, 12)
    pa, _ = np.histogram(a, bins, density=False); pb, _ = np.histogram(b, bins, density=False)
    pa = pa / max(pa.sum(), 1); pb = pb / max(pb.sum(), 1)
    m = 0.5 * (pa + pb)
    def kl(p, q):
        msk = (p > 0) & (q > 0)
        return float(np.sum(p[msk] * np.log(p[msk] / q[msk])))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def main():
    res = {}
    allC, allR, allY = [], [], []
    for t in TASKS:
        C, RICH, Y = feats(t)
        allC.append(C); allR.append(RICH); allY.append(Y)
        r = {}
        r["auc_raw_c"] = ori(auc(C[:, 0], Y))
        r["auc_transform_c"] = ori(auc(loo_logreg(C, Y), Y))
        r["auc_learned_rich"] = ori(auc(loo_logreg(RICH, Y), Y))
        mi = jsd_mi(C[:, 0], Y)
        r["mi_nats"] = mi
        # Gaussian-style illustration Phi(sqrt(2 MI)); flagged as illustrative, not the tight bound
        from math import erf, sqrt
        r["ceiling_phi_sqrt2mi"] = 0.5 * (1 + erf(sqrt(max(2*mi, 0)) / sqrt(2))) if np.isfinite(mi) else float("nan")
        r["P1_transform_ge_raw"] = bool(r["auc_transform_c"] >= r["auc_raw_c"] - 1e-9)
        r["P2_learned_le_transform"] = bool(r["auc_learned_rich"] <= r["auc_transform_c"] + 1e-9)
        res[t] = r
        print(f"   task {t} (n=20, fail={Y.mean():.2f})")
        print(f"     raw c          AUC={r['auc_raw_c']:.3f}")
        print(f"     transform P(Y|c) AUC={r['auc_transform_c']:.3f}   <- 充分性上界（给定 c）")
        print(f"     learned(rich)  AUC={r['auc_learned_rich']:.3f}")
        print(f"     MI={mi:.4f} nats  ->  示例上界 Φ(√(2MI))={r['ceiling_phi_sqrt2mi']:.3f}")
        print(f"     P1 变换≥原始: {r['P1_transform_ge_raw']}   P2 学习≤变换: {r['P2_learned_le_transform']}")
    # pooled
    C = np.concatenate(allC); R = np.concatenate(allR); Y = np.concatenate(allY)
    pooled = {"n": int(len(Y)), "fail_rate": float(Y.mean()),
              "auc_raw_c": ori(auc(C[:, 0], Y)),
              "auc_transform_c": ori(auc(loo_logreg(C, Y), Y)),
              "auc_learned_rich": ori(auc(loo_logreg(R, Y), Y))}
    pooled["P2_learned_le_transform"] = bool(pooled["auc_learned_rich"] <= pooled["auc_transform_c"] + 1e-9)
    res["pooled"] = pooled
    print(f"\n   pooled (n=60)")
    print(f"     raw c {pooled['auc_raw_c']:.3f} | transform {pooled['auc_transform_c']:.3f} | "
          f"learned(rich) {pooled['auc_learned_rich']:.3f}")
    print(f"     P2 学习≤变换: {pooled['P2_learned_le_transform']}")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/t8_results.json", "w"), indent=1)
    print("\n   saved theory/t8_results.json")


if __name__ == "__main__":
    main()
