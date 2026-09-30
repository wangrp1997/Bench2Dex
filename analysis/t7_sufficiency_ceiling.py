"""T7: is the scalar contact amount (approximately) SUFFICIENT for the task outcome?

The theory's sufficiency leg predicts: if the outcome depends on the contact state only
through a low-dimensional function, then no richer function of the tactile data can help.
That is testable without new data by asking whether a RICHER description of the same
tactile stream beats the scalar.

Episode-level, leave-one-out, 60 rollouts over 3 tasks:
  f1  scalar   : mean number of contacting sites over the episode
  f2  histogram: the FULL distribution of the contact-count over frames (8 bins)
  f3  co-occurrence: per-site contact frequencies (10 numbers) -- which sites, not how many
A logistic model on f2/f3 beating f1 would mean the scalar is NOT sufficient.
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
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def feats(task):
    F1, F2, F3, Y = [], [], [], []
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task / outcome).glob("*.hdf5")):
            with h5py.File(p, "r") as f:
                tm = f["robot/tactile"]["tacmap"]
                sites = list(tm.keys())
                n = int(f["meta/frame_count"][()])
                site_on = np.zeros((n, len(sites)), bool)
                for j, s in enumerate(sites):
                    a = np.asarray(tm[s][:], np.uint8)
                    site_on[:, j] = a.reshape(n, -1).max(1) > 0
            cnt = site_on.sum(1)
            F1.append([cnt.mean()])
            F2.append(np.histogram(cnt, bins=np.arange(0, 9), density=False)[0] / max(n, 1))
            F3.append(site_on.mean(0))
            Y.append(1.0 if outcome == "failure" else 0.0)
    return np.array(F1), np.array(F2), np.array(F3), np.array(Y)


def loo_logreg(X, y, l2=1.0):
    """leave-one-out logistic regression -> out-of-fold score per episode."""
    out = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xt = (X[tr] - mu) / sd; Xv = (X[[i]] - mu) / sd
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(300):
            z = Xt @ w + b
            pr = 1 / (1 + np.exp(-z))
            g = Xt.T @ (pr - y[tr]) / len(tr) + l2 * w / len(tr)
            h = (pr * (1 - pr))[:, None] * Xt
            H = Xt.T @ h / len(tr) + l2 * np.eye(X.shape[1]) / len(tr)
            step = np.linalg.solve(H + 1e-9*np.eye(len(w)), g)
            w -= step; b -= (pr - y[tr]).mean()
        out[i] = float(Xv @ w + b)
    return out


def ori(a):
    """Oriented AUC: the SIGN of a ranking is one bit that any model learns for free, so
    comparing an un-oriented raw statistic (AUC 0.036) against a logistic model (which picks
    its own sign) is not a fair comparison. Take max(a, 1-a) for every feature."""
    return max(a, 1.0 - a) if np.isfinite(a) else float("nan")


def main():
    res = {}
    for t in TASKS:
        f1, f2, f3, y = feats(t)
        r = {"n_eps": int(len(y)), "fail_rate": float(y.mean())}
        r["f1_scalar_raw"] = auc(f1[:, 0], y)
        r["f1_scalar_auc"] = ori(auc(f1[:, 0], y))
        r["f2_hist_auc"] = ori(auc(loo_logreg(f2, y), y))
        r["f3_sites_auc"] = ori(auc(loo_logreg(f3, y), y))
        r["f2_minus_f1"] = r["f2_hist_auc"] - r["f1_scalar_auc"]
        r["f3_minus_f1"] = r["f3_sites_auc"] - r["f1_scalar_auc"]
        res[t] = r
        print(f"   task {t}: n={r['n_eps']} fail={r['fail_rate']:.2f}")
        print(f"     f1 标量(平均接触位点数)      AUC={r['f1_scalar_auc']:.3f}")
        print(f"     f2 完整计数直方图(丰富)      AUC={r['f2_hist_auc']:.3f}   Δ={r['f2_minus_f1']:+.3f}")
        print(f"     f3 逐位点接触频率(哪些位点)  AUC={r['f3_sites_auc']:.3f}   Δ={r['f3_minus_f1']:+.3f}")
    print()
    d2 = np.mean([res[t]["f2_minus_f1"] for t in TASKS])
    d3 = np.mean([res[t]["f3_minus_f1"] for t in TASKS])
    print(f"   平均 Δ: 直方图 {d2:+.3f}   逐位点 {d3:+.3f}")
    print(f"   → 若 Δ ≈ 0 或负，则标量(近似)充分，更丰富的触觉描述【无法】超越它")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/t7_results.json", "w"), indent=1)
    print("\n   saved theory/t7_results.json")


if __name__ == "__main__":
    main()
