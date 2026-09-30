"""T2 (go/no-go): does the planar-contact model hold, and is the CRB tight?

Ground truth is available because the dataset stores, per tactile site,
  distance_along_normal_m : h_j = (x_j - p)·n   (metres, per surface point x_j)
i.e. exactly the measurement model of T1. So the contact plane (n, d=p·n) can be fit by
least squares, the planarity of real contact can be measured from the residuals, and the
achievable accuracy can be compared with the Cramer-Rao bound

     F = (1/sigma^2) sum_{h_j>0} g_j g_j^T ,  g_j = [ x_j on T_n , -1 ]  in R^3

under the quantisation noise of the 8-bit tacmap (sigma = (h_max/255)/sqrt(12)).
"""
import numpy as np, h5py, os, sys, json, glob

HAND = "kuka+sharpa"
SENS = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p/%s/tactile_sensor" % HAND
MAXD = 0.015                      # robot/tactile/meta/max_distance_m
SIGMA = (MAXD / 255.0) / np.sqrt(12.0)   # 8-bit quantisation noise (metres)


def load_geom(tag):
    # NOTE: the shipped surface geometry is in MILLIMETRES while the depth field
    # (distance_along_normal_m) is in METRES. Mixing them inflates the plane residual by
    # ~1e3 (measured 578x), which looked like "the planar model fails" but was a unit bug.
    P = np.load(f"{SENS}/tactileSensor_map_{tag}_point.npy").astype(np.float64) * 1e-3
    N = np.load(f"{SENS}/tactileSensor_map_{tag}_normal.npy").astype(np.float64)
    return P, N


def tangent_basis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def fit_plane(X, h, act=None):
    """Correct plane fit: the hyperplane h = x·n - d lives in R^4, so its normal is the
    smallest singular vector of the CENTRED data [X, h]. Unconstrained least squares plus
    post-hoc normalisation (an earlier version) does NOT solve this problem, because
    rescaling (n,d) -> (n/s, d/s) changes the predicted values."""
    D = np.concatenate([X, h[:, None]], axis=1)
    Dc = D - D.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    v = Vt[-1]
    if abs(v[3]) < 1e-12:
        return None, None
    n = -v[:3] / v[3]
    d = float(v[:3] @ D.mean(0)[:3] / v[3] + D.mean(0)[3])
    s = np.linalg.norm(n)
    if s < 1e-12:
        return None, None
    return n / s, d / s


def crb(n, X):
    """3x3 CRB for (n_tangent(2), d) given contacting points X (already filtered)."""
    t1, t2 = tangent_basis(n)
    G = np.concatenate([np.stack([X @ t1, X @ t2], axis=1), -np.ones((len(X), 1))], axis=1)
    F = (G.T @ G) / (SIGMA ** 2)
    return np.linalg.pinv(F)


def main():
    ep = sorted(glob.glob("/mnt/public/datasets/bench2dex/teleopdata/dataset/"
                          "26_canned_food_tray_line_arrangement/replay-generalization/*.hdf5"))[0]
    out = {"episode": os.path.basename(ep), "sigma_m": SIGMA, "sites": {}}
    with h5py.File(ep, "r") as f:
        tm = f["robot/tactile/tacmap"]; dm = f["robot/tactile/distance_along_normal_m"]
        for site in list(tm.keys()):
            tag = "TH" if "thumb" in site else "4F"
            P, _ = load_geom(tag)
            D = np.asarray(dm[site][:], np.float32)          # (T,240,240) metres
            T = D.shape[0]
            # frames with real contact
            frames = [t for t in range(0, T, 25) if D[t].max() > 1e-4]
            if not frames:
                continue
            pl, crbs, subs = [], [], []
            for t in frames:
                h = D[t].reshape(-1).astype(np.float64)
                X = P.reshape(-1, 3)
                act = h > 1e-5
                if act.sum() < 5:
                    continue
                n, d = fit_plane(X[act], h[act])
                res = X[act] @ n - d - h[act] if n is not None else None
                if n is None:
                    continue
                pl.append(dict(t=int(t), n_pts=int(act.sum()),
                               resid_rms_m=float(np.sqrt((res ** 2).mean())),
                               depth_max_m=float(h.max()),
                               n=[float(x) for x in n], d=float(d)))
                # CRB on the full contacting set, per-axis sigma
                C = crb(n, X[act])
                crbs.append(np.sqrt(np.clip(np.diag(C), 0, None)).tolist())
                # subsample to test whether the CRB predicts the loss of accuracy
                ss = []
                for frac in (1.0, 0.5, 0.25):
                    k = max(5, int(frac * act.sum()))
                    idx = np.random.RandomState(0).choice(np.where(act)[0], k, replace=False)
                    n2, d2 = fit_plane(X[idx], h[idx])
                    if n2 is None:
                        continue
                    ang = np.degrees(np.arccos(np.clip(abs(n2 @ n), -1, 1)))
                    C2 = crb(n, X[idx])
                    ss.append(dict(frac=frac, k=int(k), angle_err_deg=float(ang),
                                   d_err_m=float(abs(d2 - d)),
                                   crb_n_deg=float(np.degrees(np.sqrt(np.clip(C2[0, 0], 0, None)))),
                                   crb_d_m=float(np.sqrt(np.clip(C2[2, 2], 0, None)))))
                subs.append(ss)
            if not pl:
                continue
            res_rms = np.array([p["resid_rms_m"] for p in pl])
            dep = np.array([p["depth_max_m"] for p in pl])
            out["sites"][site] = dict(
                n_frames=len(pl),
                depth_max_median=float(np.median(dep)),
                resid_rms_median=float(np.median(res_rms)),
                resid_over_depth=float(np.median(res_rms / np.maximum(dep, 1e-9))),
                crb_axis_median=[float(np.median([c[i] for c in crbs])) for i in range(3)],
                subsample=subs[:1])
            print(f"   {site:26s} 帧={len(pl):3d} 深度中位={np.median(dep)*1e3:.2f}mm "
                  f"平面残差RMS中位={np.median(res_rms)*1e6:.1f}um "
                  f"残差/深度={np.median(res_rms/np.maximum(dep,1e-9)):.3f}")
    json.dump(out, open("/home/wangrenpeng/bench2dex/theory/t2_results.json", "w"), indent=1)
    print(f"\n   已保存 theory/t2_results.json")
    # subsample summary
    print(f"\n   子采样检验（CRB 是否预测精度损失）:")
    for site, s in out["sites"].items():
        for row in s["subsample"]:
            for r in row:
                print(f"     {site[:22]:24s} frac={r['frac']:.2f} k={r['k']:5d} "
                      f"角度误差={r['angle_err_deg']:7.3f}° CRB角度={r['crb_n_deg']:7.3f}°   "
                      f"d误差={r['d_err_m']*1e6:8.1f}um CRB_d={r['crb_d_m']*1e6:8.1f}um")


if __name__ == "__main__":
    main()
