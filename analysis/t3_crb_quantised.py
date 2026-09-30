"""T3: statistical CRB test on the ACTUAL sensor output (8-bit quantised tacmap).

T2 compared fits on subsets of the same data, which measures sensitivity to the point set
rather than the noise-driven variance the CRB bounds. Here the estimator sees only what the
sensor really emits -- the 8-bit tacmap -- while the reference is the full-precision
penetration field. The measured error is then compared with the CRB under quantisation
noise sigma = (max_distance/255)/sqrt(12).
"""
import numpy as np, h5py, glob, os, json

SENS = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor"
MAXD = 0.015
SIGMA = (MAXD / 255.0) / np.sqrt(12.0)


def geom(tag):
    P = np.load(f"{SENS}/tactileSensor_map_{tag}_point.npy").astype(np.float64) * 1e-3  # mm->m
    return P.reshape(-1, 3)


def tbasis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    return t1, t2


def fit(X, h, act):
    """Correct plane fit: the hyperplane h = x·n - d lives in R^4, so its normal is the
    smallest singular vector of the CENTRED data [X, h]. Unconstrained least squares plus
    post-hoc normalisation (an earlier version) does NOT solve this problem, because
    rescaling (n,d) -> (n/s, d/s) changes the predicted values."""
    D = np.concatenate([X[act], h[act][:, None]], axis=1)
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


def crb_angle_d(n, X, act):
    t1, t2 = tbasis(n)
    G = np.concatenate([np.stack([X[act] @ t1, X[act] @ t2], 1),
                        -np.ones((act.sum(), 1))], 1)
    C = np.linalg.pinv((G.T @ G) / SIGMA ** 2)
    return np.degrees(np.sqrt(max(C[0, 0], 0))), np.sqrt(max(C[2, 2], 0))


def main():
    eps = sorted(glob.glob("/mnt/public/datasets/bench2dex/teleopdata/dataset/"
                           "26_canned_food_tray_line_arrangement/replay-generalization/*.hdf5"))[:3]
    rows = []
    for ep in eps:
        with h5py.File(ep, "r") as f:
            tm = f["robot/tactile/tacmap"]; dm = f["robot/tactile/distance_along_normal_m"]
            for site in tm.keys():
                tag = "TH" if "thumb" in site else "4F"
                X = geom(tag)
                T = tm[site].shape[0]
                for t in range(0, T, 20):
                    H = np.asarray(dm[site][t], np.float64).reshape(-1)
                    U = np.asarray(tm[site][t], np.uint8).reshape(-1)
                    if H.max() < 1e-4:
                        continue
                    act = H > 1e-5
                    if act.sum() < 50:
                        continue
                    n_ref, d_ref = fit(X, H, act)
                    if n_ref is None:
                        continue
                    # NOTE: the stored `tacmap` is NOT a quantisation of
                    # `distance_along_normal_m` (measured: slope 0.5-10, offset up to 7.4 mm,
                    # residual RMS up to 2.2 mm, correlation as low as 0.35). So we must NOT
                    # use one channel as the reference for the other. Instead we quantise the
                    # depth field OURSELVES to emulate the sensor's 8-bit output, which makes
                    # the test self-consistent.
                    Hq = np.round(H / MAXD * 255.0) / 255.0 * MAXD
                    n_q, d_q = fit(X, Hq, act)
                    if n_q is None:
                        continue
                    ang = np.degrees(np.arccos(np.clip(abs(n_q @ n_ref), -1, 1)))
                    da = abs(d_q - d_ref)
                    ca, cd = crb_angle_d(n_ref, X, act)
                    rows.append(dict(ep=os.path.basename(ep), site=site, t=int(t),
                                     n_pts=int(act.sum()), depth=float(H.max()),
                                     ang_err_deg=float(ang), crb_ang_deg=float(ca),
                                     d_err_m=float(da), crb_d_m=float(cd),
                                     quant_step_m=MAXD / 255.0))
    json.dump(rows, open("/home/wangrenpeng/bench2dex/theory/t3_results.json", "w"), indent=1)
    if not rows:
        print("   no contact frames found"); return
    a = np.array([r["ang_err_deg"] for r in rows]); ca = np.array([r["crb_ang_deg"] for r in rows])
    dd = np.array([r["d_err_m"] for r in rows]); cd = np.array([r["crb_d_m"] for r in rows])
    print(f"   {len(rows)} 个接触帧（3 集演示）")
    print(f"   量化步长 = {MAXD/255*1e6:.1f} µm ; 量化噪声 σ = {SIGMA*1e6:.2f} µm")
    print(f"\n   {'量':>10s} {'实测中位':>12s} {'CRB中位':>12s} {'实测/CRB':>10s}")
    print(f"   {'角度(°)':>10s} {np.median(a):12.4f} {np.median(ca):12.4f} {np.median(a)/max(np.median(ca),1e-9):10.2f}")
    print(f"   {'d(µm)':>10s} {np.median(dd)*1e6:12.2f} {np.median(cd)*1e6:12.2f} "
          f"{np.median(dd)/max(np.median(cd),1e-12):10.2f}")
    print(f"\n   秩相关(实测 vs CRB): 角度 {np.corrcoef(a,ca)[0,1]:+.3f}   d {np.corrcoef(dd,cd)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
