"""T9 (theory defence): the POINT-MEASUREMENT LIMIT.

The classical contact-observability line (Debus & Dupont ICRA 2004; Jia & Erdmann, Montana)
analyses contact identification from POINT observations -- a single contact, a point wrench.
Our R1 statement is for a DISTRIBUTED surface. For the claim to be a genuine generalisation
rather than a different problem, it must degrade to the classical verdict as the distributed
field collapses to a point reading.

Prediction to verify: with k contacting surface points, the Fisher information for the
contact plane (n in S^2, d) has rank min(3, k). So
    k = 1  ->  rank 1: only the plane OFFSET is identifiable (the point-observation regime)
    k >= 3 ->  rank 3: the full plane is identifiable
i.e. the classical point case is exactly the rank-1 corner of our family.
"""
import numpy as np, json

SENS = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor"
SIGMA = (0.015 / 255.0) / np.sqrt(12.0)


def geom(tag):
    P = np.load(f"{SENS}/tactileSensor_map_{tag}_point.npy").astype(np.float64) * 1e-3
    return P.reshape(-1, 3)


def tbasis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); return t1, np.cross(n, t1)


def plane_fim(X, n):
    """FIM for (n_tangent(2), d) from contacting points X. Model h = x·n - d."""
    t1, t2 = tbasis(n)
    G = np.concatenate([np.stack([X @ t1, X @ t2], axis=1), -np.ones((len(X), 1))], axis=1)
    return (G.T @ G) / SIGMA ** 2


def main():
    P = geom("4F")
    n_true = np.array([0.0, 0.0, 1.0])
    n_true = n_true / np.linalg.norm(n_true)
    # place a planar contact of a fixed depth and find the active patch
    proj = P @ n_true
    p = n_true * (proj.max() - 0.002)
    h = (P - p) @ n_true
    act = np.where(h > 1e-6)[0]
    Xact = P[act]
    print(f"   活跃接触斑: {len(act)} 个采样点")

    rows = []
    for k in (1, 2, 3, 4, 6, 10, 20, 40):
        if k > len(act):
            continue
        rank_c, cond_c, crb_d_c, crb_n_c = [], [], [], []
        for trial in range(8):
            rng = np.random.RandomState(trial)
            if k == len(act):
                idx = act
            else:
                # well-spread subsample: farthest-point-ish via random then keep spread
                idx = act[rng.choice(len(act), k, replace=False)]
            F = plane_fim(P[idx], n_true)
            w = np.linalg.svd(F, compute_uv=False)
            rank_c.append(int((w > 1e-9 * max(w[0], 1e-30)).sum()))
            C = np.linalg.pinv(F)
            cond_c.append(float(w[0] / max(w[-1], 1e-300)))
            crb_d_c.append(float(np.sqrt(max(C[2, 2], 0))))
            crb_n_c.append(float(np.degrees(np.sqrt(max(C[0, 0], 0)))))
        rows.append(dict(k=int(k), rank_med=float(np.median(rank_c)),
                         cond_med=float(np.median(cond_c)),
                         crb_d_med_m=float(np.median(crb_d_c)),
                         crb_n_med_deg=float(np.median(crb_n_c))))
        r = rows[-1]
        print(f"   k={k:3d}  秩中位={r['rank_med']:.1f}  条件数中位={r['cond_med']:.2e}  "
              f"CRB_d={r['crb_d_med_m']*1e6:9.2f}µm  CRB_n={r['crb_n_med_deg']:.4f}°")

    print()
    r1 = next((r for r in rows if r["k"] == 1), None)
    r3 = next((r for r in rows if r["k"] == 3), None)
    if r1 and r3:
        print(f"   预言  秩 = min(3, k):")
        print(f"     k=1 → 秩 {r1['rank_med']:.0f}（只有平面偏置 d 可辨识 = 经典点观测情形）")
        print(f"     k=3 → 秩 {r3['rank_med']:.0f}（整个平面可辨识）")
        ok = (r1["rank_med"] == 1 and r3["rank_med"] == 3)
        print(f"     预言成立: {ok}")
    json.dump(rows, open("/home/wangrenpeng/bench2dex/theory/t9_results.json", "w"), indent=1)
    print("\n   saved theory/t9_results.json")


if __name__ == "__main__":
    main()
