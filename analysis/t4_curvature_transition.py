"""T4: identifiability transition with object curvature.

T1/T2/T3 established the PLANAR case: a half-space contact is invariant to in-plane
translation, so only the contact plane (3 DOF) is observable and the tangential contact
position is gauge. That is the R -> infinity limit of a curved object.

Here the object is a SPHERE of radius R touching the sensor surface:
        h_j = max(0, R - |x_j - c|),      parameter c in R^3 (sphere centre)
A lateral displacement of the centre changes the depth profile by roughly
        delta_h ~ delta_t * r_t / R
so the tangential sensitivity decays like 1/R: curvature is what makes lateral contact
pose identifiable at all.

We sweep R for every shipped sensor surface and report the CRB on the TANGENTIAL centre
coordinate, normalised by the surface extent. The interesting number is the ratio R/extent
at which tangential identifiability is lost.
"""
import numpy as np, os, json, glob

ROOT = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p"
SIGMA = (0.015 / 255.0) / np.sqrt(12.0)      # 8-bit quantisation noise, metres
DEPTH = 0.002                                 # representative contact depth (2 mm)
RS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 1e0]   # sphere radii to sweep


def surfaces():
    for hand in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, hand, "tactile_sensor")
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(d + "/*_point.npy")):
            tag = os.path.basename(p).replace("tactileSensor_map_", "").replace("_point.npy", "")
            P = np.load(p).astype(np.float64) * 1e-3      # mm -> m
            yield hand, tag, P.reshape(-1, 3)


def contact_and_fim(P, n_ext, R, depth=DEPTH):
    """Place a sphere of radius R penetrating the surface by `depth` along n_ext.

    Computed in the LOCAL frame at the contact point: for large R the sphere centre is far
    away and a global-frame computation suffers catastrophic cancellation (the first
    version raised LinAlgError). Rank and CRB come from an SVD of the Jacobian, which is
    numerically far more robust than eigendecomposition of an ill-conditioned FIM.
    """
    n = np.asarray(n_ext, float); n /= np.linalg.norm(n)
    t1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    p0 = P[np.argmax(P @ n)]
    X = P - p0                                   # local coordinates
    a = X @ t1; b = X @ t2; z = X @ n            # (M,) tangential x2, normal
    # |x - c| with c = (R - depth) n  ->  sqrt(a^2 + b^2 + (z - (R - depth))^2)
    dz = z - (R - depth)
    dist = np.sqrt(a ** 2 + b ** 2 + dz ** 2)
    h = R - dist
    act = h > 1e-6
    if act.sum() < 5:
        return None
    # dh/dc = -(x - c)/|x - c|, expressed on (t1, t2, n)
    aa, bb, dd = a[act], b[act], dz[act]
    nn = np.maximum(dist[act], 1e-12)
    G = np.stack([-aa / nn, -bb / nn, -dd / nn], axis=1)      # (M,3) in the local basis
    return G, int(act.sum()), n, (t1, t2)


def main():
    out = []
    for hand, tag, P in surfaces():
        ext = float(np.linalg.norm(P.max(0) - P.min(0)))     # surface extent (m)
        rows = []
        for n_ext in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            for R in RS:
                r = contact_and_fim(P, n_ext, R)
                if r is None:
                    continue
                G, npts, n, (t1, t2) = r
                # G's columns are in the LOCAL basis (t1, t2, n). svd -> rank & CRB
                sv = np.linalg.svd(G, compute_uv=False)
                rank = int((sv > 1e-9 * max(sv[0], 1e-30)).sum())
                # CRB on the tangential coordinate t1: from the pseudo-inverse of F=G^T G/s^2
                F = (G.T @ G) / SIGMA ** 2
                C = np.linalg.pinv(F)
                # first local axis IS t1, so the tangential variance is C[0,0]
                var_t = float(C[0, 0])
                rows.append(dict(n=[float(x) for x in n_ext], R=R, n_pts=int(npts),
                                 rank=rank, crb_tang_m=float(np.sqrt(max(var_t, 0))),
                                 R_over_extent=float(R / ext)))
        if rows:
            out.append(dict(hand=hand, tag=tag, extent_m=ext, rows=rows))
    json.dump(out, open("/home/wangrenpeng/bench2dex/theory/t4_results.json", "w"), indent=1)

    print(f"   {len(out)} 个传感面")
    # aggregate: tangential CRB vs R/extent (median across surfaces & normals)
    print(f"\n   {'R/extent':>10s} {'中位切向CRB':>14s} {'中位秩':>8s}")
    for R in RS:
        vals = [r["crb_tang_m"] for s in out for r in s["rows"] if r["R"] == R]
        rks = [r["rank"] for s in out for r in s["rows"] if r["R"] == R]
        if not vals:
            continue
        # R/extent varies per surface; report the median ratio for that R
        ratios = [r["R_over_extent"] for s in out for r in s["rows"] if r["R"] == R]
        print(f"   {np.median(ratios):10.3e} {np.median(vals)*1e6:13.1f}um {np.median(rks):8.1f}")
    # per-surface detail for sharpa
    for s in out:
        if s["hand"] != "kuka+sharpa":
            continue
        print(f"\n   --- {s['hand']} / {s['tag']}  (extent {s['extent_m']*1e3:.1f} mm)")
        print(f"      {'R(mm)':>7s} {'R/extent':>10s} {'接触点':>7s} {'秩':>4s} {'切向CRB':>12s}")
        for r in [x for x in s["rows"] if x["n"] == [0, 0, 1]]:
            print(f"      {r['R']*1e3:7.1f} {r['R_over_extent']:10.3f} {r['n_pts']:7d} "
                  f"{r['rank']:4d} {r['crb_tang_m']*1e6:11.1f}um")


if __name__ == "__main__":
    main()
