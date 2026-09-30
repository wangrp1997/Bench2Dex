"""T1: identifiability of contact state from a DISTRIBUTED tactile output map.

Measurement model (rigid planar contact / half-space)
-----------------------------------------------------
A rigid object occupies the half-space {x : (x - p)·n <= 0}. A tactile point x_j on the
sensor surface with outward normal m_j measures the penetration depth

        h_j(theta) = max(0, -(x_j - p)·n),          theta = (p, n)

so the tactile output map is a 57,600-dimensional vector field per surface, driven by a
5-DOF contact state (p in R^3, n in S^2).

Question: which components of theta are identifiable, and what is the Cramer-Rao bound?
Everything is computed on the REAL sensor geometry shipped with the benchmark
(Robots_p/<hand>/tactile_sensor/*_point.npy and *_normal.npy).
"""
import numpy as np, glob, os, sys, json

ROOT = "/mnt/public/datasets/bench2dex/dex2bench_dataset/Robots_p"


def load_surface(hand, tag):
    d = os.path.join(ROOT, hand, "tactile_sensor")
    # geometry is shipped in MILLIMETRES; convert to metres so the Fisher information
    # (and hence its condition number) is in physical units. Rank is scale-invariant.
    P = np.load(os.path.join(d, f"tactileSensor_map_{tag}_point.npy")).reshape(-1, 3) * 1e-3
    N = np.load(os.path.join(d, f"tactileSensor_map_{tag}_normal.npy")).reshape(-1, 3)
    return P.astype(np.float64), N.astype(np.float64)


def tangent_basis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def fisher(P, n_contact, depth, sigma=1.0):
    """Fisher information for theta=(p(3), n_tangent(2)) given a contact.

    h_j = -(x_j - p)·n for points inside the object. Let the contact plane be placed so the
    deepest penetration is `depth`. Returns the 5x5 FIM and the contacting point set.
    """
    n = np.asarray(n_contact, float); n /= np.linalg.norm(n)
    t1, t2 = tangent_basis(n)
    # Object lies in {(x - p)·n >= 0}; penetration of surface point x_j is (x_j - p)·n.
    # Place the plane so the deepest penetration over the surface equals `depth`.
    proj = P @ n
    p = n * (proj.max() - depth)          # offset giving max penetration == depth
    h = (P - p) @ n                       # > 0 only for the contacting cap
    act = h > 0
    if act.sum() == 0:
        return None, act, p, n
    # d h / d p = n ; d h / d n = -(x - p)  projected on the 2 tangent directions
    X = P[act] - p
    gh_p = np.tile(-n[None, :], (act.sum(), 1))                      # dh/dp = -n
    gh_n = np.stack([X @ t1, X @ t2], axis=1)                        # dh/dn = +(x-p) on T_n
    G = np.concatenate([gh_p, gh_n], axis=1)                         # (M,5)
    F = (G.T @ G) / (sigma ** 2)
    return F, act, p, n


def report(hand, tag):
    P, N = load_surface(hand, tag)
    ext = P.max(0) - P.min(0)
    rows = []
    # probe a few contact normals and depths
    for n_contact in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
        for depth in (0.002, 0.008):
            F, act, p, n = fisher(P, n_contact, depth)
            if F is None:
                continue
            w = np.linalg.eigvalsh(F)
            rank = int((w > 1e-9 * max(w.max(), 1e-12)).sum())
            # CRB for the plane offset (observable scalar) and for the tangential position
            rows.append(dict(n=[float(x) for x in n_contact], depth=depth,
                             n_contact_pts=int(act.sum()),
                             rank=rank, cond=float(w.max() / max(w.min(), 1e-12)),
                             crb_trace=float(np.trace(np.linalg.pinv(F)))))
    return dict(hand=hand, tag=tag, n_points=int(len(P)),
                extent=[float(x) for x in ext], rows=rows)


if __name__ == "__main__":
    out = []
    for hand in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, hand, "tactile_sensor")
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(d + "/*_point.npy")):
            tag = os.path.basename(p).replace("tactileSensor_map_", "").replace("_point.npy", "")
            out.append(report(hand, tag))
    json.dump(out, open("/home/wangrenpeng/bench2dex/theory/t1_results.json", "w"), indent=1)
    print(f"   computed {len(out)} surfaces")
    # show the key structural fact: rank vs number of contacting points
    print(f"\n   {'hand':34s} {'surf':5s} {'pts':>8s} {'接触点':>7s} {'rank':>5s} {'cond':>10s}")
    for r in out[:10]:
        for row in r["rows"][:2]:
            print(f"   {r['hand'][:33]:34s} {r['tag']:5s} {r['n_points']:8d} "
                  f"{row['n_contact_pts']:7d} {row['rank']:5d} {row['cond']:10.2e}")
