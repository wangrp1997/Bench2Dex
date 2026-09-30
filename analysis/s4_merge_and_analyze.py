"""S4: merge the REAL joint torques into the replayed (tactile-carrying) rollouts, then run
the dual-modality comparison.

Why merging is valid: replay.py reconstructs the SAME qpos trajectory that was recorded, so
frame i of the recorded rollout and frame i of the replayed rollout are the same robot state.
The recorded file has the physically-computed torques (real dynamics) but no tactile; the
replayed file has tactile but only degenerate kinematic torques. Taking qeffort from the
former and tactile from the latter gives both, correctly.
"""
import h5py, numpy as np, glob, os, json, sys
from pathlib import Path

REC = Path("/home/wangrenpeng/bench2dex/output/metric")
REP = Path("/mnt/public/datasets/bench2dex/rollouts")
TASKS = ["26", "32", "73"]


def recorded_dir(t):
    ds = [d for d in REC.glob(f"gr00t_{t}_*") if (d / "rollouts").is_dir()]
    ds.sort(key=lambda d: (d / "rollouts").stat().st_mtime)
    return ds[-1] / "rollouts" if ds else None


def merge(t):
    rd = recorded_dir(t)
    if rd is None:
        print(f"   task {t}: 无录制目录"); return 0
    n_ok = n_skip = 0
    for outcome in ("success", "failure"):
        for src in sorted((rd / outcome).glob("*.hdf5")):
            dst = REP / t / outcome / src.name
            if not dst.exists():
                n_skip += 1; continue
            try:
                with h5py.File(src, "r") as f:
                    if "qeffort" not in f["robot"]:
                        n_skip += 1; continue
                    qe = np.asarray(f["robot/qeffort"][:], np.float32)
                with h5py.File(dst, "r+") as g:
                    n = int(g["meta/frame_count"][()]) if "meta/frame_count" in g else None
                    if n is not None and len(qe) != n:
                        print(f"      !! 帧数不符 {src.name}: qeffort {len(qe)} vs frame_count {n}")
                        n_skip += 1; continue
                    if "qeffort" in g["robot"]:
                        del g["robot/qeffort"]
                    g["robot"].create_dataset("qeffort", data=qe)
                n_ok += 1
            except Exception as e:
                print(f"      !! {src.name}: {type(e).__name__} {e}")
    print(f"   task {t}: 合并 {n_ok} 个, 跳过 {n_skip} 个")
    return n_ok


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())


def ori(a): return max(a, 1-a) if np.isfinite(a) else float("nan")


def main():
    if "--merge-only" in sys.argv:
        for t in TASKS: merge(t)
        return
    for t in TASKS: merge(t)
    print("\n   合并完成。跑对照请用 s3（会读 robot/qeffort 若存在）。")


if __name__ == "__main__":
    main()
