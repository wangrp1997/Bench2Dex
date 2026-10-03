"""S6b: 诊断 S5/失败预测所用评测协议的一个缺陷 —— 跨折拼接未校准的 sigmoid 分数。

**背景**：S6 的两个协议给出互相矛盾的结论。

    pooled 5 折（S5 协议）   n=60 时 array=0.589   —— 学习臂 ≈ 甚至输给一行统计量
    fixed-test (45 集固定)   train=48 时 array=0.753 —— 学习臂大胜

同一批数据、同一个模型、同一批超参。差别只在**评测协议**。

**假设**：`run_learned` 把 5 个**各自独立训练**的模型在验证折上的 sigmoid 输出直接拼成
一个数组，再对这个数组算一个 AUC。5 个模型的输出尺度（校准）互不相同，于是拼接后的
排序被折叠到同一尺度上比较 —— 折与折之间的系统性偏移会污染 AUC。训练集越小，
5 个模型差异越大，污染越重 ⇒ 这正好解释"数据越少、学习臂越差"的假象。

**本脚本对同一批折同时算三种指标**：
    (a) pooled        ：拼接后的单次 AUC（现协议）
    (b) per-fold mean ：各折各自算 AUC 再平均（免疫折间校准差）
    (c) rank-pooled   ：拼接前先在折内做秩归一化，再算单次 AUC
并报告**折间校准差**（各折验证集平均分数的极差），用来直接指认机制。

用法: python theory/s6b_protocol_check.py --n 60 --seeds 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_s6():
    spec = importlib.util.spec_from_file_location("s6", HERE / "s6_scaling.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s6"] = m
    spec.loader.exec_module(m)
    return m


def five_fold_scores(s6, X, y, eid, dev, seed=0, epochs=25):
    """跑 S5 同款 5 折，但把每折的验证分数与标签分别留存。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    eps = np.unique(eid)
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    oof = np.zeros(len(y))
    folds = []
    for va in np.array_split(eps, 5):
        tr = np.setdiff1d(eps, va)
        mt, mv = np.isin(eid, tr), np.isin(eid, va)
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(48, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        yt = torch.as_tensor([y[eid == e][0] for e in eid[mt]], device=dev).float()
        Xt = torch.as_tensor(X[mt], device=dev).float()
        Xv = torch.as_tensor(X[mv], device=dev).float()
        for _ in range(epochs):
            perm = torch.randperm(len(yt), device=dev)
            for b in range(0, len(yt), 128):
                s = perm[b:b + 128]
                loss = F.binary_cross_entropy_with_logits(net(Xt[s]).squeeze(-1), yt[s])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            oof[mv] = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
        ve = np.unique(eid[mv])
        sc = np.array([oof[eid == e].mean() for e in ve])
        lab = np.array([y[eid == e][0] for e in ve])
        folds.append({"eps": ve, "score": sc, "label": lab})
    return oof, folds


def rank01(v):
    order = np.argsort(np.argsort(v))
    return order / max(len(v) - 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "s6b_results.json"))
    a = ap.parse_args()

    import torch
    s6 = _load_s6()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    AR, TQ, AM, Y, EID, TID = s6.load_cache()

    eids_all = np.unique(EID)
    ep_tid = np.array([int(TID[EID == e][0]) for e in eids_all])
    ep_y = np.array([int(Y[EID == e][0]) for e in eids_all])

    rows = []
    for seed in range(a.seeds):
        rng = np.random.RandomState(1000 + seed)
        sub = s6.stratified_subsample(ep_tid, ep_y, a.n, rng)
        picks = eids_all[sub]
        m, eid_new = s6.subset_frames(EID, picks)
        y_new = Y[m]
        oof, folds = five_fold_scores(s6, AR[m], y_new, eid_new, dev, seed)

        # (a) 现协议：拼接后单次 AUC
        sc = np.array([oof[eid_new == e].mean() for e in np.unique(eid_new)])
        lab = np.array([y_new[eid_new == e][0] for e in np.unique(eid_new)])
        pooled = s6.ori(s6.auc(sc, lab))
        # (b) 各折 AUC 平均
        per_fold = [s6.ori(s6.auc(f["score"], f["label"])) for f in folds]
        # (c) 折内秩归一化后再拼接
        norm = np.concatenate([rank01(f["score"]) for f in folds])
        norm_lab = np.concatenate([f["label"] for f in folds])
        rank_pooled = s6.ori(s6.auc(norm, norm_lab))
        # 折间校准差：各折验证集平均分数的极差
        calib = float(np.ptp([f["score"].mean() for f in folds]))
        stat = s6.ep_auc(-AM[m], y_new, eid_new)

        rows.append({"seed": seed, "pooled": pooled, "per_fold_mean": float(np.mean(per_fold)),
                     "rank_pooled": rank_pooled, "calib_spread": calib, "tacstat": stat,
                     "fold_aucs": [float(x) for x in per_fold]})
        print(f"   n={a.n} seed={seed}  tacstat={stat:.3f} | "
              f"(a)拼接={pooled:.3f}  (b)折均={np.mean(per_fold):.3f}  "
              f"(c)秩归一={rank_pooled:.3f} | 折间校准差={calib:.3f}", flush=True)

    agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
               "std": float(np.std([r[k] for r in rows]))}
           for k in ("pooled", "per_fold_mean", "rank_pooled", "calib_spread", "tacstat")}
    print(f"\n   n={a.n} 汇总（{a.seeds} 个种子）")
    for k, v in agg.items():
        print(f"     {k:<16} {v['mean']:.3f} ± {v['std']:.3f}")
    json.dump({"n": a.n, "rows": rows, "summary": agg}, open(a.out, "w"), indent=1)
    print(f"   saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
