"""S6: does the learned tactile monitor overtake the one-line contact statistic
as in-domain rollout data grows?

**为什么有这个实验**：`FAILURE_PREDICTION_RESULTS.md` §5.3 把"每任务 20 集 -> 100 集"
列为"数据量假设的最后一个干净检验"，但那个脚本从未写出。现在 152 集
（52/50/50）已经在盘上，本脚本把它跑掉。

**设计**：同一个数据池、同一批帧、同一批标签、同一折划分、同一个模型结构/训练协议，
**只有子采样集大小在变**。

    tacstat : 一行接触量统计量（全回合平均接触像素数）。无参数、不训练，直接算集级 AUC
              —— 这是"一行代码"基线
    array   : 触觉阵列上的小 CNN（S5 同款结构与训练协议）
    torque  : 手部关节力矩 44 维（低维力通道，参照）
    resid   : 阵列【去掉力矩线性可解释部分】之后（S5 唯一正面证据 auc_array_resid）

**自检**：n=152 / 5 折那一点应当复现 S5 的 `auc_array = 0.692`。复现不了就是流水线错了，
先查流水线再看曲线。

**通道说明**：用 `distance_along_normal_m`（与 S5 一致），**不是** `tacmap`。
DATA-5 已证明两者不是同一个量，所以本脚本的绝对数**不可**与
`FAILURE_PREDICTION_RESULTS.md` 里 tacmap 上的数直接比较；可比的是**曲线形状**。

用法
    python theory/s6_scaling.py --build-cache          # 慢，读 152 集，写 theory/scaling_cache.npz
    python theory/s6_scaling.py --scope pooled         # 主结果
    python theory/s6_scaling.py --scope per-task       # 分任务
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HAND0, HAND1 = 14, 58                      # Sharpa 手部关节（手臂占 0-13）
ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "scaling_cache.npz"
TASKS = ("26", "32", "73")


# ─────────────────────────── 数据 ───────────────────────────

def _blockmax(batch: np.ndarray, G: int) -> np.ndarray:
    """(B,240,240) float32 -> (B,G,G) uint8，块内取最大（与 S5 的 _dp 同一规则）。"""
    B = batch.shape[0]
    bh, bw = 240 // G, 240 // G
    crop = batch[:, : bh * G, : bw * G].reshape(B, G, bh, G, bw)
    return crop.max(axis=(2, 4))


def build_cache(G: int = 20, stride: int = 8, maxdepth_m: float = 0.015) -> None:
    """读全部 152 集，落盘紧凑特征。慢（NFS 上读约 40 GB），只做一次。"""
    items = []
    for t in TASKS:
        for oc in ("success", "failure"):
            for p in sorted((ROLL / t / oc).glob("*.hdf5")):
                items.append((t, oc, p))
    print(f"    {len(items)} 集待读", flush=True)

    AR, TQ, AM, Y, EID, TID, NFR = [], [], [], [], [], [], []
    t0 = time.time()
    for ep, (task, oc, p) in enumerate(items):
        try:
            with h5py.File(p, "r") as f:
                if "qeffort" not in f["robot"] or "tactile" not in f["robot"]:
                    print(f"     skip {p.name}: 缺 qeffort 或 tactile"); continue
                dm = f["robot/tactile"]["distance_along_normal_m"]
                sites = sorted(dm.keys())
                qe = np.asarray(f["robot/qeffort"][:], np.float32)[:, HAND0:HAND1]
                n = int(f["meta/frame_count"][()])
                n = min(n, qe.shape[0], dm[sites[0]].shape[0])
                idx = np.arange(0, n, stride)
                # 一次读一段带步长的 hyperslab，比逐帧点读快得多
                ar = np.zeros((len(idx), len(sites), G, G), np.uint8)
                am = np.zeros(len(idx), np.float32)
                for j, s in enumerate(sites):
                    batch = np.asarray(dm[s][0:n:stride], np.float32)
                    ar[:, j] = (_blockmax(batch, G) / maxdepth_m * 255.0).clip(0, 255).astype(np.uint8)
                    am += (batch > 0).sum(axis=(1, 2))
        except Exception as exc:                                    # noqa: BLE001
            print(f"     !! {p.name}: {type(exc).__name__}: {exc}"); continue
        AR.append(ar); TQ.append(qe[idx]); AM.append(am)
        # 注意：y 和 TID 必须【帧级】，与 eid 对齐 —— 存成集级会让 y[m]（m 是帧掩码）
        # 静默错位。第一版就是这么写的，靠 n_frames 自检（应为 15594）才发现。
        Y.append(np.full(len(idx), 1 if oc == "failure" else 0, np.int64))
        TID.append(np.full(len(idx), TASKS.index(task), np.int64))
        EID.append(np.full(len(idx), ep, np.int64))
        NFR.append(len(idx))
        if (ep + 1) % 20 == 0:
            print(f"     {ep+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(
        CACHE,
        ar=np.concatenate(AR), tq=np.concatenate(TQ), am=np.concatenate(AM),
        y=np.concatenate(Y), eid=np.concatenate(EID), tid=np.concatenate(TID),
        nfr=np.array(NFR, np.int64), G=np.array(G), stride=np.array(stride))
    print(f"   cache -> {CACHE}  ({CACHE.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)")


def load_cache():
    z = np.load(CACHE)
    return (z["ar"].astype(np.float32) / 255.0, z["tq"], z["am"], z["y"], z["eid"], z["tid"])


# ─────────────────────────── 指标 ───────────────────────────

def auc(score, label) -> float:
    score, label = np.asarray(score, float), np.asarray(label)
    pos, neg = score[label > 0], score[label <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def ori(a: float) -> float:
    """有向 AUC —— 符号是一个任何人都免费学到的比特（见 T7 的 BUG-3）。"""
    return max(a, 1 - a) if np.isfinite(a) else float("nan")


def ep_auc(score, y, eid) -> float:
    """帧级分数 -> 每集取均值 -> 集级 AUC。

    ⚠️ 必须先在【集】这一层聚合再算 AUC（S5 的 ep_auc 就是这么做的）。直接拿帧级
    分数算 AUC 是错的：每集约 103 帧且各集帧数不等，帧级 AUC 会被帧数加权，
    实测把同一个统计量从 0.6252 压到 0.5599。
    """
    sc = np.array([score[eid == e].mean() for e in np.unique(eid)])
    lab = np.array([y[eid == e][0] for e in np.unique(eid)])
    return ori(auc(sc, lab))


# ─────────────────────────── 模型（S5 同款） ───────────────────────────

def run_learned_fixed(kind, X, y, eid, tr_eps, te_eps, dev, seed=0, epochs=25):
    """固定测试集版本：在 tr_eps 上训练，只在固定的 te_eps 上评测。

    为什么需要它：`run_learned` 走 5 折，训练集变大时【测试集也变大】，于是"曲线上升"
    里混了"测试集噪声变小"这一项。固定测试集后，只有训练数据量在变。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if kind == "torque":
        net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 64),
                            nn.ReLU(), nn.Linear(64, 1)).to(dev)
    else:
        net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(48, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    mt, mv = np.isin(eid, tr_eps), np.isin(eid, te_eps)
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
        oof = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
    te = np.unique(eid[mv])
    sc = np.array([oof[eid[mv] == e].mean() for e in te])
    lab = np.array([y[eid == e][0] for e in te])
    return ori(auc(sc, lab))


def run_learned(kind, X, y, eid, dev, seed=0, epochs=25):
    """X 必须是 Conv2d 认可的 (N, C, H, W) 布局。

    ⚠️ 布局 bug 记录：S5 把阵列 `AR(N,sites,G,G)` 转置成 `(N,G,G,sites)` 再喂给
    `Conv2d(X.shape[1], ...)`，于是 **20 个网格列被当成了通道**，真正的"每站点 20x20 图"
    从未被卷积看到。S5 的 `auc_array=0.692 / auc_array_resid=0.675` 是这个错布局下的数。
    本脚本默认用**正确布局** `(N,sites,G,G)`；`--repro-s5` 才用错布局，只为复现 S5 的数字。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    eps = np.unique(eid)
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    oof = np.zeros(len(y))
    for va in np.array_split(eps, 5):
        tr = np.setdiff1d(eps, va)
        mt, mv = np.isin(eid, tr), np.isin(eid, va)
        torch.manual_seed(seed)
        if kind == "torque":
            net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 64),
                                nn.ReLU(), nn.Linear(64, 1)).to(dev)
        elif kind in ("array", "resid"):     # S5 里 resid 臂复用同一个 CNN，只是换数据
            net = nn.Sequential(nn.Conv2d(X.shape[1], 24, 3, 2, 1), nn.ReLU(),
                                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                                nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                                nn.Linear(48, 1)).to(dev)
        else:
            raise ValueError(kind)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        yep = np.array([y[eid == e][0] for e in tr])
        pos = {e: i for i, e in enumerate(tr)}
        yt = torch.as_tensor([yep[pos[e]] for e in eid[mt]], device=dev).float()
        Xt = torch.as_tensor(X[mt], device=dev).float()
        Xv = torch.as_tensor(X[mv], device=dev).float()
        ntr = len(yt)
        for _ in range(epochs):
            perm = torch.randperm(ntr, device=dev)
            for b in range(0, ntr, 128):
                s = perm[b:b + 128]
                loss = F.binary_cross_entropy_with_logits(net(Xt[s]).squeeze(-1), yt[s])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            oof[mv] = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
    sc = np.array([oof[eid == e].mean() for e in np.unique(eid)])
    lab = np.array([y[eid == e][0] for e in np.unique(eid)])
    return ori(auc(sc, lab))


# ─────────────────────────── 子采样 ───────────────────────────

def stratified_subsample(tid, y, n, rng):
    """按 (任务, 标签) 分层抽 n 集；池子不够时按比例取。"""
    strata = {}
    for i, (t, l) in enumerate(zip(tid, y)):
        strata.setdefault((int(t), int(l)), []).append(i)
    total = sum(len(v) for v in strata.values())
    n = min(n, total)
    picks = []
    for key, idxs in sorted(strata.items()):
        k = max(1, int(round(n * len(idxs) / total)))
        k = min(k, len(idxs))
        picks.extend(rng.choice(idxs, size=k, replace=False).tolist())
    picks = sorted(set(picks))
    # 分层取整可能多/少几个，按需修剪或补齐
    if len(picks) > n:
        picks = sorted(rng.choice(picks, size=n, replace=False).tolist())
    elif len(picks) < n:
        rest = [i for i in range(total) if i not in set(picks)]
        picks = sorted(picks + rng.choice(rest, size=min(n - len(picks), len(rest)),
                                          replace=False).tolist())
    return np.array(picks)


def subset_frames(eid, picks):
    """把选中的集重编号成 0..n-1，返回帧级掩码与新 eid。"""
    m = np.isin(eid, picks)
    remap = {e: i for i, e in enumerate(picks)}
    return m, np.array([remap[e] for e in eid[m]], np.int64)


# ─────────────────────────── 主流程 ───────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--repro-s5", action="store_true",
                    help="用 S5 的（转置）布局跑，验证流水线能复现 auc_array=0.692")
    ap.add_argument("--scope", default="pooled",
                    choices=["pooled", "per-task", "fixed-test"])
    ap.add_argument("--test-seed", type=int, default=77,
                    help="固定留出集的抽样种子；换它可检验结论对测试集选择的敏感性")
    ap.add_argument("--modalities", default="array,torque,resid")
    ap.add_argument("--test-size", type=int, default=45,
                    help="fixed-test 模式下的固定留出集大小")
    ap.add_argument("--sizes", default="20,40,60,100,152",
                    help="默认锚定到原失败预测研究的 40 集(2 任务)/60 集(3 任务)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "s6_results.json"))
    a = ap.parse_args()

    if a.build_cache or not CACHE.exists():
        build_cache(a.grid, a.stride)
        if a.build_cache:
            return 0

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    AR, TQ, AM, Y, EID, TID = load_cache()
    print(f"    载入缓存: {len(Y)} 帧, {len(np.unique(EID))} 集, 失败率 {Y.mean():.2f}, dev={dev}")

    # resid: 去掉力矩线性可解释部分（S5 的定义）
    A = np.concatenate([TQ, np.ones((len(TQ), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, AR.reshape(len(AR), -1), rcond=None)
    RES = (AR.reshape(len(AR), -1) - A @ coef).reshape(AR.shape).astype(np.float32)
    if a.repro_s5:
        # S5 的真实布局：把网格维当通道。只为复现 auc_array=0.692 / auc_array_resid=0.675。
        AR, RES = AR.transpose(0, 2, 3, 1), RES.transpose(0, 2, 3, 1)
        print("    [repro-s5] 使用 S5 的转置布局（网格维当通道）")

    sizes = [int(s) for s in a.sizes.split(",")]
    results = {"scope": a.scope, "grid": a.grid, "stride": a.stride, "seeds": a.seeds,
               "repro_s5": a.repro_s5,
               "pool_total": int(len(np.unique(EID))), "cells": []}

    if a.scope == "fixed-test":
        # 固定测试集：只有【训练集大小】在变，消除"测试集随 n 变大"这个混淆。
        eids_all = np.unique(EID)
        ep_tid_all = np.array([int(TID[EID == e][0]) for e in eids_all])
        ep_y_all = np.array([int(Y[EID == e][0]) for e in eids_all])
        rng0 = np.random.RandomState(a.test_seed)
        te_idx = stratified_subsample(ep_tid_all, ep_y_all, a.test_size, rng0)
        te_eps = eids_all[te_idx]
        rest = np.setdiff1d(eids_all, te_eps)
        rest_tid = np.array([int(TID[EID == e][0]) for e in rest])
        rest_y = np.array([int(Y[EID == e][0]) for e in rest])
        mv = np.isin(EID, te_eps)
        mods = set(a.modalities.split(","))
        results["test_size"] = int(len(te_eps))
        results["test_eps"] = te_eps.tolist()
        # 统计量也只在固定测试集上评（三个数组必须同为掩码后的，否则错位）
        results["auc_tacstat_test"] = float(ep_auc(-AM[mv], Y[mv], EID[mv]))
        print(f"    固定测试集 {len(te_eps)} 集（失败率 {Y[mv].mean():.3f}）；"
              f"训练池 {len(rest)} 集；测试集上 tacstat={results['auc_tacstat_test']:.3f}")
        for msz in sizes:
            if msz > len(rest):
                continue
            for seed in range(a.seeds):
                rng = np.random.RandomState(2000 + seed)
                tr_eps = rest[stratified_subsample(rest_tid, rest_y, msz, rng)]
                cell = {"scope": "fixed-test", "n": int(len(tr_eps)), "seed": seed,
                        "test_size": int(len(te_eps)), "test_seed": a.test_seed,
                        "auc_tacstat": results["auc_tacstat_test"]}
                for kind, X in (("array", AR), ("torque", TQ), ("resid", RES)):
                    if kind not in mods:
                        continue
                    cell[f"auc_{kind}"] = float(
                        run_learned_fixed(kind, X, Y, EID, tr_eps, te_eps, dev, seed))
                results["cells"].append(cell)
                extra = "  ".join(f"{k[4:]}={cell[k]:.3f}" for k in
                                  ("auc_array", "auc_torque", "auc_resid") if k in cell)
                print(f"   fixed-test train={cell['n']:<4} seed={seed}  "
                      f"tacstat={cell['auc_tacstat']:.3f}  {extra}", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"\n   saved {a.out}")
        return 0

    scopes = [("pooled", None)] if a.scope == "pooled" else [(f"task{t}", i) for i, t in enumerate(TASKS)]
    for name, t_idx in scopes:
        pool = np.arange(len(Y)) if t_idx is None else np.where(TID == t_idx)[0]
        # 显式地在【集】这一层分层，不用帧序推集序（曾写成 eids_pool[sub] 的隐式假设，脆弱）
        eids_pool = np.unique(EID[pool])
        ep_tid = np.array([int(TID[EID == e][0]) for e in eids_pool])
        ep_y = np.array([int(Y[EID == e][0]) for e in eids_pool])
        for n in sizes:
            if n > len(eids_pool):
                continue
            for seed in range(a.seeds):
                rng = np.random.RandomState(1000 + seed)
                sub = stratified_subsample(ep_tid, ep_y, n, rng)
                picks = eids_pool[sub]
                m, eid_new = subset_frames(EID, picks)
                y_new = Y[m]
                cell = {"scope": name, "n": int(len(picks)), "seed": seed,
                        "fail_rate": float(y_new.mean()), "n_frames": int(m.sum()),
                        "auc_tacstat": float(ep_auc(-AM[m], y_new, eid_new))}
                for kind, X in (("array", AR), ("torque", TQ), ("resid", RES)):
                    cell[f"auc_{kind}"] = float(run_learned(kind, X[m], y_new, eid_new, dev, seed))
                results["cells"].append(cell)
                print(f"   {name:<8} n={cell['n']:<4} seed={seed}  "
                      f"tacstat={cell['auc_tacstat']:.3f}  array={cell['auc_array']:.3f}  "
                      f"torque={cell['auc_torque']:.3f}  resid={cell['auc_resid']:.3f}",
                      flush=True)

    # 汇总
    summary = {}
    for cell in results["cells"]:
        for k in ("auc_tacstat", "auc_array", "auc_torque", "auc_resid"):
            summary.setdefault((cell["scope"], cell["n"], k), []).append(cell[k])
    results["summary"] = {f"{s}|{n}|{k}": {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                                           "n_seeds": len(v)}
                          for (s, n, k), v in summary.items()}
    json.dump(results, open(a.out, "w"), indent=1)

    print("\n   ── 均值 ± 标准差（跨子采样种子） ──")
    print(f"   {'scope':<8} {'n':>4}  {'tacstat':>15} {'array(学习)':>15} {'torque(学习)':>15}")
    for (s, n, k) in sorted({(c["scope"], c["n"], 0) for c in results["cells"]}):
        row = []
        for kk in ("auc_tacstat", "auc_array", "auc_torque"):
            d = results["summary"][f"{s}|{n}|{kk}"]
            row.append(f"{d['mean']:.3f}±{d['std']:.3f}")
        print(f"   {s:<8} {n:>4}  {row[0]:>15} {row[1]:>15} {row[2]:>15}")
    print(f"\n   saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
