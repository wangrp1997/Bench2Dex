#!/usr/bin/env python3
"""Tactile SAFETY / FAILURE EARLY PREDICTION on recorded policy rollouts.

Question
--------
Given the observation history up to frame t, will this rollout end in FAILURE?  And --
the metric that actually matters for safety -- **how early** can that be said?

Data
----
`/mnt/public/datasets/bench2dex/rollouts/<task>/{success,failure}/*.hdf5`, produced by
`99_eval_gr00t_record.sh` (which passes `--record-dir --record-all`) and then given
tactile by `106_patch_and_replay.sh`.  Episodes are labelled by their subdirectory.

Per modality we train the SAME encoder shape and compare:
  * tactile  : set over the 10 fingertip sites, MAX-pooled (contact is sparse), GRU over time
  * rgb      : single overhead camera, gray 64x64, GRU
  * proprio  : joint positions, GRU
  * fusion   : logit average
Metric: AUC over (episode, frame) samples, split into time buckets so the result reads
as a prediction TIMELINE rather than a single number.  Splits are by EPISODE.

Usage: 107_failure_early_warning.py [--window 6] [--epochs 30] [--stride 5]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
LAB = Path("/home/wangrenpeng/bench2dex/data/rollout_labels")
TAC_GRID, RGB_GRID, POLICY_STRIDE = 24, 64, 3
TASKS = {"26": "26_canned_food_tray_line_arrangement",
         "32": "32_baking_tray_prep_with_tools",
         "73": "73_jigsaw_puzzle_assembly"}


def decode_rgb(raw):
    import cv2
    r = np.asarray(raw)
    if r.ndim == 1 and r.size > 0:
        im = cv2.imdecode(np.frombuffer(r, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            return cv2.resize(g, (RGB_GRID, RGB_GRID), interpolation=cv2.INTER_AREA)
    return np.zeros((RGB_GRID, RGB_GRID), np.uint8)


def decode_rgb_color(raw, res: int):
    """Decode the JPEG and resize to (res,res,3) RGB."""
    import cv2
    r = np.asarray(raw)
    if r.ndim == 1 and r.size > 0:
        im = cv2.imdecode(np.frombuffer(r, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return np.zeros((res, res, 3), np.uint8)


def load_task(task_id: str, scene: str, stride: int, max_ep: int,
              rgb_res: int = RGB_GRID, rgb_color: bool = False, rgb_cams=None):
    import h5py
    T, R, P, Y, EP, FR, TIME, L = [], [], [], [], [], [], [], []
    ep_id = 0
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task_id / outcome).glob("*.hdf5"))[:max_ep]:
            lab = LAB / scene / outcome / (p.stem + ".npz")
            if not lab.exists():
                continue
            try:
                with h5py.File(p, "r") as f:
                    Tn = int(f["meta/frame_count"][()])
                    qp = np.asarray(f["robot/qpos"][:], np.float32)
                    tac_grp = f["robot/tactile"]["tacmap"]
                    sites = list(tac_grp.keys())
                    cams = [c for c in (rgb_cams or ["cam_overhead"]) if c in f["cameras"]]
                    if not cams:
                        cams = list(f["cameras"].keys())[:1]
                    rgbd_all = {c: f["cameras"][c]["rgb"] for c in cams}
                    n_rgb = len(rgbd_all[cams[0]])
                    idx = np.arange(0, Tn, stride)
                    tac = np.zeros((len(idx), len(sites), TAC_GRID, TAC_GRID), np.uint8)
                    for j, i in enumerate(idx):
                        for si, s in enumerate(sites):
                            a = np.asarray(tac_grp[s][i], np.uint8)
                            bh, bw = a.shape[0] // TAC_GRID, a.shape[1] // TAC_GRID
                            tac[j, si] = a[:bh * TAC_GRID, :bw * TAC_GRID].reshape(
                                TAC_GRID, bh, TAC_GRID, bw).max(axis=(1, 3))
                    if rgb_color:
                        rgb = np.zeros((len(idx), len(cams), rgb_res, rgb_res, 3), np.uint8)
                    else:
                        rgb = np.zeros((len(idx), RGB_GRID, RGB_GRID), np.uint8)
                    for j, i in enumerate(idx):
                        k = min(n_rgb - 1, int(i * n_rgb / Tn))   # cameras are stride-subsampled
                        if rgb_color:
                            for ci, c in enumerate(cams):
                                rgb[j, ci] = decode_rgb_color(rgbd_all[c][k], rgb_res)
                        else:
                            rgb[j] = decode_rgb(rgbd_all[cams[0]][k])
                    prop = qp[idx]
                    _lz = np.load(lab, allow_pickle=True)
                    lsc = np.asarray(_lz["lscr"], np.float32)[idx]
            except Exception as exc:  # noqa: BLE001
                print(f"   !! {p.name}: {type(exc).__name__}: {exc}")
                continue
            n = len(idx)
            T.append(tac); R.append(rgb); P.append(prop)
            if lsc is None or len(lsc) != n:
                L.append(np.zeros(n, np.float32))
            else:
                L.append(lsc.astype(np.float32))
            Y.append(np.full(n, 1 if outcome == "failure" else 0, np.int64))
            EP.append(np.full(n, ep_id, np.int64))
            # ABSOLUTE frame index, not fraction-of-episode: successful rollouts early-stop
            # (measured T=582-690) while failures run the full budget (T=871), so normalising
            # by the final length would leak the outcome into the time bucket.
            FR.append(idx.astype(np.float32))
            TIME.append(np.full(n, task_id, dtype="U2"))
            ep_id += 1
    if not T:
        return None
    return dict(tac=np.concatenate(T), rgb=np.concatenate(R), prop=np.concatenate(P),
                y=np.concatenate(Y), ep=np.concatenate(EP),
                frac=np.concatenate(FR).astype(np.float32), task=np.concatenate(TIME),
                lscr=np.concatenate(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-ep", type=int, default=60)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="CONTROL: permute the episode labels. A correct pipeline must then "
                         "give AUC ~0.5 for every modality; anything far from 0.5 means a "
                         "systematic bias in the evaluation rather than a real signal.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--vision-res", type=int, default=160,
                    help="square resolution fed to the frozen vision encoder")
    ap.add_argument("--vision-cams", default="cam_overhead,cam_wrist_left,cam_wrist_right",
                    help="cameras given to the vision baseline. Using only the overhead view "
                         "would understate vision: in manipulation the wrist cameras are the "
                         "main visual source.")
    ap.add_argument("--vision-encoder", default="tiny",
                    choices=["tiny", "resnet18", "dinov2"],
                    help="'tiny' is the 32K hand-rolled CNN on 64x64 grey. That is NOT a "
                         "credible vision baseline, so for any vision-vs-touch claim use a "
                         "FROZEN pretrained encoder (resnet18 / dinov2, weights already cached).")
    ap.add_argument("--steps-per-epoch", type=int, default=12,
                    help="gradient steps per epoch in the episode-level objective. Too few "
                         "steps leaves the model dominated by its random response to INPUT "
                         "MAGNITUDE, and since successful runs have more contact and richer "
                         "texture the ranking comes out INVERTED (measured AUC 0.05-0.36).")
    ap.add_argument("--tasks", default="26")
    ap.add_argument("--modalities", default="tac,rgb,prop,tac+rgb")
    ap.add_argument("--target", default="outcome", choices=["outcome", "stall"],
                    help="outcome: will the episode fail at all (label constant per episode); "
                         "stall: will progress fail to advance within the next --horizon frames "
                         "(per-frame, actionable)")
    ap.add_argument("--horizon", type=int, default=20,
                    help="frames (in sampled units) used by the stall target")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    parts = []
    for tid in args.tasks.split(","):
        _vis = args.vision_encoder != "tiny"
        d = load_task(tid, TASKS[tid], args.stride, args.max_ep,
                      rgb_res=args.vision_res if _vis else RGB_GRID, rgb_color=_vis,
                      rgb_cams=args.vision_cams.split(",") if _vis else None)
        if d is not None:
            parts.append(d)
            print(f"   task {tid}: {len(np.unique(d['ep']))} episodes, {len(d['y'])} frames, "
                  f"failure rate {d['y'].mean():.2f}")
    if not parts:
        print("   no data yet"); return 1
    off = 0
    for d in parts:
        d["ep"] = d["ep"] + off
        off = int(d["ep"].max()) + 1
    D = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    print(f"   TOTAL: {len(np.unique(D['ep']))} episodes, {len(D['y'])} frames, "
          f"failure rate {D['y'].mean():.2f}")

    if args.target == "stall":
        l = D["lscr"].copy()
        lab = np.zeros(len(l), np.int64)
        for e in np.unique(D["ep"]):
            idx = np.where(D["ep"] == e)[0]
            f = l[idx]
            # 1 = progress does NOT advance within the next `horizon` sampled frames
            for k, j in enumerate(idx):
                fut = f[k + 1:k + 1 + args.horizon]
                lab[j] = 1 if (len(fut) == 0 or fut.max() <= f[k] + 1e-6) else 0
        D["y"] = lab
        print(f"   TARGET=stall (horizon {args.horizon}): positive rate {lab.mean():.2f}")

    eps = np.unique(D["ep"])
    if args.shuffle_labels:
        _r = np.random.RandomState(123)
        _perm = _r.permutation(len(eps))
        _map = {e: eps[_perm[i]] for i, e in enumerate(eps)}
        _newy = np.zeros_like(D["y"])
        for e in eps:                       # give each episode another episode's label
            _newy[D["ep"] == e] = D["y"][D["ep"] == _map[e]][0]
        D["y"] = _newy
        print("   [CONTROL] episode labels permuted")
    rng = np.random.RandomState(0)
    rng.shuffle(eps)
    n_folds = max(2, min(args.folds, len(eps) // 2))
    folds = np.array_split(eps, n_folds)
    print(f"   {len(eps)} episodes -> {n_folds}-fold EPISODE-level cross-validation "
          f"(all episodes get evaluated exactly once)")

    W = args.window

    def windows_of(ep_set, d_ep):
        out = []
        for e in ep_set:
            idx = np.where(d_ep == e)[0]
            if len(idx) >= W:
                out.extend((idx[W - 1::max(1, args.stride)]).tolist())
        return np.array(out, dtype=np.int64)

    if args.vision_encoder != "tiny":
        import vision_feats
        print(f"   encoding RGB with FROZEN {args.vision_encoder} "
              f"({len(D['rgb'])} frames) ...", flush=True)
        _f = vision_feats.encode(D["rgb"], args.vision_encoder,
                                 tag=f"{args.stride}_{args.max_ep}_{args.vision_cams}",
                                 size=args.vision_res)
        D["rgb"] = _f.astype(np.float32)
        print(f"      -> frozen features {D['rgb'].shape}")
    tac_t = torch.as_tensor(D["tac"]); rgb_t = torch.as_tensor(D["rgb"])
    prop_t = torch.as_tensor(D["prop"]); y_t = torch.as_tensor(D["y"])
    frac_t = torch.as_tensor(D["frac"]); ep_t = torch.as_tensor(D["ep"])

    def gather(idx, dev):
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        return (tac_t[ii].to(dev).float(), rgb_t[ii].to(dev).float(),
                prop_t[ii].to(dev), y_t[idx].to(dev), frac_t[idx].to(dev))

    class TacEnc(nn.Module):
        def __init__(self, d=64, n_slots=10):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d(1))          # MAX: sparse contact
            self.proj = nn.Linear(32, d)
            self.slot = nn.Embedding(n_slots, d)  # semantic slot, not list position
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            x = x / 255.0                      # raw maps are 0-255; unnormalised input
            B, W_, S, G, _ = x.shape           # made training diverge (learned AUC < chance)
            f = self.cnn(x.reshape(B * W_ * S, 1, G, G)).reshape(B * W_, S, -1)
            f = (self.proj(f) + self.slot.weight[None]).max(1).values
            h, _ = self.gru(f.reshape(B, W_, -1))
            return self.head(h[:, -1]).squeeze(-1)

    class ImgEnc(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1))
            self.proj = nn.Linear(32, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            x = (x / 255.0 - 0.5) / 0.5
            B, W_, G, _ = x.shape
            f = self.proj(self.cnn(x.reshape(B * W_, 1, G, G)).reshape(B * W_, -1))
            h, _ = self.gru(f.reshape(B, W_, -1))
            return self.head(h[:, -1]).squeeze(-1)

    class FeatEnc(nn.Module):
        """Small trainable head on FROZEN pretrained vision features (no CNN training).

        The frozen features are z-scored with per-fold TRAIN statistics. Without this the
        head partly responds to feature MAGNITUDE, and because successful runs end on a
        visually "finished" scene the ranking comes out systematically inverted
        (measured epAUC 0.33 against a 32K tactile encoder)."""

        def __init__(self, feat_dim, d=64, mean=None, std=None):
            super().__init__()
            self.register_buffer("fmean", torch.zeros(1, 1, feat_dim) if mean is None
                                 else torch.as_tensor(mean).reshape(1, 1, -1).float())
            self.register_buffer("fstd", torch.ones(1, 1, feat_dim) if std is None
                                 else torch.as_tensor(std).reshape(1, 1, -1).float().clamp(min=1e-3))
            self.proj = nn.Linear(feat_dim, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, 1)

        def forward(self, x):                     # (B,W,D)
            f = torch.relu(self.proj((x - self.fmean) / self.fstd))
            h, _ = self.gru(f)
            return self.head(h[:, -1]).squeeze(-1)

    class PropEnc(nn.Module):
        """Joint positions need standardising: raw qpos spans +-3 rad, and feeding that
        unnormalised was one reason the learned models scored below chance."""

        def __init__(self, d=64, mean=None, std=None):
            super().__init__()
            self.register_buffer("mean", torch.zeros(1, 1, 58) if mean is None
                                 else torch.as_tensor(mean).reshape(1, 1, -1).float())
            self.register_buffer("std", torch.ones(1, 1, 58) if std is None
                                 else torch.as_tensor(std).reshape(1, 1, -1).float().clamp(min=1e-3))
            self.proj = nn.Linear(58, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            f = torch.relu(self.proj((x - self.mean) / self.std))
            h, _ = self.gru(f)
            return self.head(h[:, -1]).squeeze(-1)

    def auc(p, y):
        p, y = np.asarray(p), np.asarray(y)
        pos, neg = p[y > 0], p[y <= 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        return float((pos[:, None] > neg[None, :]).mean()
                     + 0.5 * (pos[:, None] == neg[None, :]).mean())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    buckets_of = lambda F_: {
        "f0-100": F_ < 100, "f100-200": (F_ >= 100) & (F_ < 200),
        "f200-300": (F_ >= 200) & (F_ < 300),
        "f300-450": (F_ >= 300) & (F_ < 450), "f450+": F_ >= 450}

    results = {}
    for name in args.modalities.split(","):
        use = (lambda m: True) if name == "all" else (lambda m: m in name)
        P_all, Y_all, F_all, E_all = [], [], [], []
        for k, val_eps in enumerate(folds):
            train_eps = np.setdiff1d(eps, val_eps)
            tr_idx = windows_of(train_eps, D["ep"])
            va_idx = windows_of(val_eps, D["ep"])
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            torch.manual_seed(0)
            enc_t = TacEnc().to(dev) if use("tac") else None
            if use("rgb"):
                if args.vision_encoder == "tiny":
                    enc_r = ImgEnc().to(dev)
                else:
                    fw = rgb_t[tr_idx].float()
                    enc_r = FeatEnc(D["rgb"].shape[1], mean=fw.mean((0, 1)).numpy(),
                                    std=fw.std((0, 1)).numpy()).to(dev)
            else:
                enc_r = None
            if use("prop"):
                pw = prop_t[tr_idx].float()
                enc_p = PropEnc(mean=pw.mean((0, 1)).numpy(), std=pw.std((0, 1)).numpy()).to(dev)
            else:
                enc_p = None
            params = [q for e in (enc_t, enc_r, enc_p) if e is not None for q in e.parameters()]
            opt = torch.optim.Adam(params, lr=2e-3, weight_decay=1e-4)
            def logit_of(b):
                tt, rr, pp, _, _ = gather(b, dev)
                outs = []
                if enc_t is not None:
                    outs.append(enc_t(tt))
                if enc_r is not None:
                    outs.append(enc_r(rr))
                if enc_p is not None:
                    outs.append(enc_p(pp))
                return sum(outs) / len(outs)

            # EPISODE-LEVEL supervision. Window-level BCE asks an almost unlearnable
            # question (a single 6-frame window barely separates the classes: winAUC ~0.54)
            # and we measured that it makes the learned model anti-correlated at the
            # episode level (epAUC 0.36) while a trivial contact statistic reaches 0.96.
            # The deployment question is "judging by everything so far, will this run fail?",
            # so we pool each sampled episode's windows into ONE score and apply BCE there.
            tr_by_ep = {e: g for e, g in
                        ((e, np.where(D["ep"] == e)[0]) for e in train_eps)}
            tr_by_ep = {e: (g[g >= W - 1][W - 1::max(1, args.stride)]) for e, g in tr_by_ep.items()}
            tr_by_ep = {e: g for e, g in tr_by_ep.items() if len(g) > 0}
            ep_keys = list(tr_by_ep.keys())
            ep_labels = {e: float(D["y"][tr_by_ep[e][0]]) for e in ep_keys}
            for _ in range(args.epochs):
                [e.train() for e in (enc_t, enc_r, enc_p) if e is not None]
                for _ in range(args.steps_per_epoch):
                    sel = np.random.choice(ep_keys, min(args.batch, len(ep_keys)), replace=False)
                    logs, labs = [], []
                    for e in sel:
                        g = tr_by_ep[e]
                        if len(g) > 32:
                            g = g[np.random.choice(len(g), 32, replace=False)]
                        logs.append(logit_of(g).mean())      # one score per episode
                        labs.append(ep_labels[e])
                    logits = torch.stack(logs)
                    ylabel = torch.tensor(labs, device=dev, dtype=torch.float32)
                    loss = F.binary_cross_entropy_with_logits(logits, ylabel)
                    opt.zero_grad(); loss.backward(); opt.step()
            [e.eval() for e in (enc_t, enc_r, enc_p) if e is not None]
            with torch.no_grad():
                for i in range(0, len(va_idx), 256):
                    b = va_idx[i:i + 256]
                    tt, rr, pp, yy, ff = gather(b, dev)
                    outs = []
                    if enc_t is not None:
                        outs.append(enc_t(tt))
                    if enc_r is not None:
                        outs.append(enc_r(rr))
                    if enc_p is not None:
                        outs.append(enc_p(pp))
                    P_all.append(torch.sigmoid(sum(outs) / len(outs)).cpu().numpy())
                    Y_all.append(yy.cpu().numpy()); F_all.append(ff.cpu().numpy())
                    E_all.append(ep_t[b].numpy())
        P_, Y_, F_ = np.concatenate(P_all), np.concatenate(Y_all), np.concatenate(F_all)
        E_ = np.concatenate(E_all)
        # ---- EPISODE-level aggregation, the right granularity for a safety warning:
        # one score per (episode, time bucket) = mean of that episode's window scores up to
        # the bucket. Windows of one episode are highly correlated, so a window-level AUC
        # would be badly over-confident and would not answer "how early can we warn".
        def auc_ep(pred, label, bucket_mask):
            sc, lb = [], []
            for e in np.unique(E_[bucket_mask]):
                m = bucket_mask & (E_ == e)
                if m.sum() == 0:
                    continue
                sc.append(pred[m].mean()); lb.append(label[m][0])
            return auc(sc, lb)
        bk = buckets_of(F_)
        # ---- "lead time": the earliest absolute frame at which the monitor can already
        # flag failure with AUC >= 0.8 using ONLY the frames seen so far. This is the
        # number a safety system is actually judged on, and it is what "prediction lead
        # time" means here.
        grid = [50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800]
        det = {}
        for g in grid:
            det[g] = auc_ep(P_, Y_, F_ <= g)
        first_ok = next((g for g in grid if det[g] == det[g] and det[g] >= 0.8), None)
        results[name] = {"overall": auc_ep(P_, Y_, np.ones(len(P_), bool)),
                         **{b: auc_ep(P_, Y_, m) for b, m in bk.items()},
                         "window_auc": auc(P_, Y_), "n_windows": int(len(P_)),
                         "n_episodes": int(len(np.unique(E_))),
                         "detection": {str(g): det[g] for g in grid},
                         "lead_time_frame": first_ok}
        np.savez_compressed(
            Path(f"/home/wangrenpeng/bench2dex/data/ew_preds_{name.replace('+','_')}.npz"),
            pred=P_, y=Y_, frame=F_, ep=E_)
        print(f"   {name:10s} epAUC={results[name]['overall']:.3f} "
              f"(winAUC={results[name]['window_auc']:.3f})  "
              f"lead>=0.8 @ frame {first_ok}   "
              + "  ".join(f"{b}={results[name][b]:.3f}" for b in bk), flush=True)

    # ---- hand-crafted tactile baseline: no learning, just contact statistics ----
    # This is the bar our learned encoder has to clear: if "how many fingertips are in
    # contact" already predicts the outcome, a deep model that only matches it adds nothing.
    def tac_stats(idx):
        ii = idx[:, None] + np.arange(-W + 1, 1)[None, :]
        c = (tac_t[ii].float() > 0).float()                 # (B,W,10,G,G)
        n_sites = (c.mean(dim=(3, 4)) > 0).float().mean(dim=1)   # mean # contacting sites
        density = c.mean(dim=(1, 2, 3, 4))                        # contact pixel density
        return n_sites.numpy(), density.numpy()

    all_idx = windows_of(eps, D["ep"])       # hand-crafted stats need no training
    if len(all_idx):
        ns, dn = tac_stats(all_idx)
        yv = y_t[all_idx].numpy(); fv = frac_t[all_idx].numpy()
        bk = {"f0-100": fv < 100, "f100-200": (fv >= 100) & (fv < 200),
              "f200-300": (fv >= 200) & (fv < 300),
              "f300-450": (fv >= 300) & (fv < 450), "f450+": fv >= 450}
        ev = ep_t[all_idx].numpy()
        for nm2, vec in (("tacstat(nsites)", ns), ("tacstat(density)", dn)):
            # a HIGHER contact statistic means a HEALTHIER episode, so flip the sign to
            # make it a failure score (otherwise it reads as a below-chance predictor)
            sc = -vec
            d2 = {"overall": auc_ep(sc, yv, np.ones(len(sc), bool)),
                  "window_auc": auc(sc, yv), "n_episodes": int(len(np.unique(ev)))}
            for b, m in bk.items():
                scl, lbl = [], []
                for e in np.unique(ev[m]):
                    mm = m & (ev == e)
                    scl.append(sc[mm].mean()); lbl.append(yv[mm][0])
                d2[b] = auc(scl, lbl)
            results[nm2] = d2
            print(f"   {nm2:18s} epAUC={d2['overall']:.3f} (winAUC={d2['window_auc']:.3f})   " +
                  "  ".join(f"{b}={d2[b]:.3f}" for b in bk))

    print(f"\n   baseline: always-fail predicts AUC 0.5 (overall failure rate "
          f"{D['y'].mean():.2f}); {n_folds}-fold CV over {len(eps)} episodes")
    print(f"\n   {'modality':12s} {'overall':>8s} {'f0-100':>8s} {'f100-200':>9s} "
          f"{'f200-300':>9s} {'f300-450':>9s} {'f450+':>7s}")
    for k, v in results.items():
        print(f"   {k:18s} {v['overall']:8.3f} {v['f0-100']:8.3f} {v['f100-200']:9.3f} "
              f"{v['f200-300']:9.3f} {v['f300-450']:9.3f} {v['f450+']:7.3f}")
    # ---- detection curves: how the AUC grows with the number of frames seen --------
    if results:
        print(f"\n   detection curve (AUC using only frames up to t)")
        grid = [50, 100, 150, 200, 300, 400, 600]
        hdr = "   " + f"{'modality':18s}" + "".join(f"{('t=' + str(g)):>9s}" for g in grid)
        print(hdr)
        for k, v in results.items():
            if "detection" not in v:
                continue
            print(f"   {k:18s}" + "".join(
                f"{(v['detection'].get(str(g), float('nan'))):9.3f}" for g in grid))
        print(f"\n   earliest frame with AUC>=0.8: " + "  ".join(
            f"{k}={'never' if v.get('lead_time_frame') is None else v['lead_time_frame']}"
            for k, v in results.items() if 'lead_time_frame' in v))

    out = Path("/home/wangrenpeng/bench2dex/data/failure_early_warning.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n   saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
