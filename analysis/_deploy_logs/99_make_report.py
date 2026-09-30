#!/usr/bin/env python3
"""Aggregate Bench2Dex Sharpa baseline evaluations into RESULTS.md.

Sections: 1 success rate | 2 LSCR | 3 progress | 4 deviations | 5 incidents | 6 artifacts
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import time
from datetime import datetime

ROOT = "/home/wangrenpeng/bench2dex"
METRIC = f"{ROOT}/output/metric"
OUT_MD = f"{ROOT}/RESULTS.md"
LOGS = f"{ROOT}/_deploy_logs"
ETA_STATE = f"{LOGS}/99_eta_state.json"

TASKS = ["26", "32", "73"]
PAPER = {  # project page data.js, None channel, out of 50
    "ACT":   {"26": 17, "32": 9,  "73": 0},
    "DP":    {"26": 0,  "32": 2,  "73": 0},
    "PI05":  {"26": 15, "32": 22, "73": 0},
    "GR00T": {"26": 17, "32": 12, "73": 10},
}
ORDER = ["ACT", "DP", "PI05", "GR00T"]
PATTERNS = [
    (re.compile(r"^(26|32|73)_iiwa7_sharpa"), "ACT"),
    (re.compile(r"^dp_(26|32|73)_"), "DP"),
    (re.compile(r"^gr00t_(26|32|73)_"), "GR00T"),
    (re.compile(r"^pi05_(26|32|73)_"), "PI05"),
]


def collect():
    """{(algo, task): {dir, episodes, mtime}} -- newest non-empty dir per cell."""
    found = {}
    for d in glob.glob(f"{METRIC}/*"):
        if not os.path.isdir(d):
            continue
        name, algo, task = os.path.basename(d), None, None
        for rx, a in PATTERNS:
            m = rx.match(name)
            if m:
                algo, task = a, m.group(1)
                break
        if algo is None:
            continue
        f = os.path.join(d, "per_episode.jsonl")
        if not os.path.exists(f):
            continue
        eps = []
        try:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        try:
                            eps.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            continue
        if not eps:
            continue
        mt = os.path.getmtime(f)
        if (algo, task) in found and found[(algo, task)]["mtime"] >= mt:
            continue
        found[(algo, task)] = {"dir": d, "episodes": eps, "mtime": mt}
    return found


def stats(eps):
    n = len(eps)
    succ = sum(1 for e in eps if e.get("stable_success") or e.get("success"))
    l = [e.get("latched_stage_completion_rate") for e in eps
         if e.get("latched_stage_completion_rate") is not None]
    return {
        "n": n, "succ": succ, "rate": (100.0 * succ / n) if n else 0.0,
        "mean": st.mean(l) if l else None, "med": st.median(l) if l else None,
        "min": min(l) if l else None, "max": max(l) if l else None,
        "zeros": sum(1 for v in l if v == 0), "ln": len(l),
    }


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def main():
    data = collect()
    now, nowts = datetime.now(), time.time()
    L = []
    A = L.append

    A("# Bench2Dex — Sharpa baseline results\n")
    A(f"`multi_iiwa7_with_sharpa` · tasks 26/32/73 · profile `none` · 50 episodes · "
      f"seed 100000000 · GPU 2 · generated {now:%Y-%m-%d %H:%M}\n")
    A("Paper column = the project page's own per-task numbers "
      "(`bench2dex.github.io/assets/js/data.js`), None channel, out of 50.\n")

    # ── 1. success rate ────────────────────────────────────────────────────
    A("\n## 1. Stable success rate (/50)\n")
    A("| Algorithm | 26 Canned-Food | 32 Baking Tray | 73 Jigsaw | paper 26/32/73 |")
    A("|---|---|---|---|---|")
    for algo in ORDER:
        cells = []
        for t in TASKS:
            d = data.get((algo, t))
            if d is None:
                cells.append("—")
                continue
            s = stats(d["episodes"])
            mark = "" if s["n"] >= 50 else f" ({s['n']}/50)"
            cells.append(f"**{s['succ']}/{s['n']}** = {s['rate']:.0f}%{mark}")
        pv = "/".join(str(PAPER[algo][t]) for t in TASKS)
        A(f"| **{algo}** | " + " | ".join(cells) + f" | {pv} |")

    flags = []
    for algo in ORDER:
        for t in TASKS:
            d = data.get((algo, t))
            if d is None:
                continue
            s = stats(d["episodes"])
            if s["n"] < 50 or PAPER[algo][t] < 5:
                continue
            pr = 100.0 * PAPER[algo][t] / 50
            if s["rate"] < pr / 2 or s["rate"] > pr * 2:
                flags.append(f"**{algo}/{t}**: {s['succ']}/{s['n']} = {s['rate']:.0f}% vs "
                             f"paper {PAPER[algo][t]}/50 = {pr:.0f}%")
    if flags:
        A("\nNot reproducing the paper (factor >2 apart): " + " · ".join(flags) + "\n")

    # ── 2. LSCR ────────────────────────────────────────────────────────────
    A("\n## 2. LSCR — latched stage completion rate\n")
    A("| Algorithm | Task | n | mean | median | min | max | LSCR==0 |")
    A("|---|---|---|---|---|---|---|---|")
    for algo in ORDER:
        for t in TASKS:
            d = data.get((algo, t))
            if d is None:
                A(f"| {algo} | {t} | — | — | — | — | — | — |")
                continue
            s = stats(d["episodes"])
            A(f"| {algo} | {t} | {s['n']} | {f3(s['mean'])} | {f3(s['med'])} | {f3(s['min'])} "
              f"| {f3(s['max'])} | {s['zeros']}/{s['ln']} |")

    # ── 3. progress ────────────────────────────────────────────────────────
    try:
        prev = json.load(open(ETA_STATE))
    except Exception:
        prev = {}
    cur, running = {}, []
    for algo in ORDER:
        for t in TASKS:
            key = f"{algo}|{t}"
            d = data.get((algo, t))
            n = len(d["episodes"]) if d else 0
            cur[key] = [nowts, n]
            if d is None or n >= 50:
                continue
            p = prev.get(key)
            line = f"{algo}/{t} {n}/50"
            if p and nowts - p[0] > 60 and n > p[1]:
                rate = (n - p[1]) / (nowts - p[0])
                line += f" ({rate*60:.2f} ep/min, ~{(50-n)/rate/60:.0f} min left)"
            elif p and nowts - p[0] > 900:
                line += " (stalled?)"
            running.append(line)
    try:
        json.dump(cur, open(ETA_STATE, "w"))
    except OSError:
        pass
    A("\n## 3. Progress\n")
    A("- " + ("\n- ".join(running) if running else "nothing running (all cells complete)"))
    A("")

    # ── 4. deviations ──────────────────────────────────────────────────────
    A("\n## 4. Deviations & caveats\n")
    A("""
- **torch cu128 everywhere.** The official pins (`torch==2.0.0+cu118` DP, `2.5.1` GR00T,
  `jax[cuda12]==0.5.0` pi0.5) have no `sm_120` (Blackwell) kernels. Used torch `2.7.0+cu128` and
  jax **`0.6.2`** -- the newest jax that both ships Blackwell kernels and keeps
  `jax.experimental.layout.DeviceLocalLayout`, which `orbax-checkpoint==0.11.1` imports (>=0.7
  removed it, 0.5.x has no sm_120 kernel).
- **Split server/client.** The repo's `eval_double_env.sh` hard-codes its sim-side conda env as
  `dex2bench`; ours is `bench2dex`, so `70_eval_dp.sh` / `72_eval_gr00t.sh` / `73_eval_pi05.sh`
  re-implement the orchestration with correct env names plus four Isaac fixes: local `libGLU.so.1`,
  the lab proxy, an `omni.kit.registry.nucleus` override for the unreachable CloudFront registry,
  and `CUDA_VISIBLE_DEVICES`.
- **Robot USD: every `*/visuals` prim fails to resolve** (e.g. `Unresolved reference prim path
  @.../multi_iiwa7_with_sharpa_physics.usd@</visuals/left_index_fingertip>`).
  `multi_iiwa7_with_sharpa_base.usd` (21 MB, holds the meshes) references visual prims absent from
  the 19 KB physics-only USD. It appears in ACT and DP logs alike and the asset download matches the
  documented 52,764 files, so it is an upstream asset trait -- but wrist cameras may lack hand
  geometry, a candidate explanation for the ACT/26 shortfall.
- **Run-to-run variance.** Two identical smoke runs (same seed, same checkpoint) gave LSCR 1.00 vs
  0.25 on the same episode. Treat each cell as one sample.
- **GR00T's numbers rest on a reconstructed `experiment_cfg`** (see section 5). If the authors
  finetuned on a different subset or config, their normalization differs and the scores shift.
- Data/weights live on `/mnt/public/datasets/bench2dex/` (NFS, symlinked); only envs and code on `/home`.
""")

    # ── 5. incidents ───────────────────────────────────────────────────────
    A("\n## 5. Incidents & fixes\n")
    A("""
- **GR00T: unrunnable as released, then unblocked.** The released checkpoints hold only 4 files and
  no `experiment_cfg/`, which `policy/GR00T_n15/deploy_policy.py:127` requires and
  `src/gr00t/model/policy.py:110` reads for normalization statistics; the server died in 8 s with
  `FileNotFoundError`. That file is a deterministic function of the training data and the repo
  computes it itself: `Dex2BenchHDF5Dataset.__init__` calls `_compute_statistics()` and builds a
  `DatasetMetadata`, and `src/gr00t/experiment/runner.py:76-97` writes `{tag: metadata}` to
  `<output_dir>/experiment_cfg/metadata.json` inside `TrainRunner.__init__`, before any training step.
  `_deploy_logs/80_rebuild_gr00t_experiment_cfg.py` reproduces the exact `train.sh` arguments
  (hdf5-native, `Dex2BenchGR00TDataConfig`, `new_embodiment`, 4cam map, active-DOF, truncate-at-homing,
  scene `description:` as prompt) and regenerates it in ~1 s per task (26: 62,438 samples,
  32: 55,900, 73: 81,514; state/action dim 54).
  *(An earlier revision of this report wrongly said the repo had no statistics entry point -- it had
  only looked at `utils/experiment.py`, which is merely the checkpoint callback.)*
- **`flash-attn==2.8.2` was missing from my env build** -- declared at
  `policy/GR00T_n15/pyproject.toml:93` and imported unconditionally by the Eagle2 backbone
  (`radio_model.py` calls `replace_vit_attn_with_flash_attn()` at import time). Installing it let the
  GR00T server load. A dependency audit now shows 34/35 declared deps present; the remaining one,
  `eva-decord`, is macOS-only.
- **pi0.5's first launch failed silently.** JAX preallocates 75 % of the GPU by default; three servers
  held 48,430 MiB each, driving GPU 2 down to 179 MiB free. Isaac logged `Skipping NVIDIA GPU due CUDA
  being in bad state` and **exited 0 with zero episodes**. Fixed with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (+ `MEM_FRACTION=0.15`, `ALLOCATOR=platform`): measured
  server footprint 48,430 -> 6,982 MiB. A guard now turns "exit 0 with 0 episodes" into a failure.
- **ModelScope silently under-downloaded two pi0.5 shards**, leaving 512-byte `.incomplete` files
  where 314 KB / 403 KB belonged, and still exited 0; loading then failed with
  `Truncated Zstd-compressed stream`. Both were re-downloaded, and `95_verify_weights.py` now checks
  every checkpoint file-by-file against the ModelScope listing -- latest run 12/12 complete.
- **Two GPU semaphore hangs** (`cudaErrorMisalignedAddress` / `Wait for external semaphore failed`):
  processes stayed alive at ~150 % CPU with GPU utilisation at 0. They hit ACT (44/44/34 of 50) and
  DP task 26 (43/50); both were resumed with `--start-episode` + `--append-output`, losing no
  completed episode.
- **My stall watchdogs were net-harmful and are disabled** (`97_stall_watchdog*.sh.DISABLED_BROKEN`).
  v1 resolved logs via `/proc/<ppid>/fd/1`, which is a pipe under `<script> | tee <log>`, so it
  silently skipped everything. v2 keyed logs by task id only, so the stale `20_eval_26.log` from the
  finished ACT stage matched the live task-26 process of the current stage; it killed all three DP
  clients and, via `kill -9 -$ppid`, took every tmux session with them, including the newly started
  pi0.5 clients. Stalls are now handled by hand.
- **One intermediate commit is misleading by construction:** `ed405b7` was auto-pushed while the JAX
  blow-up had left pi0.5's output dirs empty, so that revision reports pi0.5 as 0/0. Later commits
  supersede it.
""")

    # ── 6. artifacts ───────────────────────────────────────────────────────
    vf = f"{LOGS}/95_weights_verification.txt"
    summ = ""
    if os.path.exists(vf):
        for line in open(vf, errors="ignore"):
            if line.startswith("SUMMARY"):
                summ = line.strip()
    A("\n## 6. Artifacts\n")
    A("- per-episode data: `output/metric/*/per_episode.jsonl`")
    A("- raw logs: `_deploy_logs/` (`20_eval_*` ACT, `70_/76_*` DP, `72_gr00t_*` GR00T, `77_/78_*` pi0.5)")
    A("- repro scripts: `_deploy_logs/{20,21,70,72,73,75,76,77,78,80,95,99}_*`"
      + (f"  ·  {summ}" if summ else ""))
    A("- report source: `_deploy_logs/99_make_report.py` (regenerates this file)\n")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[report] wrote {OUT_MD} at {now:%Y-%m-%d %H:%M:%S}")
    for algo in ORDER:
        row = []
        for t in TASKS:
            d = data.get((algo, t))
            row.append("—" if d is None else f"{stats(d['episodes'])['succ']}/{stats(d['episodes'])['n']}")
        print(f"[report]   {algo:5s} 26={row[0]:>7s} 32={row[1]:>7s} 73={row[2]:>7s}")


if __name__ == "__main__":
    main()
