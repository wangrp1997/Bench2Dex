# Bench2Dex — Sharpa baseline results

`multi_iiwa7_with_sharpa` · tasks 26/32/73 · profile `none` · 50 episodes · seed 100000000 · GPU 2 · generated 2026-09-30 19:07

Paper column = the project page's own per-task numbers (`bench2dex.github.io/assets/js/data.js`), None channel, out of 50.


## 1. Stable success rate (/50)

| Algorithm | 26 Canned-Food | 32 Baking Tray | 73 Jigsaw | paper 26/32/73 |
|---|---|---|---|---|
| **ACT** | **7/50** = 14% | **8/50** = 16% | **0/50** = 0% | 17/9/0 |
| **DP** | **0/50** = 0% | **3/50** = 6% | **0/50** = 0% | 0/2/0 |
| **PI05** | **19/50** = 38% | **19/50** = 38% | **0/50** = 0% | 15/22/0 |
| **GR00T** | **2/5** = 40% (5/50) | **0/2** = 0% (2/50) | **4/20** = 20% (20/50) | 17/12/10 |

Not reproducing the paper (factor >2 apart): **ACT/26**: 7/50 = 14% vs paper 17/50 = 34%


## 2. LSCR — latched stage completion rate

| Algorithm | Task | n | mean | median | min | max | LSCR==0 |
|---|---|---|---|---|---|---|---|
| ACT | 26 | 50 | 0.500 | 0.500 | 0.000 | 1.000 | 11/50 |
| ACT | 32 | 50 | 0.195 | 0.000 | 0.000 | 1.000 | 35/50 |
| ACT | 73 | 50 | 0.070 | 0.000 | 0.000 | 0.750 | 38/50 |
| DP | 26 | 50 | 0.205 | 0.250 | 0.000 | 0.750 | 23/50 |
| DP | 32 | 50 | 0.175 | 0.000 | 0.000 | 1.000 | 33/50 |
| DP | 73 | 50 | 0.170 | 0.000 | 0.000 | 0.750 | 26/50 |
| PI05 | 26 | 50 | 0.600 | 0.750 | 0.000 | 1.000 | 13/50 |
| PI05 | 32 | 50 | 0.415 | 0.000 | 0.000 | 1.000 | 26/50 |
| PI05 | 73 | 50 | 0.355 | 0.250 | 0.000 | 1.000 | 12/50 |
| GR00T | 26 | 5 | 0.950 | 1.000 | 0.750 | 1.000 | 0/5 |
| GR00T | 32 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 2/2 |
| GR00T | 73 | 20 | 0.775 | 0.750 | 0.500 | 1.000 | 0/20 |

## 3. Progress

- GR00T/26 5/50 (0.40 ep/min, ~113 min left)
- GR00T/32 2/50
- GR00T/73 20/50


## 4. Deviations & caveats


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


## 5. Incidents & fixes


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


## 6. Artifacts

- per-episode data: `output/metric/*/per_episode.jsonl`
- raw logs: `_deploy_logs/` (`20_eval_*` ACT, `70_/76_*` DP, `72_gr00t_*` GR00T, `77_/78_*` pi0.5)
- repro scripts: `_deploy_logs/{20,21,70,72,73,75,76,77,78,80,95,99}_*`  ·  SUMMARY: 12/12 checkpoint dirs complete
- report source: `_deploy_logs/99_make_report.py` (regenerates this file)

