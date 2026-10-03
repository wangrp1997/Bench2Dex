# Bench2Dex — Sharpa baseline results

`multi_iiwa7_with_sharpa` · tasks 26/32/73 · profile `none` · 50 episodes · seed 100000000 · GPU 2 · generated 2026-10-03 11:04

Paper column = the project page's own per-task numbers (`bench2dex.github.io/assets/js/data.js`), None channel, out of 50.


## 1. Stable success rate (/50)

| Algorithm | 26 Canned-Food | 32 Baking Tray | 73 Jigsaw | paper 26/32/73 |
|---|---|---|---|---|
| **ACT** | **7/50** = 14% | **8/50** = 16% | **0/50** = 0% | 17/9/0 |
| **DP** | **0/50** = 0% | **3/50** = 6% | **0/50** = 0% | 0/2/0 |
| **PI05** | **19/50** = 38% | **19/50** = 38% | **0/50** = 0% | 15/22/0 |
| **GR00T** | **19/50** = 38% | **12/50** = 24% | **8/50** = 16% | 17/12/10 |

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
| GR00T | 26 | 50 | 0.700 | 0.750 | 0.000 | 1.000 | 5/50 |
| GR00T | 32 | 50 | 0.505 | 0.625 | 0.000 | 1.000 | 13/50 |
| GR00T | 73 | 50 | 0.710 | 0.750 | 0.250 | 1.000 | 0/50 |

## 3. Progress

- nothing running (all cells complete)

**Excluded from sections 1-2.** Every run writes `run_meta.json` via `run_policy.py`; only `condition == "baseline"` may enter the tables, and the choice of directory per cell is listed here rather than made silently:
- 21 dir(s) written under a **different experimental condition** (behaviour-changing env switch -- these are results of the intervention study, not baselines):
  - `baseline-recording`: 15 dir(s), e.g. `gr00t_26_0929_2336`
  - `close0.25`: 2 dir(s), e.g. `gr00t_26_0930_1318`
  - `evt`: 1 dir(s), e.g. `gr00t_26_0930_1538`
  - `evtK15`: 1 dir(s), e.g. `gr00t_26_0930_1543`
  - `evtMixed`: 1 dir(s), e.g. `gr00t_26_0930_1550`
  - `unknown`: 1 dir(s), e.g. `gr00t_26_0930_1534`
- 3 **superseded/partial** baseline dir(s) (a complete >= 50-episode run outranks a newer fragment):
  - ACT/26 `26_iiwa7_sharpa_policy_best_none_0928_1948` (n=2)
  - ACT/26 `26_iiwa7_sharpa_policy_best_none_0928_2052` (n=2)
  - GR00T/26 `gr00t_26_0929_1110` (n=1)


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
- **Run-to-run variance sets the resolution limit -- but the worst "outlier" was not variance.**
  Clean task-26 runs with the same checkpoint, seed, scene and protocol give **19/50** and **15/50**
  (plus 17/40 on the shifted episode range 21-60): an 8-10 pp spread, i.e. within the binomial
  standard error of n=50 (~7 pp). So differences below ~15 pp are not resolvable here: read the
  table as "reproduces the paper's ordering", not as a ranking.
  The **0/35** run that first looked like a 38 pp variance blow-up was a *different experimental
  condition*: it ran with `B2D_EXTRA_CLOSE=0.25` (`run_policy.py:1297`), the **unconditional
  extra-closure intervention arm**, which adds +0.25 rad to 28 hand flexion joints. It printed
  `[intervention] B2D_EXTRA_CLOSE=0.25 applied to 28 hand flexion joints` in its log, and
  `LSCR=0.0` on all 35 episodes is the intervention working as designed -- the hand is clamped
  shut, so no stage can complete. Not a bad policy, not harness noise, and (see section 5) it was
  invisible in `output/metric/` because the directory name and `per_episode.jsonl` carry no record
  of the condition. Earlier smoke runs also gave LSCR 1.00 vs 0.25 on the same episode.
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
- **`output/metric/gr00t_*` is not one experiment.** The same namespace is written by three
  different things: baseline evals (`72_eval_gr00t.sh`), rollout recordings
  (`99_eval_gr00t_record.sh`), and **intervention pilots that set a behaviour-changing env var**
  (`B2D_EXTRA_CLOSE`, `B2D_EVENT_REPLAN`). Neither the directory name nor `per_episode.jsonl`
  records which one it was -- the only trace is an env-var-dependent line in the stdout log.
  Known intervention dirs: `gr00t_26_0930_1318` and `gr00t_26_0930_1322`
  (`B2D_EXTRA_CLOSE=0.25`); `gr00t_26_0930_1538`, `gr00t_26_0930_1543`, `gr00t_26_0930_1550`
  (`B2D_EVENT_REPLAN`). `gr00t_26_0930_1322` is the 0/35 run: it is the *unconditional
  extra-closure* arm of the intervention study (`run_policy.py:1290-1314` -- "Unset ->
  byte-identical behaviour to the stock baseline"), and its `LSCR=0.0` on every episode is the
  intervention doing its job, not a broken policy.
  **Fix applied (2026-10-03).** Provenance now lives in the data, not in log archaeology:
  `run_policy.py::experiment_condition()` appends the condition to the output directory
  (`_close025`, `_evtK6`, ...; stock runs keep their exact old names) and `write_run_meta()`
  writes `<dir>/run_meta.json` once per directory. It is done there rather than in the eval
  shell scripts precisely because the offending pilot was launched by hand, so a script-level
  fix would not have caught it. The four eval scripts source `_deploy_logs/00_condition_tag.sh`
  so their own `$OUT` (and therefore `$OUT/rollouts`) still matches the directory actually
  written; that shell rule is unit-checked to produce byte-identical suffixes to the Python one.
  `99_make_report.py` now admits a directory **only** when `run_meta.json` says
  `condition == "baseline"`, and refuses an untagged dir outright -- that single rule is what
  stops this bug recurring, because the 0/35 dir would simply not be eligible.
  Historical dirs were backfilled with retro-inferred metadata
  (`"provenance": "retro-inferred 2026-10-03"`, with the evidence recorded), so the strict rule
  is decidable for old data too.
- **The report itself reported a false GR00T baseline for four days.** `99_eval_gr00t_record.sh`
  (rollout recording for the failure-prediction study) writes into the *same*
  `output/metric/gr00t_<task>_<mmdd_HHMM>` namespace as the baseline evals and leaves a
  `<dir>/rollouts/` subdir. `collect()` used to take the newest non-empty dir per cell, so from
  2026-09-30 it reported a crashed 7-episode fragment as GR00T's result (**1/7, 0/3, 0/5**) instead
  of the real complete runs (**19/50, 12/50, 8/50**, matching the paper's 17/12/10).
  Fixed: recording dirs are excluded, a complete >= 50-episode run outranks a newer fragment, and
  every discarded directory is listed in section 3. *Lesson: never key the report on mtime alone.*
- **Rollout-dataset provenance audited (clean).** An intervention run writes full-length episodes
  that are indistinguishable, in the data, from genuine baseline failures -- so a failure-predictor
  trained on them would be learning the intervention, not the task. The 152-episode tactile+torque
  dataset was therefore checked against every run by hashing `robot/qpos`: all 152 files trace to
  the known-good runs (26: 43+7+2, 32: 46+1+3, 73: 15+15+15+5), with **zero** files from any
  intervention run (1322/1534/1538/1543/1550 all contribute 0), and no duplicate episode indices.
  (Replay preserves `qpos` bit-exactly, so the hash is a valid provenance fingerprint.)
  Note the flip side: recording files are keyed by episode index only, so a later run silently
  overwrites an earlier one at the same index -- provenance must be audited, never assumed.
- **One intermediate commit is misleading by construction:** `ed405b7` was auto-pushed while the JAX
  blow-up had left pi0.5's output dirs empty, so that revision reports pi0.5 as 0/0. Later commits
  supersede it.


## 6. Artifacts

- per-episode data: `output/metric/*/per_episode.jsonl`
- run provenance: `output/metric/*/run_meta.json` (written by `run_policy.py`; the report admits a directory only when `condition == "baseline"`)
- raw logs: `_deploy_logs/` (`20_eval_*` ACT, `70_/76_*` DP, `72_gr00t_*` GR00T, `77_/78_*` pi0.5)
- repro scripts: `_deploy_logs/{00,20,21,70,72,73,75,76,77,78,80,95,99}_*`  ·  SUMMARY: 12/12 checkpoint dirs complete
- report source: `_deploy_logs/99_make_report.py` (regenerates this file)

