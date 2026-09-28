# Bench2Dex — Deployment & Evaluation Record

Machine: `test-ws` (Coder workspace container) · Date: 2026-09-28 · Operator: wangrenpeng

Companion to `DEPLOYMENT_NOTES.md`. That file describes the *intended* path and 13
generic pitfalls; **this file records what actually happened on this machine** and
the six problems that the notes do not cover.

---

## 1. Machine

| Item | Value |
|---|---|
| GPU | **4× NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each** (sm_120) |
| Driver | 590.48.01 (open kernel module), CUDA driver API 13.1 |
| CPU / RAM | 96 cores / 503 GiB |
| Storage | `/home` local btrfs NVMe 7.0 TB (2.2 TB free) · `/mnt/public` shared NFS 31 TB (13 TB free, ~930 MB/s) |
| Container | no root, no `sudo`; `/dev/nvidia*` present; `libGLU.so.1` absent |

The machine is **shared** — all four GPUs are normally busy with other users' jobs.
`nvidia-smi` showed 99 % / 100 % / 0 % / 93 % utilisation at deployment time; only
GPU 2 was idle (66 GB free). All evaluation is pinned to GPU 2 with
`CUDA_VISIBLE_DEVICES=2`.

---

## 2. Final directory layout

Both sides use the same top-level name, `bench2dex`. The large artefacts live on the
shared NFS share and are symlinked into the working tree so the hard-coded relative
asset paths (`../../dex2bench_dataset/...` from `scenes/`, `../dex2bench_dataset/...`
from `robots/`) keep resolving.

```
/home/wangrenpeng/bench2dex/                    # code side (local NVMe)
├── Bench2Dex/                                  # this repo
├── IsaacLab/                                   # Isaac Lab v2.3.2
├── _deploy_logs/                               # every script + log from this deployment
├── _syslibs/                                   # locally extracted libGLU.so.1
├── output/                                     # evaluation results
├── dex2bench_dataset -> /mnt/public/datasets/bench2dex/dex2bench_dataset
├── teleopdata        -> /mnt/public/datasets/bench2dex/teleopdata
└── policy_ckpt       -> /mnt/public/datasets/bench2dex/policy_ckpt

/mnt/public/datasets/bench2dex/                 # data side (shared NFS)
├── dex2bench_dataset/   19 GB   52,764 files
├── teleopdata/          49 GB   300 hdf5 (100 per task)
└── policy_ckpt/        1.9 GB   12 files
```

## 3. Environment

Dedicated conda env `bench2dex` (Python 3.11.15). **No existing ROS / conda / CUDA
environment was modified.**

| Package | Version |
|---|---|
| isaacsim | 5.1.0.0 (`[all,extscache]`) |
| isaaclab | 0.54.2 (from tag `v2.3.2`) |
| torch / torchvision / torchaudio | 2.7.0+cu128 / 0.22.0+cu128 / 2.7.0+cu128 |
| numpy | 1.26.0 |
| pinocchio (`pin`) | 2.7.0 |
| dex-retargeting | pulled via `requirements.txt` |

`$CONDA_PREFIX/etc/conda/activate.d/bench2dex_env.sh`:

```sh
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH
export LD_LIBRARY_PATH=/home/wangrenpeng/bench2dex/_syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

---

## 4. Six problems not covered by `DEPLOYMENT_NOTES.md`

### 4.1 `nvidia-smi` fails with "Failed to initialize NVML: Unknown Error" — it is the *agent sandbox*, not the GPU

Every existing conda env reported `torch.cuda.is_available() == False` and NVML
returned 999. This is **not** a driver or container problem: the agent's file sandbox
(Landlock, `workspace-write`) denies `O_RDWR` on device files outside the workspace,
and NVML/CUDA must open `/dev/nvidia*` read-write.

Proof: `/dev/nvidiactl` opens fine `O_RDONLY` but fails `O_RDWR`; `/etc/hostname`
behaves identically; `/dev/null` `O_RDWR` is the sandbox's single hard-coded exception.

**Fix:** run with the file policy `danger-full-access` (or whitelist `/dev/nvidia*`
and `/dev/shm`). Then all four GPUs appear normally.

### 4.2 Isaac Sim hangs at startup on the online extension registry — and it cannot simply be taken offline

`omni.kit.registry.nucleus` blocks on `ovextensionsprod.blob.core.windows.net` and
`dw290v42wisod.cloudfront.net`; from this network the TCP connection opens and then
stalls (`cwnd:1`, retransmit treadmill).

The naive fix (empty registries) **does not work**, because Isaac Lab 2.3.2 requires an
extension version that Isaac Sim 5.1.0 does not ship:

```
[isaaclab.python.rendering-2.3.2 -> isaaclab.python-2.3.2] dependency:
  'isaacsim.asset.importer.urdf' = { version='=2.4.31' } can't be satisfied.
  Available versions: isaacsim.asset.importer.urdf-2.4.30+107.3.3
```

`/mnt/public` and the lab HTTP proxy *can* reach `blob.core.windows.net`, so the
registry is synced once through the proxy (slow, ~1.85 files/s, but cached in
`~/.local/share/ov/data/exts/v2/`; subsequent runs take ~45 s). Only `kit/community`
(cloudfront) is unreachable even through the proxy and is redirected to a local empty
directory:

```sh
--kit_args "--/exts/omni.kit.registry.nucleus/registries/2/url=/tmp/empty_reg \
            --/exts/omni.kit.registry.nucleus/registries/2/optional=true"
```

### 4.3 `libGLU.so.1` missing → the RTX renderer silently fails

```
[Error] [rtx.neuraylib.plugin] Failed to open .../libneuray.so: libGLU.so.1:
        cannot open shared object file
[Warning] [omni.hydra.rtx] HydraEngine rtx failed creating scene renderer.
```

There is no root here. Fix without sudo:

```sh
cd /tmp && apt-get download libglu1-mesa
dpkg-deb -x /tmp/libglu1-mesa_*.deb ~/bench2dex/_syslibs
export LD_LIBRARY_PATH=~/bench2dex/_syslibs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 4.4 Pitfall 7 is real, but only on numpy **1.26.0**

The released `dataset_stats.pkl` references `numpy._core.multiarray`. `numpy/_core`
first appears in a later 1.26.x — it loads on 1.26.4, but
`requirements.txt` (`numpy>=1.26,<2`) resolves to **1.26.0**, where it fails with
`ModuleNotFoundError: No module named 'numpy._core'` while loading the ACT policy.

All three checkpoints were re-pickled with the round-trip from the notes
(read with numpy 2 → JSON → rewrite with numpy 1.26, protocol 2). Originals kept as
`dataset_stats.pkl.orig_numpy2pickle`; values verified bitwise equal.

### 4.5 `git clone github.com` does not work; use a mirror

`git clone` dies after ~136 s and even `git ls-remote` times out. Git works **through
the lab proxy** (`git -c http.proxy=http://host.docker.internal:1081 ...`), and plain
downloads work through `gh-proxy.com` (~3 MB/s):

```sh
curl -fSL -o /tmp/isaaclab-v2.3.2.tar.gz \
  https://gh-proxy.com/https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.3.2.tar.gz
```

### 4.6 ModelScope CLI moved packages

`uv tool install modelscope` fails with *"No executables are provided by package
`modelscope`"*. The CLI now lives in **`modelscope-hub`**, and the repo selector is
`--repo-type dataset|model` instead of `--dataset` / `--model`.

---

## 5. Verified state

Section 9 checklist: **7/7 PASS** — `nvidia-smi`, `torch 2.7.0+cu128 cuda 12.8 sm (12,0)`
with a real matmul, `import isaacsim` (no EULA prompt), `isaaclab 0.54.2`,
`pinocchio 2.7.0` from this env, `IPython/h5py/cv2/trimesh/dex_retargeting`,
`numpy 1.26.0`.

Smoke test — task 26 / Sharpa / ACT / profile `none`, `--num-episodes 2`
(`--episode-steps 871`, `--anchor-dir .../replay-generalization`):

```
[run_policy] All imports OK.
[init] Robot key: multi_iiwa7_with_sharpa (from --robot-key)
[init] Anchor dir loaded: ... (50 episodes)
[init] Active DOF: full_dof=58 active_dof=54
[ep 1] Camera preflight...  cam_overhead/cam_wrist_right/cam_wrist_left/
                            cam_stereo_left/cam_stereo_right  rgb=(480,640,3)
[ep 1] Episode finished: FAIL  current_stage_completion_rate=0.75
                               latched_stage_completion_rate=1.00
[ep 2] Episode finished: FAIL  current_stage_completion_rate=0.25
                               latched_stage_completion_rate=0.75
```

Non-zero `latched_stage_completion_rate` ⇒ not the Pitfall 12/13 failure mode.
`success=False` is expected: the paper reports 17/50 = 34 % for this cell.

### 5.1 ⚠️ Run-to-run variance (same seed, different result)

The smoke test was repeated after the data migration with the **same seed**
(`100000000`), same checkpoint, same task:

| | episode 1 LSCR | episode 2 LSCR |
|---|---|---|
| run A | **1.00** | 0.75 |
| run B | **0.25** | 0.75 |

So this pipeline is **not deterministic** — consistent with non-deterministic RTX
rendering and GPU atomics, aggravated by a shared GPU. Single-seed single-run numbers
must not be used to claim an improvement; report mean ± spread over seeds/repeats.

---

## 6. Data facts

`replay-generalization` is not just scene-reset anchors — it is the full recorded
dataset. Each of the 300 episodes contains:

* `cameras/` — 6 RGB streams: `cam_chest`, `cam_overhead`, `cam_stereo_left`,
  `cam_stereo_right`, `cam_wrist_left`, `cam_wrist_right`
* `robot/tactile/` — **10 tactile sites**, `{left,right}_{thumb,index,middle,ring,pinky}_elastomer`,
  each with `tacmap`, `contact_mask` and `distance_along_normal_m`
* `robot/{qpos,qvel,qeffort}`, `action/commanded`, `labels/`, `metrics/`, `objects/`
* `meta/modalities == ["object_pose","joint_state","rgb","tactile"]`

**No extra download is needed for visuo-tactile work on tasks 26/32/73.**

---

## 7. Which hand / which tasks (for a visuo-tactile contribution)

Only **three** of the 26 benchmark tasks use `multi_iiwa7_with_sharpa`, and the hand is
the only one with a real tactile implementation in this repo
(`collector/tacmap_sensor/sharpa_tacmap_*.py`), the only one with released tactile
data, and the only one with a full set of released baselines:

| Task | Scene |
|---|---|
| 26 | `26_canned_food_tray_line_arrangement` — canned-food tray line arrangement |
| 32 | `32_baking_tray_prep_with_tools` — baking tray prep with tools |
| 73 | `73_jigsaw_puzzle_assembly` — jigsaw puzzle assembly (most contact-critical) |

Released checkpoints (`Bench2Dex/New_Policy`, per task per algorithm):
`act_active` 0.63 GB · `dp` 1.48 GB · `gr00t_n15` 7.07 GB · `pi05` 11.59 GB.

`run_policy.py --policy-type` accepts only **`ACT` / `DP` / `REMOTE`**; π0.5 and
GR00T N1.5 must be served via `script/policy_model_server.py` from their own conda
envs (`policy/setup_policy_envs.sh`).

**Tactile baselines:** `policy/ACT-Tactile`, `policy/GR00T_n15_Tactile`,
`policy/GR00T_n15_Tactile_Cross`. **No tactile weights have been released** — they
must be trained, which is precisely the room for a visuo-tactile contribution.
Vision-only controls: `policy/ACT`, `DP`, `pi05`, `GR00T_n15`.

---

## 8. How to run

All scripts and logs live in `~/bench2dex/_deploy_logs/`. The working smoke-test
script is `13_smoke.sh`; the full-evaluation script is `20_eval_act.sh`
(`<TASK_ID> <SCENE_STEM> <EPISODE_STEPS> <GPU>`). Both already carry the five
required fixes: `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` for libGLU, the proxy
exports, the `--kit_args` registry override, and the EULA variables.

Per-task rollout budgets, computed with `script/eval_budget.sh`:

| Task | `--episode-steps` | physics steps |
|---|---|---|
| 26 | 871 | 2613 |
| 32 | 852 | 2556 |
| 73 | 1213 | 3639 |

Measured ~2.5 min per episode (ACT, task 26, GPU 2, shared machine); 50 episodes is
roughly 2 h per cell.

> Note: if the agent session is switched back from `danger-full-access` to
> `workspace-write`, writing to `/mnt/public` is denied by the sandbox (reads are
> fine). Evaluation outputs go to `~/bench2dex/output`, so runs are unaffected.
