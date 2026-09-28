# Bench2Dex — Deployment Notes, Pitfalls & Verified Setup

> **Audience:** an automated agent (or a human) deploying Bench2Dex on a fresh
> Ubuntu + NVIDIA machine, likely behind a slow/restricted international network.
>
> **Status:** every command and fix below was executed and verified end-to-end on
> the reference machine. The benchmark produces a valid, paper-comparable score at
> the end of this document.
>
> **Repo:** https://github.com/Bench2Dex/Bench2Dex (MIT licensed)
> **Paper:** https://arxiv.org/abs/2609.15726

---

## 0. TL;DR — the fast path

```bash
# 1) Isolated conda env (NEVER reuse an existing Isaac Lab env)
conda create -n bench2dex python=3.11 -y
conda activate bench2dex

# 2) Isolation fixes — MUST be done first (see Pitfall 3 & 4)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/bench2dex_env.sh <<'EOF'
export ACCEPT_EULA=Y                  # Isaac Sim EULA (non-interactive)
export OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH                      # ROS 2 py3.12 packages poison py3.11
EOF

# 3) Fast, parallel installer (see Pitfall 1) — do NOT use plain pip here
pip install -i https://mirrors.aliyun.com/pypi/simple/ uv
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=180
UV="uv pip install --python $CONDA_PREFIX/bin/python"

# 4) Isaac Sim 5.1.0 — must come from pypi.nvidia.com (huge; parallelism is essential)
$UV --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.nvidia.com \
    "isaacsim[all,extscache]==5.1.0"

# 5) PyTorch — MUST be the cu128 build (see Pitfall 8 — RTX 50-series)
$UV --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
    "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128"

# 6) Build tools (see Pitfall 5)
$UV --index-url https://mirrors.aliyun.com/pypi/simple/ "setuptools<81" wheel

# 7) Isaac Lab v2.3.2 (must match — see Pitfall 9)
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/source/isaaclab
$UV --index-url https://mirrors.aliyun.com/pypi/simple/ --no-build-isolation -e .
cd -

# 8) Bench2Dex requirements — remove the bogus `pinocchio` pin first (Pitfall 6)
grep -vE '^\s*pinocchio' requirements.txt > /tmp/req.txt
$UV --index-url https://mirrors.aliyun.com/pypi/simple/ -r /tmp/req.txt ipython

# 9) Data (see Section 6 — the full dataset is ~1 TB, you only need a slice)
# 10) Run — see Section 7
```

---

## 1. Verified reference environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.3 LTS, kernel 6.14.0-33-generic |
| GPU | NVIDIA RTX 5080, 16303 MiB (sm_120 / Blackwell) |
| Driver | 580.95.05 (`nvidia-driver-580-open`) |
| CPU / RAM | Intel i9-14900K (32 logical) / 62 GB |
| Python (env) | 3.11.16 (conda) |
| isaacsim | **5.1.0.0** |
| isaaclab | **0.54.2** (from `IsaacLab` tag `v2.3.2`) |
| torch / torchvision / torchaudio | **2.7.0+cu128** / 0.22.0+cu128 / 2.7.0+cu128 |
| numpy | **1.26.0** (must stay `<2`) |
| pin (Pinocchio) | 2.7.0 |
| dex-retargeting | 0.4.6 |
| usd-core | 26.8 |

**Measured VRAM during ACT evaluation:** ~12.5 GB of 16 GB total
(≈10 GB Isaac Sim + ≈2.2 GB ACT model). This is the hard constraint on this
class of machine — see Section 5.

---

## 2. Required directory layout

The scene YAMLs reference assets as `../../dex2bench_dataset/...`, so the layout
is **not optional**:

```
<root>/
├── Bench2Dex/              # this repo
├── dex2bench_dataset/      # assets  (~19 GB)
├── teleopdata/             # teleoperation dataset (huge — download selectively)
└── policy_ckpt/            # pretrained checkpoints (huge — download selectively)
```

Example: `<root> = ~/Projects`.

---

## 3. Network strategy (critical if the machine is in China)

Measured on the reference machine:

| Path | Speed |
|---|---|
| Domestic mirror (Tsinghua/Aliyun) **direct** | **~10 MB/s** |
| International, single connection, via proxy | **~15 KB/s** |
| International, **32 parallel connections**, via proxy | **~40 MB/s** |

### Rules

1. **Domestic sources must NOT go through the proxy.** ModelScope downloads hang
   indefinitely when routed through an overseas proxy. Run them as:
   ```bash
   env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
       modelscope download ...
   ```
2. **International sources must be parallelised.** The bottleneck is
   *per-connection*, not total bandwidth. Use `uv` (never plain `pip` for the big
   installs) with `UV_CONCURRENT_DOWNLOADS=32`.
3. **Prefer mirrors for PyTorch:** `https://mirrors.aliyun.com/pytorch-wheels/cu128/`
   or `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/` serve the `+cu128`
   wheels at ~12–17 MB/s.

---

## 4. Pitfalls — symptom → cause → fix

### Pitfall 1 — `pip` downloads at ~15 KB/s (10 GB ≈ 7 days)

* **Symptom:** `pip install "isaacsim[all,extscache]==5.1.0"` appears to hang;
  `/proc/<pid>/io` shows ~59 KB/s.
* **Cause:** the proxy throttles **per connection**. `pip` downloads serially.
* **Fix:** use `uv` with high concurrency. Measured: 8 connections ≈ 123 KB/s,
  32 connections ≈ 40 MB/s (near-linear scaling).
  ```bash
  pip install -i https://mirrors.aliyun.com/pypi/simple/ uv
  export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=180
  uv pip install --python $CONDA_PREFIX/bin/python ...
  ```

### Pitfall 2 — ModelScope download hangs forever

* **Symptom:** `modelscope download` prints nothing, 0 bytes on disk for minutes.
* **Cause:** `http_proxy`/`https_proxy` in the environment route a **domestic**
  service through an overseas node.
* **Fix:** unset proxy vars for ModelScope (see Section 3, rule 1).
* **Bonus:** ModelScope's CLI defaults to low concurrency. Always pass
  `--max-workers 32`. Measured: **502 → 6161 files/min (12×)**.

### Pitfall 3 — Isaac Sim blocks on an interactive EULA prompt

* **Symptom:** `Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit
  kernel: EOF when reading a line` — `import isaacsim` fails.
* **Cause:** non-interactive shells cannot answer the prompt.
* **Fix:** export both variables **before** any import:
  ```bash
  export ACCEPT_EULA=Y
  export OMNI_KIT_ACCEPT_EULA=YES
  ```
  Recommended: put them in `$CONDA_PREFIX/etc/conda/activate.d/*.sh` so they apply
  **only to this env** (see Pitfall 4 for the same file).

### Pitfall 4 — ROS 2 `PYTHONPATH` poisons the conda env

* **Symptom:** `import pinocchio` resolves to
  `/opt/ros/jazzy/lib/python3.12/site-packages/pinocchio/__init__.py` and raises
  `ModuleNotFoundError: No module named 'pinocchio.pinocchio_pywrap_default'`.
* **Cause:** a global `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`
  (python3.12) is prepended to the conda env's python3.11 `sys.path`.
* **Fix:** unset `PYTHONPATH` on env activation — **do not** edit the global shell
  config, other tooling may depend on it:
  ```bash
  cat > $CONDA_PREFIX/etc/conda/activate.d/bench2dex_env.sh <<'EOF'
  export ACCEPT_EULA=Y
  export OMNI_KIT_ACCEPT_EULA=YES
  unset PYTHONPATH
  EOF
  ```
* **General rule:** never `source /opt/ros/*/setup.bash` inside a conda env, and
  never `conda activate` inside a sourced ROS terminal.

### Pitfall 5 — `setuptools>=81` removed `pkg_resources`; `flatdict==4.0.1` has no wheel

* **Symptom:** installing Isaac Lab editable fails with
  `ModuleNotFoundError: No module named 'pkg_resources'`, hinting at `flatdict`.
* **Cause:** Isaac Lab pins `flatdict==4.0.1`, which ships **only an sdist**
  (its `setup.py` needs `pkg_resources`), while setuptools ≥81 no longer provides
  `pkg_resources`.
* **Fix:**
  ```bash
  uv pip install --python $CONDA_PREFIX/bin/python "setuptools<81" wheel
  cd IsaacLab/source/isaaclab
  uv pip install --python $CONDA_PREFIX/bin/python --no-build-isolation -e .
  ```
  (`--no-build-isolation` is required so the build sees the downgraded setuptools.)

### Pitfall 6 — PyPI `pinocchio` is the WRONG package

* **Symptom:** `No solution found ... only pinocchio<=0.4.3 is available and you
  require pinocchio>=2.7`.
* **Cause:** `requirements.txt` pins `pinocchio>=2.7`, but the PyPI project named
  `pinocchio` is an unrelated HTTP-API client. The real Pinocchio robotics library
  is published as **`pin`** (import name is still `pinocchio`).
* **Fix:** drop the `pinocchio` line and let `dex-retargeting` pull `pin` in:
  ```bash
  grep -vE '^\s*pinocchio' requirements.txt > /tmp/req.txt
  uv pip install ... -r /tmp/req.txt
  # verify: python -c "import pinocchio; print(pinocchio.__version__)"  -> 2.7.0
  ```

### Pitfall 7 — `dataset_stats.pkl` was pickled with numpy 2.x

* **Symptom:** `ModuleNotFoundError: No module named 'numpy._core'` while loading
  the ACT checkpoint's `dataset_stats.pkl`.
* **Cause:** the released pickle references `numpy._core` (numpy ≥2), but
  `requirements.txt` pins `numpy>=1.26,<2`. Their own release is inconsistent.
* **Fix:** re-pickle with numpy 1.26. The file is a flat dict of four `(54,)`
  float32 arrays, so a lossless round-trip through JSON is safe:
  ```python
  # step 1 — read with a numpy 2 environment (any env that has numpy>=2)
  import pickle, json
  d = pickle.load(open(CKPT_DIR + "/dataset_stats.pkl", "rb"))
  json.dump({k: v.tolist() for k, v in d.items()}, open("/tmp/stats.json", "w"))

  # step 2 — rewrite with numpy 1.26 (the bench2dex env)
  import json, pickle, numpy as np
  d = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(open("/tmp/stats.json")).items()}
  pickle.dump(d, open(CKPT_DIR + "/dataset_stats.pkl", "wb"), protocol=2)
  ```
  Keys: `action_mean`, `action_std`, `qpos_mean`, `qpos_std`. **Back up the
  original first.** Values are compared element-wise in float32, so the conversion
  is exact.

### Pitfall 8 — `CUDA error: no kernel image is available` on RTX 50-series ⚠️

* **Symptom:** the scene loads, then the first CUDA op fails:
  ```
  RuntimeError: CUDA error: no kernel image is available for execution on the device
    ... in isaaclab/utils/math.py normalize() -> x / x.norm(...)
  ```
* **Cause:** **PyPI's default `torch==2.7.0` wheel is `+cu126`**, which has no
  kernels for Blackwell consumer GPUs (**sm_120**). Only `+cu128` does.
  This is the single most likely failure on any RTX 50-series card.
* **Fix:** install the cu128 build from a mirror (fast) rather than
  `download.pytorch.org` (slow):
  ```bash
  uv pip install --python $CONDA_PREFIX/bin/python \
      --index-url https://mirrors.aliyun.com/pypi/simple/ \
      --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
      "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128"
  ```
* **Verify before running anything else:**
  ```bash
  python -c "
  import torch
  print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))
  a = torch.randn(2048, 2048, device='cuda'); print((a @ a).sum().item())
  print(torch.randn(1000, device='cuda').norm().item())"
  ```
  Expect `2.7.0+cu128 12.8 (12, 0)` and two finite numbers.

### Pitfall 9 — Isaac Lab version must be exactly `v2.3.2`

* Isaac Lab 2.x ↔ Isaac Sim 5.x and Isaac Lab 3.x ↔ Isaac Sim 6.x are
  **breaking-change incompatible**. Do not "reuse" an existing Isaac Lab 3.0 env,
  and do not let the clone default to `main`.
  ```bash
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git
  ```
* **Also:** `isaacsim` on pypi.nvidia.com is Python-version specific —
  `5.1.0.0` exists as `cp311`; the `6.x` line is `cp312`. Hence Python **3.11**.

### Pitfall 10 — missing `IPython` breaks the ACT policy import chain

* **Symptom:**
  ```
  ModuleNotFoundError: No module named 'IPython'
  ...
  ImportError: attempted relative import with no known parent package
  ```
  (the second error is a *knock-on effect* — `act_policy.py` has
  `try: from detr.main ... except: from .detr.main ...`, and the first import
  fails only because `IPython` is missing).
* **Cause:** ACT's `detr_vae.py` imports `IPython` without declaring it.
* **Fix:** `uv pip install ... ipython` — do **not** start patching imports.

### Pitfall 11 — `run_policy.py` argument gotchas

| Wrong | Right | Note |
|---|---|---|
| `--temporal-agg true` | `--temporal-agg` | `store_true` flag — passing a value yields `unrecognized arguments: true` |
| `--use_active_dof true` | *(omit)* or `--active-dof` | `--use_active_dof` belongs to `deploy_policy.py`, not `run_policy.py`. `--active-dof` is a `BooleanOptionalAction` defaulting to **True** |
| `--state-dim 36` | *(omit)* | auto-derived from `--robot-key` (Sharpa active = **54**) |

### Pitfall 12 — wrong episode budget ⇒ near-zero score ⚠️

* **Symptom:** success rate ≈ 0 % and `latched_stage_completion_rate` mostly 0.
* **Cause:** `--episode-steps` defaults to **400** (≈20 s), but long-horizon tasks
  need far more. The official budget is computed per task:
  ```bash
  source script/eval_budget.sh
  eval_budget_compute "26_canned_food_tray_line_arrangement" "" ""
  echo $EVAL_EPISODE_STEPS      # -> 871  (≈43 s)
  ```
* **Fix:** always pass the computed `--episode-steps`.

### Pitfall 13 — wrong anchor directory ⇒ near-zero score ⚠️

* **Symptom:** same as Pitfall 12 — 0 % success even with the right step budget.
* **Cause:** `script/auto_eval.sh` line 89 defines the anchor as
  ```
  ANCHOR_DIR="../teleopdata/dataset/${TASK}/replay-generalization"
  ```
  Using `origin-generalization` (raw teleop, joint states only) does **not**
  reproduce the scene initial state, so the policy runs out of distribution.
* **Fix:** download and pass **`replay-generalization`**.
* **Related:** `--generalization-profile none` (and `cov_only` / `inv_only`)
  **require** an anchor; only `inv_cov` can run without one. The channel mapping is:
  `none → None`, `cov_only → Equi.`, `inv_only → Inv.`, `inv_cov → Full`.

---

## 5. Hardware limits — read before promising results

Measured during ACT evaluation on a 16 GB card:

| Component | VRAM |
|---|---|
| Isaac Sim (headless, 5 cameras, 640×480) | ≈10 GB |
| ACT policy | ≈2.2 GB |
| **Total** | **≈12.5 GB / 16 GB** |

Per-policy checkpoint sizes (identical for every task/embodiment):

| Policy | Checkpoint dir size |
|---|---|
| ACT (`act_active`) | 641 MB |
| DP (`dp`) | 1.5 GB |
| GR00T N1.5 (`gr00t_n15`) | 7.2 GB |
| π0.5 (`pi05`) | 11.9 GB |

**Implication:** ACT and DP fit comfortably alongside the simulator.
**GR00T N1.5 and π0.5 will most likely OOM** on a 16 GB card. Options:

1. Run them on a bigger GPU (recommended), **or**
2. Use Bench2Dex's built-in **client–server split**, which is exactly what it is
   designed for:
   ```bash
   # on the big-GPU machine
   python script/policy_model_server.py --host 0.0.0.0 --port 9000 \
       --config policy/ACT/deploy_policy.yml --overrides ...
   # on the simulation machine
   python run_policy.py --policy-type REMOTE --remote-host <ip> --remote-port 9000 ...
   ```
3. `--rendering_mode performance` to shave simulator VRAM.

**Timing reference:** ~1.5 min per episode (ACT, 871 steps). 50 episodes ≈ 75 min
per task/algorithm/channel cell.

---

## 6. Data: sizes and selective downloading

| Resource | Size | Notes |
|---|---|---|
| **Assets** (`dex2bench_dataset`) | **19 GB** (52,764 files) | required for every task |
| **teleopdata** | **440 GB** total | ~16 GB per task for `replay-generalization` |
| **policy_ckpt** | **541 GB** total | 1056 checkpoints; one is ~0.6–12 GB |

**Do not download everything** — total ≈ 1 TB. Download per cell:

```bash
# Assets (required, all files)
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
modelscope download --dataset Bench2Dex/Bench2Dex \
    --local_dir <root>/dex2bench_dataset --max-workers 32

# One task's anchor data only (~16 GB)
env -u http_proxy ... modelscope download --dataset Bench2Dex/teleopdata \
    --local_dir <root>/teleopdata \
    --include "dataset/26_canned_food_tray_line_arrangement/replay-generalization/*" \
    --max-workers 16

# One checkpoint only (example: ACT, task 26, Sharpa)
env -u http_proxy ... modelscope download --model Bench2Dex/New_Policy \
    --local_dir <root>/policy_ckpt \
    --include "26/multi_iiwa7_with_sharpa/act_active/*" \
    --max-workers 16
```

HuggingFace mirrors of the same content exist (`Bench2Dex/Assets`,
`Bench2Dex/teleopdata`, `Bench2Dex/policy_ckpt`), but ModelScope is much faster
from China.

### Recommended scope (see Section 8)

Only **three** tasks use the **Sharpa** hand — the same hand family as the IROS
2026 Robotic Origami Challenge rig:

```
tasks: 26 (Canned Food Tray Arrangement), 32 (Baking Tray Prep), 73 (Jigsaw Puzzle Assembly)
robot: multi_iiwa7_with_sharpa   (54 active DoF / 58 full DoF)
```

Scan `robots/` for all 12 supported embodiments, and `robots/active_dof_maps.yml`
for the active/full DoF of each.

---

## 7. Verified run command

```bash
conda activate bench2dex
cd <root>/Bench2Dex

python run_policy.py \
    --policy-type ACT \
    --task scenes/26_canned_food_tray_line_arrangement.yaml \
    --ckpt-dir <root>/policy_ckpt/26/multi_iiwa7_with_sharpa/act_active \
    --ckpt-name policy_best.ckpt \
    --robot-key multi_iiwa7_with_sharpa \
    --enable-rgb \
    --temporal-agg --temporal-agg-k 0.2 \
    --episode-steps 871 \
    --warmup-steps 60 \
    --num-episodes 50 \
    --seed 100000000 \
    --generalization-profile none \
    --anchor-dir <root>/teleopdata/dataset/26_canned_food_tray_line_arrangement/replay-generalization \
    --headless
```

### Expected output (proof it is healthy)

```
[INFO][AppLauncher]: Using device: cuda:0
[run_policy] All imports OK.
[init] Robot key: multi_iiwa7_with_sharpa (auto-detected from ckpt_dir)
[init] Cameras: ['cam_overhead','cam_wrist_right','cam_wrist_left','cam_stereo_left','cam_stereo_right']
[ep 1] Camera preflight...
  [camera] cam_overhead: rgb=(480, 640, 3) ...
[ep 1] Running policy for up to 2613 steps (policy @ every 3 phys steps)...
[ep 1] Episode finished: ... latched_stage_completion_rate=1.00 ...
  [STATS] Success rate: N/M = X%
```

Results are appended to:
```
<root>/output/metric/<task>_<robot>_<ckpt>_<profile>_<timestamp>/per_episode.jsonl
```

### Paper reference numbers (task 26, Sharpa, ACT, counts out of 50)

```python
# from the project page data (channels = [None, Equi., Inv., Full])
"26": { ACT:[17, 2, 0, 0], DPC:[0,0,0,0], PI05:[15,4,2,3], GR00T:[17,5,3,0] }
```
| Channel | ACT success |
|---|---|
| **None** | **17/50 = 34 %** |
| Equi. | 2/50 = 4 % |
| Inv. | 0/50 = 0 % |
| Full | 0/50 = 0 % |

> Sanity check: with the **wrong** anchor/step budget the observed rate was 0/12.
> After fixing Pitfalls 12 & 13 it became 2/9 ≈ 22 % on the None channel and
> LSCR values of 0.5–1.0 — consistent with the paper. **If you see a flat 0 %, you
> have one of those two bugs — not a research result.**

---

## 8. Isolation, hygiene and housekeeping

* **Never modify an existing Isaac Lab / ROS environment.** Create a dedicated
  conda env. Two envs can coexist safely; they only share disk and the pip/uv
  *caches*.
* Keep all settings in `$CONDA_PREFIX/etc/conda/activate.d/` so they are scoped to
  this env and do not affect the rest of the machine.
* Disk budget for the recommended slice: assets 19 GB + 3 × 16 GB anchors +
  3 × ~0.64 GB ACT checkpoints ≈ **70 GB**.
* If you aborted a big `teleopdata` download, the partial per-task directories are
  safe to delete — they are useless unless complete.

---

## 9. Validation checklist

Run these in order; each one isolates a class of failure.

| # | Command | Expected |
|---|---|---|
| 1 | `nvidia-smi` | GPU table, driver 580+ |
| 2 | `python -c "import torch;print(torch.__version__,torch.version.cuda,(torch.randn(999,device='cuda')@torch.ones(999,device='cuda')).item())"` | `2.7.0+cu128 12.8 <finite>` (Pitfall 8) |
| 3 | `python -c "import isaacsim"` | no EULA prompt (Pitfall 3) |
| 4 | `python -c "import isaaclab;print(isaaclab.__version__)"` | `0.54.2` |
| 5 | `python -c "import pinocchio;print(pinocchio.__version__, pinocchio.__file__)"` | `2.7.0`, path under **this** env (Pitfall 4) |
| 6 | `python -c "import IPython, h5py, cv2, trimesh, dex_retargeting; print('ok')"` | `ok` |
| 7 | `ls dex2bench_dataset/` and `ls teleopdata/dataset/<task>/replay-generalization/` | assets + `episode_*.hdf5` present (Pitfall 13) |
| 8 | 2-episode smoke run with the Section 7 command (`--num-episodes 2`) | at least one episode with `latched_stage_completion_rate >= 0.25`; **a flat 0.00 everywhere means something is wrong** |

---

## 10. Pointers

* Repo: https://github.com/Bench2Dex/Bench2Dex
* Paper: https://arxiv.org/abs/2609.15726
* Project page (per-task numbers, task catalogue): https://bench2dex.github.io/
* Docs: https://bench2dex.github.io/doc/
* WeChat community group: QR code in the repo README
* Datasets: https://modelscope.cn/datasets/Bench2Dex · https://huggingface.co/Bench2Dex

> **Licence note:** the **code is MIT** (commercial use fine), but the **arXiv
> paper** is `CC BY-NC-ND 4.0`. Check the `LICENSE`/release notes of the *datasets
> and checkpoints* before any commercial use.
