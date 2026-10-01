# Tactile / contact-rich manipulation enablers newly available 2025–2026
Compiled 2026-09-30. "[arXiv 26xx]" = 2026 preprint. Licence/weights marked "verify" = not confirmed in this pass.

## 1a. Simulators / platforms with tactile support
| item | released | licence | scale / HW | tactile modalities | what it enables | what is missing |
|---|---|---|---|---|---|---|
| **Genesis World tactile module** ("Tactile Genesis", [arXiv 2606.22332](https://arxiv.org/abs/2606.22332), [docs](https://genesis-world.readthedocs.io/en/latest/user_guide/sensing/tactile.html), [project](https://neuroagents-lab.github.io/tactile-genesis/)) | shipped in Genesis World v1.4.x (2026); paper 2026 | Apache-2.0 | >16,384 envs, >1,000 taxels/hand, >600k env-steps/s on ONE RTX 5090; 3–20× prior tactile sims | contact bits, contact depth, 3/6-axis per-taxel kinematic F/T, elastomer marker displacement (3D), proximity F/T, surface distance, temperature grid; imperfections: drift, hysteresis, dead taxels, crosstalk, gain, delay | **Q1 answer: yes** — open-source, GPU-parallel, multi-hand (Wuji, Sharpa, Allegro) tactile RL at scale; sensor abstraction is a swappable experimental variable | readouts are geometric, uncalibrated; contact-audio prototype NOT in v1.4.1; no pretrained tactile policies released (teacher–student scripts only); public code link is anonymous/double-blind |
| Taccel ([arXiv 2504.12908](https://arxiv.org/abs/2504.12908), v2 9/2025) | 2025 | paper CC BY 4.0; **code licence verify** | IPC+ABD, GPU, 18× faster than real time over thousands of envs (H100 80 GB used) | vision-based tactile (VBTS) images + force, sim-to-real grasping/articulated objects | first GPU-parallel accurate VBTS sim validated sim-to-real | VBTS only (no taxel arrays); hand support unclear; repo status unverified |
| TacSL in Isaac Lab ([docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html); Akinola et al., T-RO 2025) | 2025 | Isaac Lab BSD-3 + proprietary Isaac Sim | GPU-parallel 1000s envs, RTX GPU; needs SDF meshes + SDFView | tactile RGB, tactile depth, normal + shear force fields (Taxim-style), GELSIGHT_MINI/R1.5 cfgs | visuo-tactile obs in same PPO/IL stack as vision | vendor lock-in; objects must be declared before sim (SDFView); demo sensor R1.5 discontinued |
| TacEx ([GitHub](https://github.com/DH-Ng/TacEx), [arXiv 2411.04776](https://arxiv.org/abs/2411.04776)) | 2024/25 | extension; paper CC BY 4.0 | Isaac Sim/Isaac Lab | GelSight-style soft-body + visuotactile | bridging soft-body (FEM) and visuotactile sim | Isaac Sim dependency |
| Tac2Real (ECCV 2026, [note](https://papernotes.org/ECCV2026/robotics/tac2real_reliable_and_gpu_visuotactile_simulation_for_online_reinforcement_learn/)) | 2026 | verify | GPU visuotactile, claims online RL + zero-shot real | visuotactile | faster sim-to-real loop for online RL | licence/HW unverified |
| TaCauchy ([arXiv 2606.20426](https://arxiv.org/abs/2606.20426)) | 2026 | verify | FEM framework | vision-based tactile (FEM-accurate) | extensible FEM sensor modelling | not GPU-parallel |
| MuJoCo tactile stacks: [dexrobot_mujoco](https://github.com/flyingGH/dexrobot_mujoco), [smolvla-tactile-dexterous-mujoco](https://github.com/flyingGH/smolvla-tactile-dexterous-mujoco) | 2026 | verify | MuJoCo, single-GPU | fingertip tactile + dexterous hand, SmolVLA | cheap entry point for tactile VLA experiments | MuJoCo has no native tactile physics; research-grade code |

## 1b. Datasets / benchmarks 2025–2026
| item | released | licence | scale | modalities | what it enables | what is missing |
|---|---|---|---|---|---|---|
| REBOOT ([arXiv 2609.22591](https://arxiv.org/abs/2609.22591)) | 9/2026 | CC BY-NC-SA 4.0 | 2,160 demos, 18 precision-assembly tasks, 5 phases, ~half failure+expert recovery, phase-level labels | RGB-D ×4 views, language | **failure-as-first-class** supervision; exposes failure modes hidden by binary success | **no tactile channel**; NC licence |
| FreeTacMan ([arXiv 2506.01941](https://arxiv.org/abs/2506.01941); HF `yxma/gelsight-mini-pretrain-nc`, `yxma/React`) | 2025 | verify | robot-free hand-held GelSight Mini collection | visuo-tactile images + actions | real tactile data without owning a robot arm | rig-dependent; not a policy benchmark |
| SoftVTBench ([arXiv 2607.04234](https://arxiv.org/pdf/2607.04234v1)) | 2026 | verify | deformable-object tasks | visuo-tactile, safety-violation metrics | safety-constrained contact-rich evaluation | failure-episode content unverified |
| Sparsh ([PMLR v270](https://pmlr.com.cn/v270/higuera25a.html)) | 2025 | verify | self-supervised touch representation corpus | vision-based tactile | reusable touch encoder (touch-DINO analogue) | VBTS sensors only |
| Tactile Genesis sim data ([HF `jalal06/tactile-genesis-heavy-mg-v2`](https://huggingface.co/datasets/jalal06/tactile-genesis-heavy-mg-v2)) | 2026 | verify | simulated taxel episodes | taxel/elastomer | large cheap pretraining corpus | sim-only |

## 1c. Hardware purchasable now
| item | released | price / openness | scale | what it enables | what is missing |
|---|---|---|---|---|---|
| **PaXini PX-6AX GEN3** ([specs](https://www.roboticscenter.ai/hardware/paxini-gen3)) | GEN3, 2026 | quote only (lease option) | fingertip tile 13 mm dia: **6-axis F/T (Fx..Tz) + 2D taxel array, ~1.5 mm pitch, 0–50 N, 200 Hz**; up to 10 tiles/hub; ROS2 driver | **Q3 answer: yes** — co-located 6-axis force AND dense array at the same fingertip; mountable on Orca Hand fingertips | no hand ships with it pre-integrated; price undisclosed; Windows-only PXSR config tool |
| Sharpa Wave ([store](https://www.roboticscenter.ai/store/product/sharpa-wave-right)) | 2025/26 | **$50,000**, in stock | human-scale 22-DoF, >1,000 tactile px per fingertip (vision-based Dynamic Tactile Array), 0.005 N sensitivity, 0–30 N, 180 Hz | whole-hand high-res touch for dexterous policies | array only (no 6-axis per-taxel F/T); price; weight spec inconsistent (0.8 vs 1.2 kg) |
| Wuji Hand ([ref](https://www.roboticscenter.ai/hardware/paxini-gen3)) | 2026 | verify | built-in 768-point tactile | ready-made tactile dexterous hand used in Tactile Genesis study | price/availability unverified |
| Allegro Hand V5 Sense ([store](https://www.roboticscenter.ai/store/product/allegro-hand)) | V5 | $13,000 | 4-finger, tactile-sensing version | standard research hand; Xela uSkin compatible | 3-axis taxels, not dense arrays |
| AGIBot Omnihand 2025 / Pro ([store](https://www.roboticscenter.ai/store/product/agibot-omnihand-2025)) | 2025 | $3,775 / $14,610 | 2025 hand | cheapest credible dexterous platform | tactile coverage unverified |
| Linker Hand O7 ([retail](https://uk.robotshop.com/products/linkerhand-linker-hand-o7-right-7-dof-lightweight-dexterous-robotic-hand-with-built-in-tactile-array)) | 2025/26 | listed retail | 7-DoF, built-in tactile array | low-cost tactile-capable hand | resolution/protocol undocumented |
| Sensors ([2026 buyer guide](https://www.roboticscenter.ai/guides/best-tactile-sensors-robot-learning)) | 2024–26 | GelSight Mini low-4-figures; DIGIT open HW <$1k; TacTip DIY ~$500; ReSkin/AnySkin open; Contactile 4-figures; Xela 4–5-figures | 25–30 Hz (vision) … 1 kHz (Contactile) | default community stacks: GelSight Mini (papers), DIGIT/TacTip/ReSkin/AnySkin (open), Xela/Contactile (calibrated 3-axis) | vision sensors capped ~30 Hz; calibrated 3-axis taxel arrays expensive; BioTac discontinued |
| Meta Digit 360 ([GitHub](https://github.com/facebookresearch/digit360)) | 2024/25 | open designs/firmware | multimodal fingertip (tactile, force, thermal, audio) | richest single-fingertip modality set | limited availability; not a mass-market product |

## 1d. Foundation / pretrained models accepting touch
| item | released | licence | what it enables | what is missing |
|---|---|---|---|---|
| N₀-TWAM ([GitHub](https://github.com/neoteai/N0-TWAM)) | 2026 | verify | tactile-native world-action model | weights/licence unverified |
| N₀-VTLA ([ref](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/commit/ed2b4ce6d5bc740c846ff6423eb9572b1a1dedd3)) | 8/2026 | verify | scales VLA with latent tactile tokens | code/checkpoints unverified |
| ME-Dex 1.0 ([arXiv 2609.21449](https://arxiv.org/pdf/2609.21449)) | 9/2026 | verify | one world-action model across heterogeneous tactile hardware | cross-hardware coverage unverified |
| TLA / BiTLA / VLA-Touch ([refs](https://www.semanticscholar.org/paper/TLA%3A-Tactile-Language-Action-Model-for-Contact-Rich-Hao-Zhang/165dc7444c6bc95673dc13244704baa47789ddf7)) | 2025–26 | verify | tactile-conditioned language/action policies | no single dominant public checkpoint |
| UniTouch ([CVPR 2024](https://github.com/cfeng16/UniTouch)), Sparsh (2025) | 2024/25 | verify | unified/self-supervised touch encoders | VBTS-centric |

## 1e. Tooling that lowers the barrier
| item | what it enables | what is missing |
|---|---|---|
| Genesis tactile "clean/mild/strong" imperfection profiles ([docs](https://genesis-world.readthedocs.io/en/latest/user_guide/sensing/tactile.html)) | built-in **domain randomisation for touch** (drift, hysteresis, dead taxels, crosstalk, gain, delay) with clean ground-truth branch for supervised calibration studies | fixed profiles, not learned/automatic DR adapted to a real sensor |
| BIFTA ([arXiv 2609.08673](https://arxiv.org/pdf/2609.08673)) | few-shot adaptation to unknown tactile sensors (cross-sensor transfer) | no released unified sensor-model zoo |
| PaXini ROS2 driver ([ref](https://www.roboticscenter.ai/hardware/paxini-gen3)) | standard-ish ROS2 topics: `/paxini/wrench`, `/paxini/taxels`, `/paxini/contact_state` | vendor-specific messages; no cross-vendor tactile standard |
| Open hardware: DIGIT, TacTip, ReSkin, AnySkin, Digit 360 | buildable/repeatable sensor hardware + firmware | calibration/protocol still per-lab; no shared tactile calibration suite found |
| Isaac Lab + Genesis + Gym registries | tactile envs usable with PPO/IL without bespoke infra | few released tactile task suites with grading beyond success rate |

## 2. Three most consequential new capabilities of 2026
1. **Tactile sim at vision-scale, open source.** Genesis World's tactile module runs >16k envs / >1,000 taxels/hand at >600k steps/s on one consumer GPU with hardware-realistic imperfections — tactile RL is no longer 10–100× slower than vision RL.
2. **Sensor abstraction as a controlled experimental variable.** Same hand, same task, swap tactile representation (depth vs aggregate force vs taxel F/T vs elastomer vs proximity): the Tactile Genesis sweep shows the best representation is task-dependent (depth for grasp-and-lift, taxel-resolved F/T for in-hand rotation) — a design question that could not be asked systematically before.
3. **Failure/recovery supervision arrives — for vision, not touch.** REBOOT (2,160 demos, half failure+recovery, phase-level labels) plus cheap GPU-parallel tactile sim now permits, in principle, failure-aware tactile learning; the tactile failure dataset itself does not exist yet.

## 3. Questions newly answerable (not in 2024)
- Which tactile representation/abstraction should a dexterous policy consume, as a function of task regime and sensor imperfection level? (sweep at 16k envs with swappable readouts)
- How does policy robustness degrade with dead taxels / crosstalk / hysteresis / drift, quantified per task? (imperfection profiles + clean ground truth)
- Does co-located 6-axis F/T + dense array beat array-only or F/T-only fingertips for slip, twist and in-hand rotation? (PaXini PX-6AX GEN3 hardware + taxel-resolved force in sim as the matched abstraction)
- Can touch-conditioned VLAs/world-action models (N₀-VTLA, N₀-TWAM, ME-Dex) transfer across tactile hardware without per-sensor retraining? (heterogeneous-tactile models + cross-sensor adapters such as BIFTA)
- Can a policy be trained to *recover* from contact failures with tactile feedback, and graded per phase rather than by binary success? (REBOOT-style protocol now portable to tactile sim)

## 4. Cheapest credible route today
- **Sim-only (recommended first experiment): ~$0 marginal, 1× RTX 4090/5090.** Genesis World (Apache-2.0) tactile + Franka/Allegro/Wuji/Sharpa assets, PPO privileged teacher then tactile-student distillation. At >600k steps/s, a 20M-step teacher ≈ 9 GPU-h; a full 3-task × 8-representation distillation sweep ≈ 20–50 GPU-h (≈ $20–150 at $1–3/GPU-h cloud). This is enough to answer representation/imperfection questions owned by no one else yet.
- **Add real touch cheaply: ~$500–$1,500.** DIGIT/TacTip/ReSkin/AnySkin + a low-cost arm (or FreeTacMan-style hand-held collection rig) ≈ low-4-figures total; a low-cost visuo-tactile gripper design is published ([arXiv 2602.00514](https://arxiv.org/abs/2602.00514)).
- **Full dexterous tactile hardware: ≥$3,775 (AGIBot Omnihand) or $13,000 (Allegro V5 Sense)**; co-located 6-axis + dense array requires PaXini PX-6AX GEN3 tiles (quote) on an Orca-style hand — likely $10k+ total. Sharpa Wave at $50,000 is the turnkey high-resolution option.
- Note the still-large sim-to-real taxel gap: Genesis readings are uncalibrated geometric estimates, so budget one real sensor for calibration/validation.

## 5. What I could NOT verify
- Licences / repo existence / checkpoints for Taccel, N₀-TWAM, N₀-VTLA, ME-Dex 1.0, Tac2Real, TaCauchy, and the MuJoCo tactile repos (search results only; GitHub pages returned navigation-only).
- Exact prices for GelSight Mini, PaXini PX-6AX GEN3, Wuji Hand, Linker Hand O7; whether AGIBot Omnihand's price includes tactile skin.
- Whether any dataset contains **tactile** failure/negative episodes: REBOOT has failures but no touch; SoftVTBench and the HF tactile sets were not inspected page-by-page. **Q2 therefore stands as an open gap for touch** (vision-only failure data now exists).
- Whether the anonymous Tactile Genesis code release is public and matches the shipped Genesis World implementation (docs warn behaviour may differ).
- Nature MI 2025 "Embedding high-resolution touch across robotic hands" ([link](https://www.nature.com/articles/s42256-025-01053-3)) — fetch blocked by cross-origin redirect; likely the Sharpa/Wuji dense-skin study, unconfirmed.
