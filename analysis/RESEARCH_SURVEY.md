# Tactile manipulation — literature survey for our Bench2Dex research

Generated 2026-09-29. Five parallel sub-surveys, ~150 papers screened, **~70 read in full**
(arXiv HTML → text; 24 full texts cached in `~/survey/txt/` for re-grepping).

Sub-reports:
`tactile_world_models_survey.md` · `tactile_vla_survey.md` · `tactile_cross_embodiment_survey.md`
· `bench2dex_lit_survey.md` · `tactile_servoing_contact_phase_survey.md`

**Trust level.** Venues are confirmed only where a proceedings page was fetched (TACTIC = RSS 2026,
AnyTouch = ICLR 2025, SARM = ICLR 2026, TacFiLM = ECCV 2026, 3D-ViTac = CoRL 2024, ForceVLA =
NeurIPS 2025). Everything else is reported as an arXiv preprint. Papers listed as "index-only" in the
sub-reports were **not** read and must not be cited without fetching. Paywalled and therefore unread:
IEEE 11593873 (DexPiHNet), IEEE 11543317 (HIVE-6D), IEEE 11330173 (ViTaDex), Nature s41467-026-68753-1.

---

## 0. Executive summary

Four independent sub-surveys converged on the same empty intersections. Ranked by confidence:

| # | Empty intersection | Confidence | Established by |
|---|---|---|---|
| **E1** | **Tactile × stage/progress supervision** — the stage-aware family is uniformly RGB/joint/language/force; the tactile family never predicts stages or progress | high | ④⑤ |
| **E2** | **Cross-hand / cross-morphology transfer of a tactile-conditioned policy or predictive model**, zero-shot to a held-out hand | ~85% | ①③ |
| **E3** | **Tactile-derived contact phase routing among several policy heads** | high | ⑤ |
| **E4** | **Touch-vs-no-touch photometric-invariance measurement** (lighting/texture/background/camera) | 0 papers | ②① |

And the strategic fact that frames all four: **the Bench2Dex authors have the pieces for E2 but have not
combined them, and have never run a closed-loop tactile experiment** (their own design review, below).

---

## 1. Sub-area 1 — tactile world / action models: **exploded, do not enter frontally**

Went from ~3 to **20+ papers between Jun and Sep 2026** (TacWAM, TouchWorld, N0-TWAM, Dream-Tac,
FeelWorld, TACO, HiTac-WAM, DexTacWAM, DTWM, HapticWAM, ME-Dex 1.0, TacPAC, ViTacWorld, VT-WAM,
Tactile-WAM, TacForeSight, OmniVTA, Agile-WAM, …).

- **Converged recipe:** Mixture-of-Transformers + frozen video VAE (Wan2.2 / Cosmos) + flow matching,
  jointly denoising future video + future touch + actions.
- **Live controversy** is no longer "predict touch?" but *how to route predicted touch into the action
  branch without destroying the visual path* ("tactile pollution" — naive future-touch tokens collapse a
  WAM from 5.8% → 1.3%). Four incompatible answers: attention masks (TacWAM AGT, Tactile-WAM VideoClean),
  contact gating (FeelWorld, HiTac-WAM, VT-WAM AVTAG), weight isolation (N0-TWAM tactile expert),
  asymmetric phase-aligned routing (OVTF/AFM).
- **Headline numbers:** TacWAM 75.0% vs 37.5% best baseline (4 real tasks, parallel gripper);
  TouchWorld 65.0% clean / 53.7% perturbed (6 real long-horizon, Wuji hands + tactile glove);
  N0-TWAM 6 embodiments / 450 tasks / >30,000 h, UniVTAC 84.5% / real 46.3%.
- **Closest to our setting:** **ViTacFormer** (2506.15953) — bimanual **SharpaWave 17-DoF**, autoregressive
  next-touch prediction, 11-stage 2.5-minute hamburger task. Read this first.
- **Only ~4 of 20+ WAMs measure visual perturbation**; N0-TWAM is the only one with a real result
  (lighting/background 45% vs π0.5 25%).
- **White space (their words: "the biggest")**: cross-embodiment transfer of the *predictive* model
  itself — only N0-TWAM (one sensor family) and ME-Dex 1.0 attempt it.
- **Likely-missed but important:** **DexTouch-WM (2609.20649)** — action-conditioned tactile world model
  from human touch, scales human data 0→100 h, improves robot-domain prediction on disjoint tasks. Closest
  thing to a human→robot cross-embodiment *tactile world model*. Read before pursuing E2 via prediction.

## 2. Sub-area 2 — tactile VLA fusion: **crowded on mechanism, empty on invariance**

- **Crowded:** "bolt a tactile encoder onto π0 / OpenVLA and concat tokens" (Tactile-VLA, OmniVTLA,
  ForceVLA, TacVLA, HapTile). **Proven negative:** naive concat wastes tokens and hurts OOD (ForceVLA,
  TacFiLM — FiLM modulation 54.67% → 86.67% OOD beats TactileConcat 73.33% and Cross-Attn 49.3%).
- **GR00T N1.5 + tactile: exactly one** — MoSS (2604.23272): decoupled tactile + joint-torque streams on the
  action expert, two-stage training; 20.8% → 42.7% (tactile) → 49.0% (tactile+torque) over 4 real tasks.
  It also ports Tactile-VLA (30.2%) and ForceVLA (34.4%) onto GR00T N1.5 as baselines.
- **π0.5 + tactile: several** (TacVLA, STAR, T-Rex; HapTile uses π0).
- **Cross-embodiment tactile fusion: exactly one** — ME-Dex 1.0 (2609.21449): Canonical Hand Model maps
  grippers and dexterous hands onto shared finger/palm regions + Unified Tactile Autoencoder; 3-expert MoT.
  But it spans *different benchmarks/datasets*, not held-out-hand zero-shot.
- **E4 is empty:** occlusion robustness is studied widely (TacVLA: block the front camera → ~30% → >60%;
  ForceVLA 90%; 3D-ViTac occlusion-vs-#cameras). **No paper runs a controlled photometric-nuisance
  protocol with a touch-vs-no-touch comparison, and none decomposes invariance vs geometry equivariance.**
- **Counter-evidence to E4:** HATO reports touch did **not** fix appearance shift; HapTile's own limitations
  note its data lacks lighting/background variation. So E4 is open *and contested*, not trivially true.

## 3. Sub-area 3 — cross-embodiment tactile: **no, not demonstrated (~85%)**

Nothing trains a tactile-conditioned policy on one set of dexterous hands and transfers it zero-shot to a
different finger structure **and** a different tactile layout, especially in sim across many hands.

- **Bench2Dex is the substrate and says so itself:** 12 bimanual dexterous hands, 26 tasks, ~1.3 K demos,
  unified **240×240** surface-aligned tactile maps (8–10 sites/hand). Its own text: *"tasks and embodiments
  are not factorially crossed"*, and the conclusion proposes *"extending evaluation to tactile-conditioned
  policies and transfer across dexterous hands"* as future work.
- **Cross-hand zero-shot is already solved — without touch:** CrossDex (4→2 hands), **DexGrasp-Zero**
  (RSS 2026; 4 hands → 2 unseen, 85% zero-shot, +59.5% over SOTA, 82% real), XL-VLA, UHAS (the word
  "tactile" appears 0 times).
- **The single most useful citation for motivation:** DexGrasp-Zero makes **tactile privileged in the sim
  teacher and removes it at deployment** (student replaces touch with a 5-step LSTM) — and names missing
  touch as a dominant failure cause. The field throws touch away because it cannot cross hands.
- **Tactile cross-*sensor* transfer works, but morphology is fixed or re-encoded:** FTP-1 (unseen sensors
  15.0% → 46.6%, but a fresh per-sensor encoder is trained), T3, UniTacHand (MANO UV map, one robot hand),
  TactAlign (human glove → robot, one hand).
- **Canonicalization is a convergent program that stops one step short:** UniTacHand (MANO UV) · Bench2Dex
  (surface-aligned 240×240 × 12 hands) · Wu et al. 2409.17549 (taxel unit-frame + 6-D sensor pose + 3-D
  force: **0% without canonicalization, 78% with**, but one LEAP hand) · TacMap · FTP-1 MTTS.
- **Unresolved lead:** IEEE Access ~Jul 2026, *"A Simulation Benchmark for Dexterous Peg-in-Hole Assembly
  With Force-Tactile-based Pose Estimation Across Multiple Embodiments"* (DexPiHNet, Isaac Gym, 49.68%).
  Paywalled; only a secondary Chinese summary was readable. By title it is the closest match to E2 and must
  be read before any novelty claim.

## 4. Sub-area 4 — tactile benchmarks + stage-aware: **the empty intersection E1**

**E1 (high confidence): no work combines stage/progress supervision with tactile.**

- Stage-aware family, all without touch: **SARM** (ICLR 2026, 2509.25358 — jointly predicts discrete stage +
  fine progress from natural-language subtask labels; T-shirt folding 83%/67% vs 8%/0%), **STARE/STARE-VLA**
  (2512.05107), **PACE** (2608.15026 — **phase boundaries from simulator semantic predicates**, the clearest
  privileged-predicate example, but no tactile), PRIMO R1, Bi-HIL.
- Tactile family never predicts stages or progress.
- **The privileged→deployable pattern exists but is inverted from the naive framing:** privilege = contact
  force / object pose / physics; **tactile = the deployable channel.** CableVLA (2609.25606: TacSense, a
  128-D deployable tactile encoder supervised by sim contact-kinematics + slip/rolling event labels; 84.9%
  vs 62.6%), HydroShear (2603.00446: privileged force+pose in the critic, tactile shear in the actor; 93%
  zero-shot sim-to-real vs 34% for tactile-image policies), PTLD (2603.04531: privileged object pose/shape
  oracle distilled into a tactile state estimator).
- **No work uses simulated tactile as the privileged teacher**, and **no work combines stage supervision
  with tactile.** Bench2Dex uniquely has all three ingredients: per-stage executable predicates, a
  surface-aligned tactile stream, and object/contact ground truth.

**Bench2Dex follow-up scan:** arXiv `all:"Bench2Dex"` → 1 entry (the paper). Semantic Scholar → 1 citing
paper, which greps as a single reference-list hit in a co-author's 3D-reconstruction paper. Zero hits for
"Bench2Dex" in ME-Dex, HiTac-WAM, CableVLA, ReTouch, PTLD. (Coverage-limited: indexing lags.)

**The benchmark's own baselines are vision+proprioception only.** Full-text grep of the paper: no
"cross-attention", no "Tactile_Cross", no "post_dit", no tactile-trained row in the results table. The repo
ships four tactile forks with **no published numbers**. So **there is currently no reported
tactile-vs-no-tactile result on Bench2Dex**, and any claim needs a matched no-tactile baseline.

**Benchmark comparison highlights (7 tactile-adjacent benchmarks read in full):** only **roto 2.0** and
**SoftVTBench** are closed-loop sim policy benchmarks; EgoTactile/HT-Bench are human-hand only and
representation-level; RCT is retrieval only; HRDexDB has no policy baselines and lacks tactile for Allegro;
TactiDex uses touch as a reward and its deployed policies *do not use tactile feedback*. **None splits
robustness into invariance vs equivariance axes** — so Bench2Dex's framing is novel but has no comparable
baseline, which will complicate cross-benchmark claims.

## 5. Sub-area 5 — tactile servoing / contact phase

- **Q1 — "vision in free space, tactile on contact" is an established recipe**, but almost always with
  parallel grippers + GelSight/force-torque on real robots. Cleanest learned instances: **CAAT**
  (2608.01102 — a CNN contact classifier on |T_t − T_ref| scales vision vs tactile attention), **Kamijo
  et al.** (2309.15681 — tactile threshold triggers a dual-policy handover, 90% at <0.1 mm clearance vs 5%),
  MS-Bot/Play to the Score (2408.01366), RDP (2503.02881 — slow visual + fast tactile loop, bimanual).
  **Warnings:** CAAT's own ablation shows binary gating **loses** (51.8% vs 69.6% for continuous scaling);
  RDP reports observation-level tactile fusion did **not** beat DP.
- **Q2 — {sim + bimanual + multi-finger + many morphologies + vision→tactile handover}: infrastructure
  exists (Bench2Dex), the recipe does not.**
- **E3 — no paper routes among several learned policy heads using a tactile-derived contact phase.**
  Phase routers exist but are vision/proprioception-gated (BRIDGE 2606.26603 +36.7%; TRACT 2607.29285 10/10
  vs 3/10; MoE-ACT 2601.21971; StageACT 2509.13200). Tactile gating exists but gates *modalities* (CAAT) or
  does a *two-way* handover (Kamijo).

## 6. The authors' own position (repository + internal design review)

Read from the official repo (`github.com/Bench2Dex/Bench2Dex`; the local clone is a fork).

| Component | What it is | Tactile? | Status |
|---|---|---|---|
| `GR00T_XE` | **Cross-embodiment**: 12 hands × 26 tasks unified into a 64-D action space (arm ee 6+6, hand 22+22 semantic slots, 8 padding); `embodiment_mapping.yml` (12 hands → 44 slots); pretrain → per-task finetune | **no** | code present, no published results |
| `GR00T_n15_Tactile` | tactile fused **pre-DiT** (single pooled tactile token appended to Eagle VL tokens) | yes | no published results |
| `GR00T_n15_Tactile_Cross` | tactile fused **post-DiT** via cross-attention: DiT action tokens = Q, tactile tokens = K/V; zero-init `out_proj` (identity at init); 0.3 whole-modality dropout | yes | no published results |
| `ACT-Tactile` | tactile ACT | yes | no published results |

**`Tactile_Cross` means TacMap Cross-Attention, NOT cross-embodiment** — the README's first line is
"# GR00T N1.5 TacMap Cross-Attention".

**The design review (`policy/GR00T_n15_Tactile_Cross/TACTILE_DESIGN_REVIEW.md`, dated 2026-09-08, six days
before arXiv v1) is effectively the authors' own problem list.** It states the code "尚未训练新模型或运行
Isaac Sim 闭环实验，因此不声称成功率已提升" — **no closed-loop tactile experiment has been run**. Confirmed
defects, with their own priorities:

| Defect | Consequence | Their priority |
|---|---|---|
| ~0.8 s sim-time feedback interval (16 queued steps before replanning) | queued actions ignore new observations | — |
| `pre_dit` Q is noised actions, not yet fused with vision/language | residual in action-feature space ≠ physical correction | — |
| only the current tactile frame is read | cannot estimate contact *trends* | **high** |
| 240×240 pooled to 2×2 | attenuates sparse local contact | medium |
| signal = CPD-filtered normal geometric distance + contact mask | "**cannot claim to measure normal force, shear, or friction**" | **high** |
| gradients only from action flow-matching | needs a tactile-intervention test to prove the network uses touch | **high** |
| fixed 0.3 whole-modality dropout | fewer effective tactile samples; optimum unknown | medium |

Their stated next steps: short-chunk replanning → **causal tactile history `[t-3..t]` + small temporal
encoder** → more contact-phase coverage and misalignment/recovery demos → **auxiliary contact-state /
change-prediction losses** → slow-fast (RDP/VLA-Touch-style) structure if latency forbids full replanning.
They also admit they could not verify the "TouchRefine" paper their design references.

**Methodological warning worth adopting verbatim:** *"不要只比较 A 与 E 后就把所有收益归因于
cross-attention。无触觉基线也要匹配执行前缀长度，区分更多视觉重规划和触觉本身的收益。"* — i.e. the
no-tactile baseline must match the execution-prefix length, so that extra visual replanning is not credited
to touch. **We already have the matched no-tactile baseline** (ACT/DP/π0.5/GR00T, 50 episodes × 3 tasks).

**Also note:** the review says "本项目跨多种手型" — cross-hand is their *intent* — but `GR00T_XE`
(cross-hand, no touch) and the tactile forks (touch, single embodiment) are **not combined**.

## 7. Counter-evidence assembled (read before betting)

| Evidence | Threatens |
|---|---|
| HATO: touch did **not** fix appearance shift; policies stay appearance-sensitive | E4 |
| HapTile: touch **hurt** DP+marker on wipe 80% → 30% and pour 50% → 20% | E4, all tactile claims |
| CAAT: binary contact gating (51.8%) **loses** to continuous scaling (69.6%) | E3 |
| TacPAC: naive future-tactile prediction recovers only ~**1/3** of achievable gain (prediction precedes execution, touch arrives during it) | prediction-based designs |
| "Tactile pollution": naive future-touch injection collapses a WAM 5.8% → 1.3% | E1/E2 via world models |
| RDP: observation-level tactile fusion did **not** beat DP | naive fusion |
| The authors intend cross-hand + tactile | E2 novelty |
| Field moves 1–2 papers/week (Jun→Sep 2026: 3 → 20+) | everything; needs a pre-submission re-check |
| Bench2Dex's invariance/equivariance axes have no comparable benchmark | cross-benchmark claims |

## 8. Candidate proposals

**P-A (recommended) — stage-supervised tactile contact phase, routed into the policy, evaluated on
held-out hands.** Use the benchmark's **executable stage predicates** (privileged simulation) to supervise a
tactile contact-phase/progress head. Use that phase to **route or gate** the policy (phase-specific heads or
a phase-conditioned residual), with a canonicalized tactile representation. Measure with the benchmark's own
**invariance/equivariance axes + LSCR + a held-out-hand matrix**.

- Hits E1 (stage × tactile — empty), E3 (tactile phase routing — empty), E2 (held-out hands).
- Differentiation from the authors: they do **single-policy residual fusion + more data + temporal history**;
  nothing in their plan uses stage predicates as tactile supervision, and they have run no closed-loop
  tactile experiment.
- Differentiation from the literature: phase routers are vision/proprio-gated; tactile gating gates
  modalities; the stage-aware family has no touch.
- Assets available: the Bench2Dex tactile stream itself (240x240 surface-aligned maps, 8-10 sites per
  hand, 12 embodiments), the repo's own executable stage predicates and evaluator, and our reproduced
  vision-only baselines as a matched control.

**P-B (cheap, clean, smaller) — the touch-vs-no-touch invariance protocol.** Contribute a controlled
photometric-nuisance protocol with a touch-vs-no-touch decomposition of invariance vs geometry equivariance,
on top of Bench2Dex's existing axes. E4 is 0 papers deep and the benchmark supplies the axes.
Risk: small story; HATO/HapTile counter-evidence means the result may be null — though a *rigorous null* is
itself publishable and would sharpen the field.

**P-C (highest ceiling, highest risk) — cross-hand tactile world model.** Attack the sub-survey's "biggest
white space": a predictive tactile model that transfers across hand morphologies. Head-on with the authors'
intent and with 20+ WAM papers; needs the canonicalization program (E2's OP3: *what should be invariant vs
equivariant under a change of hand?*).

## 9. Minimal falsifiable first experiments (data-only; no policy training, no GPU queue)

All four are cheap and each one either establishes or kills the motivation for P-A.

1. **Tactile → stage probe.** Train a classifier from tactile maps alone to predict the benchmark's latched
   stage. If tactile cannot predict stage at all, the phase head of P-A is unfounded.
2. **RGB → "seated / slipping / contact established" probe.** If RGB is near-random, that is the hard
   evidence for tactile necessity — the paper's Figure 1.
3. **Cross-hand tactile similarity.** For the same contact event across the 12 hands, quantify how different
   the tactile maps are. Motivates canonicalization and tests OP3 directly.
4. **Invariance quantification (offline).** Under invariance-axis perturbations (lighting / tabletop
   texture / background / camera pose), measure ΔRGB vs Δtactile on the same frames. If RGB moves a lot and
   tactile barely moves, E4's premise is established without touching a policy.

Candidate tasks: **73** (all vision-only baselines 0/50; GR00T LSCR 0.710 with **zero** zero-progress
episodes but only 8/50 success — failure is concentrated in the final contact-critical stage) and **32**
(partial: ACT 8, DP 3, π0.5 19, GR00T 12). Task 26 is the wrong target: vision already works (π0.5 and GR00T
both 19/50) so headroom is small.

## 10. Tier-1 reading list before writing anything

| # | Work | Why |
|---|---|---|
| 1 | **ViTacFormer** 2506.15953 | bimanual SharpaWave 17-DoF, multi-stage — closest to our hardware/task setting |
| 2 | **DexGrasp-Zero** RSS 2026 | the motivation citation: touch privileged in sim, removed at deployment, blamed for failures |
| 3 | **ME-Dex 1.0** 2609.21449 §3.3/§4.4 | the only cross-embodiment tactile fusion; the Canonical Hand Model |
| 4 | **PACE** 2608.15026 | phase boundaries from simulator semantic predicates — closest to P-A's supervision idea |
| 5 | **MoSS** 2604.23272 §3/§4.4/App D | the only GR00T N1.5 + tactile recipe, with numbers |
| 6 | **TacFiLM** 2603.14604 | fusion-mechanism ablation; matches our OpenVLA-OFT codebase |
| 7 | **TacVLA** 2603.12665 + **T-Rex** 2606.17055 + **STAR** 2609.12549 | π0.5/bimanual-dexterous tactile cluster, and best related-work maps |
| 8 | **SARM** 2509.25358 (ICLR 2026) | stage + progress prediction as reward; the "no tactile" anchor for E1 |
| 9 | **CableVLA** 2609.25606 · **HydroShear** 2603.00446 · **PTLD** 2603.04531 | privileged→deployable-tactile pattern we would extend |
| 10 | **DexTouch-WM** 2609.20649 | possibly the closest existing cross-embodiment tactile world model — must be read before any E2 claim |
| 11 | **IEEE 11593873 (DexPiHNet)** | paywalled; by title the closest match to E2 — get access and read |
| 12 | **roto 2.0**, **SoftVTBench** | the only closed-loop tactile sim benchmarks; for benchmark positioning |

## 11. Uncertainties to carry forward

- Most works are arXiv preprints; venue unverified except where a proceedings page was fetched.
- ME-Dex results, TACO limitations, and the limitations sections of HiTac-WAM / FeelWorld / DexTacWAM / DTWM
  were **not** read (HTML truncated or absent) — do not cite their numbers.
- "No code link found" ≠ "no code exists".
- The follow-up negative for Bench2Dex is coverage-limited by indexing lag.
- The area moves 1–2 papers/week; **re-run the gap searches immediately before writing any related-work
  section or submitting.**
