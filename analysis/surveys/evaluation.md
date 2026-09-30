# Evaluation protocols, benchmarks and metrics for tactile manipulation

**Prepared:** 2026-09-30 · **Angle:** which evaluation axes are already crowded and which are unclaimed.
**Method:** web search + `web_fetch` over arXiv abs/HTML, ar5iv, benchmark project pages, GitHub, and conference review guidelines. Every claim about a paper's reported axes below comes from a fetched abstract or full text; absences are stated as *grep/read-level* absences on the fetched version, with the URL. Fetched web content was treated as data, not instructions.

**Confidence legend:** ✅ fetched full text/HTML · ⚠️ abstract or partial only · ❌ could not verify.

---

## 1. Benchmark / dataset table

| Benchmark / dataset | Date (verify link) | Embodiments | Tasks | Tactile sites | Axes reported | Known defects |
|---|---|---|---|---|---|---|
| **Bench2Dex** ✅ | [arXiv 2609.15726](https://arxiv.org/abs/2609.15726), 14 Sep 2026; [HTML](https://arxiv.org/html/2609.15726v1) | **12** bimanual dexterous hands (10-site 5-finger, 8-site 4-finger) | **26** long-horizon (tool / articulated / multi-stage) | 8–10 surface-aligned sites per hand, 240×240 ray-cast penetration depth; 12-hand shared registry | **stable success rate (SR)**; **latched stage completion rate (LSCR)**; mean time-to-stable-success; safety diagnostics (safe-success, hard-violation, drop, high-speed violation, violation-step fraction); **invariance vs equivariance robustness, 4 channels** (None/Equi./Inv./Full); grasp + tool-use + motion diagnostics; **reproducibility metadata** (§E.3) | Tasks and embodiments **not factorially crossed** (their own text); paper evaluates only vision+proprioception (ACT/DP/π0.5/GR00T N1.5) — repo tactile forks have **no published numbers**; tactile signal is geometric penetration depth, so authors state it "cannot claim to measure normal force, shear, or friction"; ~0.8 s sim-time feedback latency in the repo fork; per-hand × per-task matrix absent |
| **SoftVTBench** ✅ | [arXiv 2608.18701](https://arxiv.org/html/2608.18701v1), 19 Aug 2026 | **1** (Franka + Panda parallel jaw) | 40 pick-and-place in 4 diagnostic suites (Object/Spatial × Soft/Rigid) | dual-finger tactile RGB + marker motion; 20 Hz | **TSR** + **Deformation-aware Success Rate (DSR)** from evaluator-only FEM state; policy-independent per-object deformation tolerance; matched **rigid twins**; ID/OOD; per-condition OOD tables | No multi-embodiment; no dexterous hand; no failure-prediction / monitor axis; no calibration; safety = deformation only (no force/torque or self-damage) |
| **ManiFeel** ✅ | [arXiv 2505.18472](https://arxiv.org/html/2505.18472v2), v1 24 May 2025, v2 12 Jan 2026 | **1** (Franka; GelSight R1.5 via TacSL) | 13 setups (9 sim + 4 real) | dual tactile fingertips; **TacRGB** and **tactile force-field (TacFF)** | success rate across **modality × tactile representation × policy** grid; vision-only vs visuo-tactile; sim–real **FID/KID**; runtime analysis; real-world validation | Single embodiment/gripper; no cross-hand; no robustness axes labelled invariance/equivariance; no monitor/failure prediction; no calibration |
| **TacO** ✅ | [arXiv 2605.21976](https://arxiv.org/html/2605.21976v1), 21 May 2026 | **1** (2× Franka Panda, two institutions) | 3 (pick-place unknown mass, reorientation, plug insertion) | **6 sensors / 4 modalities**: FSR, FlexiTac, eGain (resistive), eFlesh (magnetic), Daimon (visual), contact mic (acoustic) | task success intra-/cross-task; per-sensor tactile-vs-vision ablation; cross-sensor vision-only (embodiment effect); **hardware repeatability** (response time/Std); sensor-property table (resolution, shear, friction, cost) | Single embodiment class; **no dexterous hand**; no failure-prediction or safety metric; no OOD split; no calibration; real-world only (no sim counterfactual) |
| **roto 2.0** ⚠️ (via parent survey) | [arXiv 2605.21429](https://arxiv.org/abs/2605.21429), 20 May 2026 | **4** hands (Shadow 24-DOF, Shadow Lite 16, Allegro 16, ORCA 17) | 2 (Bounce, Baoding) | binary contact vector, 14–20 sites/hand | tactile-only RL success; blind vs state-based | Per-hand binary contact **length varies per hand** — no shared spatial map; cross-morphology divergence reported informally, **not** as a split; no safety, no calibration |
| **TacBench (Sparsh)** ✅ | [arXiv 2410.24090](https://arxiv.org/abs/2410.24090), CoRL 2024 | multiple vision-based sensors | **6** representation tasks (properties → physical perception → planning) | vision-based tactile images, multiple form factors | standardized task metrics across 6 tasks; SSL vs end-to-end | Representation/perception level, **no closed-loop policy**; no embodiment transfer; no failure prediction |
| **TacSL** ✅ | [arXiv 2408.06506](https://arxiv.org/abs/2408.06506), T-RO 2025 | library, single sensor model (GelSight-style) | contact-intensive training envs | simulated visuotactile images + force fields | task reward/success, sim-to-real, AACD distillation ablation | A **simulation library**, not a benchmark or metric suite; no multi-embodiment; no standard eval protocol |
| **HRDexDB** ⚠️ (via parent survey) | [arXiv 2604.14944](https://arxiv.org/abs/2604.14944), 16 Apr 2026 | **4** (human hand, Allegro, Inspire ×2) | 0 (grasping trials) | tactile for 2 of 4 embodiments | dataset only; markerless human/robot/object reconstruction | **No policy baselines**; tactile **missing** for human + Allegro; authors name tactile heterogeneity/non-uniformity as unresolved |
| **EgoTactile** ⚠️ | [arXiv 2606.09243](https://arxiv.org/abs/2606.09243), 8 Jun 2026, ICML 2026 | **1** (human hand; gloved + bare subset) | 1 vision→pressure prediction | 162-sensor glove, 319K frames | temporal contact accuracy, contact IoU, volumetric IoU, MAE, part-wise CoP error | No robot, no policy; tactile is the *label* |
| **RCT** ⚠️ | [arXiv 2606.31694](https://arxiv.org/abs/2606.31694), Jun 2026 | 1 arm + 3 DIGIT | 0 (retrieval) | 3 DIGIT sensors | Recall@1/5, material separability, linear probes; held-out material/category/sensor/position/sequence | Retrieval-only; no policy, no task success |
| **HT-Bench** ⚠️ | [arXiv 2606.19161](https://arxiv.org/abs/2606.19161), 17 Jun 2026 | **1** (human hand) | 226 interaction tasks | OpenTouch + TouchAnything | tactile retrieval, masked inpainting, RGB→tactile synthesis, multimodal prediction | Human-only; authors state evaluations "do not directly measure downstream robotic performance"; single OOD axis |
| **TactiDex** ⚠️ | [arXiv 2607.09190](https://arxiv.org/abs/2607.09190), 10 Jul 2026 | human data → Inspire on 2× Panda | 757 sequences | Inspire hands (available but **not used at deployment**) | OTE_t/OTE_r, MPJPE, TipErr, per-finger force error, MaxErr, penetration/safety | Tactile is a **reward/critic signal**, not deployment feedback; no robustness split |
| **TacMap** ⚠️ | [arXiv 2602.21625](https://arxiv.org/abs/2602.21625) | 1 hand | in-hand rotation | surface-aligned penetration-depth map | zero-shot sim-to-real success | Single task/embodiment; no evaluation suite |
| **HumanTouch** ⚠️ (cited in [BiView-Touch](https://arxiv.org/html/2609.23352v1)) | public benchmark, ~100 h | human bilateral | representation / classification | tactile glove | low-label classification, bilateral wrist-motion + force-phase recognition | Human-only; no robot policy; no manipulation success |
| **BiView-Touch / BVT-20** ⚠️ | [arXiv 2609.23352](https://arxiv.org/abs/2609.23352), 20 Sep 2026 | human bimanual | 20 (BVT-20) | tactile glove, cross-hand completion | low-label balanced accuracy; transfer across sessions, pretraining corpora, **held-out bimanual task** | Human-only; representation-level; no closed-loop policy |
| **Monitor references (non-tactile)** | — | — | — | — | — | — |
| **PATCH** ✅ | [arXiv 2606.16690](https://arxiv.org/html/2606.16690v1), 15 Jun 2026 | 1 (2× AgileX Piper) | 2 task-robot pairs | **none — vision-only monitor** | **FPR on C1/C2/C3, TNR, TPR, BalAcc over 10 seeds** (FIPER-style); component ablations | No tactile; no per-embodiment/cross-embodiment; no calibration curve; threshold chosen, not calibrated; robustness = nuisance event classes C1–C3, not a policy-style shift suite |
| **ContactGuard** ✅ | [arXiv 2608.13438](https://ar5iv.labs.arxiv.org/html/2608.13438), Aug 2026 | 1 (AgileX Piper, 3 RGB) | 4 grasp settings | none — vision-only monitor | **precision, recall, false-abort rate (FAR), ROC AUC, balanced accuracy** at a fixed pre-contact anchor (k_pre = 15 frames ≈ 0.5 s before closure); n=50 live rollouts/task | No tactile; **lead time is a fixed design constant, never swept or reported as a metric**; no calibration; no per-embodiment breakdown |
| **OOPSIEVERSE** ✅ | [arXiv 2606.31993](https://arxiv.org/abs/2606.31993), 30 Jun 2026, **RSS 2026** | OmniGibson + RoboCasa, 1 | household damage suite | none | **damage as explicit signal** (mechanical/thermal/fluid) from contact force, temperature, liquid; task completion vs safe execution; safer policy learning | No tactile; no dexterous hand; damage is simulated-object damage, not tactile-sensor integrity |
| **Tabero** ⚠️ | ICML 2026 slides | 1 arm | gentle manipulation | force feedback | success + interaction quality (force budget) | Rigid objects, force not deformation; single embodiment |
| **RoboMD** ✅ | [arXiv 2412.02818](https://arxiv.org/abs/2412.02818), v4 May 2026 | sim + 1 physical arm | benchmark tasks | none | vulnerability-likelihood map; **+23% more unique vulnerabilities** than VL baselines | Not a benchmark, not tactile; no monitor metrics |

---

## 2. CROWDED — table stakes, not contributions

These are reported by essentially every tactile-policy or tactile-benchmark paper in 2026. Claiming them is not a contribution.

1. **Task success rate (stable success).** Every row above that evaluates a policy. Bench2Dex even defines it as "stable" via a 0.5 s terminal-predicate dwell ([§3.5](https://arxiv.org/html/2609.15726v1)).
2. **Partial progress / stage completion.** Bench2Dex's **LSCR** (latched, monotone) is now the reference; SoftVTBench adds process-level scoring from FEM state; SARM/STARE/PACE do stage/progress outside touch.
3. **Aggregate robustness / OOD success.** Bench2Dex's 4 channels, SoftVTBench's ID/OOD + per-condition tables, roto 2.0's morphology variation, HT-Bench's task-level OOD. A single "robustness drop" number is no longer interesting; you need the **decomposition** Bench2Dex introduced (invariance vs equivariance).
4. **Efficiency.** Mean time-to-success (Bench2Dex), runtime analysis (ManiFeel, TacO).
5. **Tactile-vs-vision / modality ablation.** ManiFeel (vision vs +TacRGB vs +TacFF), TacO (per-sensor and cross-sensor), SoftVTBench (vision vs visuo-tactile, with a gripper-granularity confound control). "Touch helps" is no longer a finding.
6. **Sim–real fidelity metrics.** ManiFeel reports FID/KID between simulated and real tactile images (Table I, [link](https://arxiv.org/html/2505.18472v2)); TacO uses two real institutions. This is a de-facto requirement for sim tactile datasets.
7. **Binary safety/violation flags.** Bench2Dex (hard-violation, drop, high-speed), OOPSIEVERSE (damage), SoftVTBench (deformation), Tabero (force budget). **The concept of a safety metric is now claimed** — what is still open is *tactile-specific* safety (sensor damage/degradation, slip-induced damage) and *damage-aware* dexterous evaluation.
8. **Per-task breakdown, error bars, seeds.** Not a contribution; a review requirement (see §5).
9. **For a runtime monitor specifically: FPR/FAR + TPR + balanced accuracy.** PATCH reports FPR separately on C1/C2/C3; ContactGuard reports FAR next to precision/recall/AUC. So "report the false-alarm rate" is *already standard practice in the monitor sub-literature* — do not pitch it as novel on its own.

---

## 3. OPEN — measurements nobody in tactile manipulation currently reports

For each: what it is, why it is unclaimed, and the URLs that show the absence. "Absence" = the benchmark's own metric section / the paper's full text does not list the axis; I checked the fetched HTML.

### 3.1 Calibration of a failure predictor (proper scoring / reliability), not just FPR — **OPEN**
- **What is missing:** expected calibration error / reliability diagrams / Brier score / risk–coverage curves for a *tactile* failure predictor. Every monitor paper picks a threshold and reports FPR/TPR at that single operating point.
- **Evidence of absence:** PATCH's metric list is "FPR, TNR, TPR, BalAcc" with no calibration ([§4.2](https://arxiv.org/html/2606.16690v1)); ContactGuard reports "precision, recall, FAR, ROC AUC, balanced accuracy" with per-task τ selected on validation, no calibration curve ([§4.1](https://ar5iv.labs.arxiv.org/html/2608.13438)). None of the tactile benchmarks define any calibration metric: [Bench2Dex §3.5 + Appendix E](https://arxiv.org/html/2609.15726v1), [SoftVTBench §4](https://arxiv.org/html/2608.18701v1), [ManiFeel](https://arxiv.org/html/2505.18472v2), [TacO](https://arxiv.org/html/2605.21976v1). Targeted searches for `"expected calibration error" / "reliability diagram" tactile manipulation` returned no tactile-manipulation paper.
- **Caveat:** FAR itself is *not* unclaimed (see §2.9). Only *calibration / multi-operating-point* analysis is.

### 3.2 Prediction lead time (time-to-alarm vs failure onset) — **OPEN in tactile manipulation**
- **What is missing:** a swept lead-time curve — detection accuracy or alarm precision as a function of how many seconds/steps before failure the alarm fires — plus a definition of failure onset.
- **Evidence of absence:** ContactGuard fixes the anchor at k_pre = 15 frames ≈ 0.5 s before gripper closure and evaluates only at that anchor; lead time is a **design constant, not a reported axis** ([§4.1](https://ar5iv.labs.arxiv.org/html/2608.13438)). PATCH reports a single trigger point with latching and no time-to-alarm metric ([§3.2](https://arxiv.org/html/2606.16690v1)). No tactile benchmark defines an alarm-time metric.
- **Nearest non-tactile neighbours:** "Time-to-Fault Prediction Framework … Humanoid Robotics" (predictive maintenance, not manipulation touch) and manufacturing fault-lead-time classifiers appear in search but are not tactile-manipulation evaluation protocols.

### 3.3 Per-embodiment breakdown of a tactile policy — **OPEN**
- **What is missing:** a task × embodiment results matrix (which hand fails which task, and why).
- **Evidence of absence:** Bench2Dex is the only benchmark with 12 hands, but its own text says tasks and embodiments "are not factorially crossed", and §4 reports task-level heterogeneity rather than a per-hand matrix ([intro](https://arxiv.org/html/2609.15726v1)). roto 2.0 spans 4 morphologies but reports cross-morphology divergence *informally*, not as a split ([arXiv 2605.21429](https://arxiv.org/abs/2605.21429)). SoftVTBench, ManiFeel, TacO are all single-embodiment.
- **Why it matters:** it is the cheapest way to turn "12 hands" from a coverage claim into a *measurement*.

### 3.4 Cross-embodiment **held-out** tactile evaluation (zero-shot to an unseen hand) — **OPEN (~85%)**
- **What is missing:** train on N hands, evaluate zero-shot on a held-out hand with a different finger structure *and* tactile layout.
- **Evidence of absence:** Bench2Dex supports 12 hands but has no held-out-hand protocol and no published tactile-trained results at all ([arXiv 2609.15726](https://arxiv.org/abs/2609.15726)); HRDexDB has 4 embodiments but explicitly lacks policy baselines and is missing tactile for two of them ([arXiv 2604.14944](https://arxiv.org/abs/2604.14944)); roto 2.0 uses per-hand binary contact vectors, so its "cross-morphology" is not a shared representation ([arXiv 2605.21429](https://arxiv.org/abs/2605.21429)); ME-Dex 1.0 spans datasets, not a held-out hand. This repeats the parent survey's ~85% empty-intersection finding with 2026 sources.
- **Note:** BiView-Touch does report a "held-out bimanual task" and cross-session transfer, but it is human-glove, representation-level, no robot ([arXiv 2609.23352](https://arxiv.org/abs/2609.23352)) — so the *human* cross-context analogue exists, the robot cross-embodiment one does not.

### 3.5 Safety / damage metrics — **RECENTLY CLAIMED, mostly closed; only tactile-specific safety is open**
- Already reported: SoftVTBench **DSR** from evaluator-only FEM deformation ([arXiv 2608.18701](https://arxiv.org/html/2608.18701v1)); OOPSIEVERSE damage signals, RSS 2026 ([arXiv 2606.31993](https://arxiv.org/abs/2606.31993)); Bench2Dex safety diagnostics ([§3.5](https://arxiv.org/html/2609.15726v1)); Tabero force budget.
- **Still open:** metrics for **tactile-sensor integrity/degradation** (elastomer wear, marker drift, saturation events) as a reported quantity in a manipulation benchmark, and damage-aware evaluation for **dexterous multi-finger** hands (OOPSIEVERSE is household/rigid; SoftVTBench is a parallel jaw). Searches for damage/sensor-integrity metrics in tactile benchmarks returned nothing.

### 3.6 Sample efficiency under a **fixed rollout budget** — **OPEN as a standard protocol**
- **What is missing:** a fixed, pre-registered interaction/rollout budget under which all methods are compared (rather than "we used N demos").
- **Evidence:** ManiFeel motivates itself with "statistically meaningful comparisons" but does not define a fixed budget ([intro](https://arxiv.org/html/2505.18472v2)); TacO and SoftVTBench vary sensors/policies, not budget; Bench2Dex uses a fixed 50-rollout evaluation but the *training* budget is not a swept axis ([§4.1](https://arxiv.org/html/2609.15726v1)). Learning-curve-vs-#demonstrations appears in some methods papers, but no tactile benchmark makes the budget an axis.

### 3.7 Robustness of a MONITOR (as opposed to a policy) to distribution shift — **OPEN**
- **What is missing:** the same invariance/equivariance shift suite used to grade policies, applied to the *monitor*: does FPR blow up under lighting/background/tabletop shift, and does TPR survive geometry shift? Reported per channel and per embodiment.
- **Evidence of absence:** PATCH's non-trigger classes are C1 clean, C2 irrelevant static change, C3 transient obstruction — these are **nuisance event classes inside nominal operation**, not the benchmark's invariance/equivariance axes, and the monitor is vision-only ([§4.2](https://arxiv.org/html/2606.16690v1)). ContactGuard evaluates four tasks on one robot, no shift suite ([§4.1](https://ar5iv.labs.arxiv.org/html/2608.13438)). No tactile benchmark defines any monitor metric at all ([Bench2Dex](https://arxiv.org/html/2609.15726v1), [SoftVTBench](https://arxiv.org/html/2608.18701v1), [ManiFeel](https://arxiv.org/html/2505.18472v2), [TacO](https://arxiv.org/html/2605.21976v1)). Searches for monitor-robustness protocols in tactile manipulation returned nothing.
- **Why it matters:** a monitor is a classifier under shift; the entire evaluation apparatus the field built for policies (inv vs equi channels) has never been pointed at one.

---

## 4. Best unclaimed axis + how hard it is to measure

**Recommendation — the axis to claim:**

> **Calibrated, lead-time-resolved tactile failure monitoring under the invariance/equivariance shift suite, reported per embodiment and with a cross-embodiment held-out split.**

This is the union of §3.1 + §3.2 + §3.7 (+ optionally §3.3/§3.4), i.e. an *evaluation protocol for monitors* that does for tactile monitoring what Bench2Dex did for policies. It is attractive because:
- the pieces already exist on the substrate (Bench2Dex has 12 hands, stage predicates → failure onsets, and the inv/equi suite);
- no tactile paper reports any monitor metric, and no monitor paper reports calibration or lead time;
- it is a *measurement* contribution — exactly what a strong paper needs — and it does not require winning a policy comparison.

**Concrete metric set to report (and where each comes from):**
| Metric | What it adds | Cheapest source |
|---|---|---|
| FPR on each non-failure shift channel (inv, equi, full) | monitor robustness under shift | reuse Bench2Dex channels |
| TPR / recall at a fixed failure-onset window | detection | stage predicates |
| **Lead time** Δt = t_fail − t_alarm (distribution, plus accuracy vs Δt curve) | *pre-contact* margin, unclaimed | defines t_fail from latched predicates / privileged object state |
| **ECE / reliability curve / Brier** | *calibrated* failure probability, unclaimed | needs probabilistic score, cheap |
| **Risk–coverage / alarm-precision–recall** across thresholds | removes single-threshold cherry-picking | cheap once scores exist |
| Per-hand and held-out-hand FPR/TPR/lead time matrix | cross-embodiment monitor transfer, unclaimed | 12 hands |

**How hard to measure:** **low-to-medium for a single embodiment, medium-to-high for the full claim.**
- *Easy:* calibration + lead time on logged rollouts — no policy training needed. Bench2Dex makes rollouts expensive to generate but gives per-stage latched predicates and privileged object state, so a "doomed/failure onset" label and a probabilistic monitor score are both obtainable offline. ECE/Brier/risk–coverage are then trivial.
- *Hard:* defining **failure onset causally** (the first step after which no recovery is possible) rather than terminal failure — this is the main methodological risk, and the reason lead time is not already reported.
- *Hard:* **cross-embodiment held-out monitor transfer** — the tactile maps differ per hand, and no one has shown a monitor transfers across hands at all; this is the ambitious half of the claim and could legitimately come back negative (a rigorous negative is still publishable here).
- *Caveat:* Bench2Dex's tactile signal is geometric penetration depth, not force/shear; a monitor built on it measures a *proxy*, which the benchmark itself flags. State this explicitly.

---

## 5. What reviewers currently demand (evidence)

1. **Error bars / statistical significance with correctly defined variability.** NeurIPS' paper checklist item 7 is explicit: "Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?", with guidance that the source of variability and the method of computing error bars must be stated ([NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)). PATCH and ContactGuard already report "mean ± std over 10 seeds" / over 5 seeded 5-fold CV runs.
2. **A principled, tightly bounded evaluation under a small rollout budget.** Vincent et al., *How Generalizable Is My Behavior Cloning Policy? A Statistical Approach to Trustworthy Performance Evaluation* ([arXiv 2405.05439](https://arxiv.org/abs/2405.05439)) provides a CDF lower bound with user-specified confidence/tightness from a minimal number of rollouts, and compares policies in OOD settings. This is the strongest citable argument that "success rate over N rollouts" is insufficient and that **budget-aware, distribution-aware bounds** are the modern standard.
3. **Reproducibility as a first-class axis.** Bench2Dex ships a **reproducibility metadata** section (§E.3) alongside its metrics ([arXiv 2609.15726](https://arxiv.org/html/2609.15726v1)); NeurIPS checklist items 4–6 (reproducibility, code/data, full experimental setting) are gated in practice ([checklist](https://neurips.cc/public/guides/PaperChecklist)). ManiFeel's headline pitch is "reproducible and scalable" ([arXiv 2505.18472](https://arxiv.org/html/2505.18472v2)).
4. **Honest limitations + explicit scope.** NeurIPS checklist item 2 and the 2024 reviewer guidelines both instruct reviewers not to penalise honest limitations, and to probe "the factors that influence the performance" ([Reviewer Guidelines](https://neurips.cc/Conferences/2024/ReviewerGuidelines), [checklist](https://neurips.cc/public/guides/PaperChecklist)). This is why Bench2Dex's own caveat ("does not assume simulated tactile replaces real tactile") reads as a strength; the parent survey's note that you must match execution-prefix length in no-tactile baselines is the kind of confound reviewers now look for (see SoftVTBench's gripper-granularity control and ManiFeel's modality-isolation design).
5. **Best-paper bar is "excellent evaluation, reproducibility, resources".** NeurIPS' overall-score rubric ties 7–10 explicitly to "good-to-excellent evaluation, resources, and reproducibility" ([Reviewer Guidelines](https://neurips.cc/Conferences/2024/ReviewerGuidelines)).

---

## 6. Known data-quality problems in released tactile datasets

1. **No released tactile dataset is fully annotated/verified; annotation is being retrofitted.** TLabel is pitched as "the world's first sensor-agnostic tactile data annotation toolkit" and exists because existing data lacks a unified schema and quality assessment ([GitHub](https://github.com/liesliy/tlabel); papers: *TLabel: A Unified Annotation Framework for Cross-Sensor Tactile Manipulation Data* and *A Python library for standardized tactile data annotation and quality assessment*, linked from the repo). A concrete reported defect class: in a PaXini-based set, **all 301 "soft anomalies" were `phase_skip` events caused by human-like fragmented manipulation** — i.e. the labels, not the policy, were wrong (search-indexed snippet of the TLabel paper; full PDF could not be fetched, see §7).
2. **Tactile heterogeneity / missing modalities are unresolved.** HRDexDB states tactile specs "are not uniform across different robotic platforms", tactile is **absent** for its human-hand and Allegro sequences, and calls unification "future work" ([arXiv 2604.14944](https://arxiv.org/abs/2604.14944), via parent survey).
3. **Device/label quality**: a public forum thread specifically documents "Sensor noise filtering and contact label quality for builders/researchers" on the Orca hand — evidence that contact-label noise is a known practitioner problem, not a solved one ([roboticscenter.ai](https://www.roboticscenter.ai/es/forum/posts/orca-hand-sensor-noise-filtering-and-contact-label-quality-for-builders-researchers-advanced)).
4. **Simulated-vs-real tactile distribution gap is quantified, not eliminated.** ManiFeel reports FID ≈ 3–7 and KID ≈ 2.5–7.4 ×10⁻² between simulated and real tactile images ([Table I](https://arxiv.org/html/2505.18472v2)); Bench2Dex states its tactile interface "does not reproduce the output of any specific physical sensor" ([abstract](https://arxiv.org/abs/2609.15726)).
5. **Quantised tactile representations can destroy the signal that matters.** The Bench2Dex repo's own design review (2026-09-08) notes the tactile field is CPD-filtered normal geometric distance + contact mask, **cannot be claimed to measure normal force, shear, or friction**, and pools 240×240 to 2×2, attenuating sparse contact (via parent survey, `TACTILE_DESIGN_REVIEW.md`).
6. **Release/infrastructure gaps.** Bench2Dex's HF/ModelScope checkpoint repos could not be fetched from this sandbox (HTTP blocked / client challenge) — so the contents and any released-data defects remain unverified. See §7.

---

## 7. What I could not verify

- **HuggingFace / ModelScope dataset and checkpoint contents.** The TLabel PyPI page and GitHub raw PDFs returned client-challenge/404, and the earlier Bench2Dex survey already recorded HF/ModelScope as unreachable (HTTP 000). Therefore: (a) whether Bench2Dex's released tactile stream has documented defects, and (b) whether tactile checkpoints are actually released, are **unverified**.
- **TLabel full paper text.** Only repo metadata + search-indexed snippets (the "301 phase_skip" figure) were obtainable; the exact dataset(s), counts and methodology could not be read. Treat §6.1 as ⚠️.
- **Bench2Dex per-hand results table.** The HTML fetch truncated mid-§4/Appendix E. I read §3.5 and the appendix TOC (which lists primary, diagnostic, safety, grasp, tool-use, and reproducibility metrics) but did not see the full metric definitions or any per-hand matrix, so "no per-embodiment breakdown" is a read-level absence, not a proof.
- **Position: Benchmarking is Broken — Don't Let AI Be Its Own Judge** (ODU Digital Commons) returned HTTP 403; I could not use it as a reviewer-culture citation.
- **Full text of roto 2.0, HRDexDB, EgoTactile, RCT, HT-Bench, TactiDex, TacMap, HumanTouch, Tabero.** These are included at ⚠️/abstract or via the parent survey's verified reads; their rows should be re-checked against full text before being cited in a paper.
- **Venue status** of several 2026 items (SoftVTBench, TacO, ManiFeel v2, BiView-Touch = ICRA 2027 submission) is as reported by the authors/arXiv comments, not confirmed via proceedings pages. OOPSIEVERSE's **RSS 2026** acceptance is stated in its arXiv comment; Bench2Dex is a technical report.
- **Exhaustiveness.** arXiv full-text search and Semantic Scholar indexing lag for Sept–Oct 2026 papers; Chinese-venue and non-arXiv tactile benchmarks may be missed. The field moved from ~3 to ~20+ papers Jun→Sep 2026 (parent survey), so a pre-submission re-check is required.

---

### Key URLs
[Bench2Dex](https://arxiv.org/abs/2609.15726) · [Bench2Dex HTML](https://arxiv.org/html/2609.15726v1) · [SoftVTBench](https://arxiv.org/html/2608.18701v1) · [ManiFeel](https://arxiv.org/html/2505.18472v2) · [TacO](https://arxiv.org/html/2605.21976v1) · [roto 2.0](https://arxiv.org/abs/2605.21429) · [TacBench/Sparsh](https://arxiv.org/abs/2410.24090) · [TacSL](https://arxiv.org/abs/2408.06506) · [HRDexDB](https://arxiv.org/abs/2604.14944) · [EgoTactile](https://arxiv.org/abs/2606.09243) · [RCT](https://arxiv.org/abs/2606.31694) · [HT-Bench](https://arxiv.org/abs/2606.19161) · [TactiDex](https://arxiv.org/abs/2607.09190) · [TacMap](https://arxiv.org/abs/2602.21625) · [BiView-Touch](https://arxiv.org/abs/2609.23352) · [PATCH](https://arxiv.org/html/2606.16690v1) · [ContactGuard](https://ar5iv.labs.arxiv.org/html/2608.13438) · [OOPSIEVERSE](https://arxiv.org/abs/2606.31993) · [RoboMD](https://arxiv.org/abs/2412.02818) · [statistical eval](https://arxiv.org/abs/2405.05439) · [TLabel](https://github.com/liesliy/tlabel) · [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) · [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2024/ReviewerGuidelines)
