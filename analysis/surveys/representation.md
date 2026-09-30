# Tactile REPRESENTATION and TRANSFER — focused literature survey

Date: 2026-09-30. Angle: tactile representation & transfer for a mechanism-level (ACT/Diffusion-Policy ambition) contribution.
Scope: (a) cross-embodiment / cross-hand transfer & zero-shot held-out morphology; (b) cross-sensor transfer (optical arrays / taxels / simulated penetration-depth) & sim-to-real; (c) unified tactile representations across sensor types & resolutions; (d) canonicalization / equivariance / invariance; (e) touch foundation models / large-scale pretraining.

Verification rule used: every entry below was read from a fetched abstract/project page (not a snippet). Claims I could not confirm to primary text are listed in §5.

---

## 1. Table — what transfers across what

| Work | Venue / date | Exactly what transfers across what | HELD-OUT morphology tested? | HELD-OUT sensor tested? | Evaluation scale |
|---|---|---|---|---|---|
| [ME-Dex 1.0](https://arxiv.org/abs/2609.21449) | arXiv 2609.21449, 18–21 Sep 2026 (Li Auto) | Tactile force-fields from different embodiments + sensing layouts → **Canonical Hand Model** (finger/palm region slots, unobserved regions masked) + **Unified Tactile Autoencoder** shared latent; joint video/tactile/action flow matching (MoM) | Not explicitly: trains across grippers + dexterous hands it later evaluates. No unseen-hand zero-shot protocol in abstract/intro/method | No: same sensor set mapped to canonical regions | RoboTwin, DexJoCo, ManiFeel sim + real SO-101 (Paxini PX6AX) + Xynova Flex2 hands |
| [UniTacHand](https://arxiv.org/abs/2512.21233) | arXiv 2512.21233, Dec 2025 | Human-glove touch ↔ robot dexterous-hand touch → both projected onto **MANO 2D hand surface**, contrastively aligned into one latent (10 min paired data) | Human hand → one robot hand (the alignment target); not a held-out robot morphology | No (single robot hand) | 10 min paired data; zero-shot to real robot, unseen objects; human+robot co-training |
| [TactAlign](https://arxiv.org/abs/2602.13579) | arXiv 2602.13579, RSS 2026 | Human-collected tactile → robot of **different embodiment** via shared latent learned with **rectified flow**, no paired data (hand–object pseudo-pairs) | Human vs robot embodiment gap; no held-out robot hand | No | 3 contact-rich tasks (pivoting, insertion, lid closing) + zero-shot light-bulb screwing; <5 min human data |
| [DexGrasp-Zero](https://arxiv.org/abs/2603.16806) | arXiv 2603.16806, RSS 2026 | Grasping skill across hands via **morphology-aligned graph** (anatomically grounded nodes + tri-axial orthogonal motion primitives) + MAGCN + physical-property injection | **Yes — zero-shot to unseen hands** (LEAP, Inspire) after training on Allegro/Shadow/Schunk/Ability | n/a — **no tactile sensing** (kinematic/proprioceptive) | 85% zero-shot sim on YCB; 82% real on LEAP/Inspire/Revo2 |
| [DexFormer](https://arxiv.org/abs/2602.08278) | arXiv 2602.08278, Feb 2026 | Cross-embodiment dexterous manipulation via history-conditioned transformer; morphology inferred on the fly | **Yes — zero-shot to Leap/Allegro/Rapid**, trained on procedurally generated hand assets | n/a — no tactile | Sim, 3 unseen hands |
| [Morphometric Imitation](https://arxiv.org/abs/2609.28660) | arXiv 2609.28660, Sep 2026 | Human HOI → robot across hand morphologies, **contact-preserving retargeting** + residual RL + visuomotor distillation | Yes (3 hands, retargeting); contact F1 +8–28 pts | n/a — visuomotor, contact from human motion | 3 hands, 10 HOIs; Sharpa 89.3% zero-shot, 300 trials / 30 objects |
| [T3 / FoTa](https://arxiv.org/abs/2406.13640) | CoRL 2024 | 13 sensors × 11 tasks via shared trunk + **sensor-specific encoders** + task decoders; zero-shot in *some* sensor–task pairings, else finetune | No | Partial: zero-shot in certain pairings; unseen-sensor pairing not systematic | FoTa 3M points, 13 sensors, 11 tasks; sub-mm insertion +25% success |
| [FTP-1](https://arxiv.org/abs/2606.13102) | arXiv 2606.13102, Jun 2026 (Tsinghua/Sharpa) | Image/array/state tactile → **Morphology-Aware Tactile Token Space (MTTS)** + functional-area embeddings → shared tactile expert; pretrain ~3000 h, 26 sources, 21 sensors (7 image/5 array/9 state) | Claims embodiment-transferable; "unseen platforms" reported, but held-out *morphology* not isolated from held-out sensor | **Yes — Xense (image) and Contactile (array)**, but by **fine-tuning a new sensor-specific encoder** while reusing the expert (not zero-shot) | 5 seen hardware configs (+17.2% real, +17.5% UniVTAC); 3 unseen-sensor tasks +31.6% (46.6 vs 15.0) |
| [UniForce](https://arxiv.org/abs/2602.01153) | arXiv 2602.01153, Feb 2026 | GelSight / TacTip / uSkin → shared **latent force space** via inverse (image→force) + forward (force→image) dynamics constrained by force equilibrium + reconstruction; sensor–object–sensor pairs avoid external F/T | No | Yes across 3 sensor families; encoder "plugs in with zero-shot transfer, no retraining/finetuning" | 3 sensor types, force estimation + Vision-Tactile-Language-Action wiping |
| [GenForce](https://arxiv.org/abs/2503.01058) | arXiv v3 Sep 2025; Nature Communications 2026 | Force model transfers across **homogeneous sensors with varying configurations and heterogeneous modalities/materials** via shared marker representations | No | Yes (transfer to sensors without fresh force data) | Grasping, slip detection/avoidance |
| [SITR (Sensor-Invariant Tactile Representation)](https://arxiv.org/abs/2502.19638) | ICLR 2025 | Transformer trained on **diverse simulated sensor designs** → zero-shot to **new real optical sensors** with minimal calibration | No | **Yes — zero-shot unseen real sensors** (optical only) | SITR dataset; multiple tactile applications |
| [AnyTouch](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html) / [AnyTouch 2](https://arxiv.org/abs/2602.09617) | ICLR 2025 / ICLR 2026 | Unified static+dynamic representation across multiple optical tactile sensors; AnyTouch 2 adds force-aware dynamic perception + ToucHD hierarchy | No | Yes across optical sensors/tasks | ToucHD dataset; static + dynamic benchmarks + real manipulation |
| [UniTouch](https://arxiv.org/abs/2401.18084) | CVPR 2024 | Vision-based tactile → aligned to pretrained image embeddings (vision/language/sound) + **learnable sensor-specific tokens** for heterogeneous sensors | No | Yes (multi-sensor joint training; zero-shot tasks) | Grasping prediction, touch VQA, zero-shot |
| [Sparsh-X / Tactile Beyond Pixels](https://arxiv.org/abs/2506.14754) | arXiv 2506.14754, Jun 2025 (Meta/CMU) | Fuses image+audio+motion+pressure from **Digit 360** into one SSL representation | No | No (single sensor family) | ~1M interactions; +63% policy success, +90% robustness, +48% physical-property accuracy |
| [CTSRL](https://icml.cc/virtual/2026/poster/66793) | ICML 2026 | Sensor-agnostic representation via **Cross-Sensor Modulator** (removes sensor-specific bias) + 2-stage: aligned synthetic cross-sensor SSL, then real multimodal alignment | No | Yes — "generalization to unseen sensors" | Multi-sensor generalization experiments |
| [BIDETA](https://arxiv.org/abs/2609.08673) | arXiv 2609.08673, Sep 2026 | Gradient-free **few-shot test-time adaptation** to unseen sensors: frozen encoder + support memory + spectral graphs + reliability-gated recurrence | No | Yes — needs ~10% *labeled* target contacts | SITR / TacVerse-Shape / TacQuad: SITR Sparsh 6.86%→87.09%; ~20× faster adaptation |
| [TacVerse](https://arxiv.org/abs/2606.25877) | arXiv 2606.25877, Jun 2026 | Benchmark: within-sensor, **zero-shot cross-sensor**, few-shot adaptation over 7 VBTS | No | Yes (7 sensors) | 106,800 images; shape/grating classification + force regression. Direct cross-sensor transfer degrades substantially; MAE pretraining best |
| [Tacmap](https://arxiv.org/abs/2602.21625) | arXiv 2602.21625, Feb 2026 | Sim↔real unified through **geometry-consistent penetration-depth deform map** (sim 3D intersection volumes ↔ learned real image→depth map) | No | No — same sensor sim/real | Zero-shot sim→real in-hand rotation |
| [Norm2Tex](https://arxiv.org/abs/2609.25398) | arXiv 2609.25398, Sep 2026 | Visuo-tactile sim augmented with normal-map high-frequency texture; plug-in to different tactile simulators | No | No | Material classification + RL sim-to-real |
| [EquiContact](https://arxiv.org/abs/2507.10961) | arXiv v4 Jan 2026 (submitted RSS) | **SE(3)-equivariant** vision→force pipeline (Diff-EDF planner + G-CompACT), actions in EE frame, localized GCEV + F/T | No | No (F/T + wrist camera) | Peg-in-hole, screwing, wiping; near-perfect success, unseen spatial configs |
| [TacSE3](https://arxiv.org/abs/2605.17929) | arXiv 2605.17929, May 2026 | Low-texture visuotactile → decoupled 3D force field → **incremental SE(3) motion** (translation from contact centroid, rotation from shear) | No | No (paired DM-Tac fingertips) | In-gripper tracking + compensation improves downstream disturbance tolerance |
| [N0-TWAM](https://arxiv.org/abs/2607.23783) | arXiv 2607.23783, Jul 2026 | Large-scale tactile-native world-action model; **NeoForce** unified force-based representation; 6 embodiments, 450 tasks | Not stated | Not stated | Real + simulated contact-rich benchmarks; data-scaling study |
| [Semantic-Contact Fields](https://roboticsconference.org/2026/program/papers/4/) | RSS 2026 | Unified 3D representation fusing visual semantics + dense contact; 2-stage sim-to-real contact learning | No | No | Category-level generalization to **unseen tools** (scraping, drawing, peeling) |
| [Contact-Grounded Policy](https://roboticsconference.org/2026/program/papers/5/) | RSS 2026 | Predicts coupled robot-state + tactile trajectories, then contact-consistency mapping to a compliance controller | No | No | Allegro V5 + Digit360 real; simulated Tesollo DG-5F with dense whole-hand arrays |
| [TacBPM](https://arxiv.org/abs/2609.18174) | arXiv 2609.18174, Sep 2026 | Tactile-conditioned behavior prior (multi-scale specialists → latent controller), residual latent actions | No | No | In-hand reorientation + Grasp-to-AnyPose; sim-to-real |

---

## 2. CROWDED — mechanism-level ideas now taken

**C1. Per-sensor encoder + shared trunk/expert.** One shared transformer with sensor-specific front ends (and often sensor-specific tokens).
- T3: shared trunk, sensor-specific encoders [arXiv 2406.13640](https://arxiv.org/abs/2406.13640)
- FTP-1: heterogeneous encoders → unified token space [arXiv 2606.13102](https://arxiv.org/abs/2606.13102)
- UniTouch: learnable sensor-specific tokens [arXiv 2401.18084](https://arxiv.org/abs/2401.18084)
- UniForce: per-sensor adapters into a latent force space [arXiv 2602.01153](https://arxiv.org/abs/2602.01153)

**C2. Hand-indexed canonical template.** Map every tactile surface to a semantic hand region (or MANO UV), pool per region, mask unobserved regions.
- ME-Dex **Canonical Hand Model** (finger/palm region slots + masked learned pooling) [arXiv 2609.21449](https://arxiv.org/abs/2609.21449)
- UniTacHand MANO-2D surface unification [arXiv 2512.21233](https://arxiv.org/abs/2512.21233)
- FTP-1 **functional-area embeddings** [arXiv 2606.13102](https://arxiv.org/abs/2606.13102)

**C3. Physical-intermediate grounding.** Force/deformation/penetration-depth as the shared currency between sensors.
- UniForce (force equilibrium, image↔force) [arXiv 2602.01153](https://arxiv.org/abs/2602.01153)
- GenForce (shared marker representation for force) [arXiv 2503.01058](https://arxiv.org/abs/2503.01058)
- Tacmap (shared deform/penetration-depth map for sim-to-real) [arXiv 2602.21625](https://arxiv.org/abs/2602.21625)
- Semantic-Contact Fields (dense contact field) [RSS 2026](https://roboticsconference.org/2026/program/papers/4/)
- Norm2Tex (depth-map texture augmentation) [arXiv 2609.25398](https://arxiv.org/abs/2609.25398)

**C4. Unpaired latent alignment across embodiments.** Contrastive / rectified-flow latent transport between human and robot touch, no paired data.
- UniTacHand (contrastive, 10 min paired) [arXiv 2512.21233](https://arxiv.org/abs/2512.21233)
- TactAlign (rectified flow, pseudo-pairs) [arXiv 2602.13579](https://arxiv.org/abs/2602.13579)

**C5. Simulated sensor-design randomization → real sensor.** Train over many simulated sensor geometries so the encoder generalizes to unseen real hardware.
- SITR (ICLR 2025) [arXiv 2502.19638](https://arxiv.org/abs/2502.19638)
- CTSRL (stage-1 aligned synthetic cross-sensor SSL) [ICML 2026](https://icml.cc/virtual/2026/poster/66793)
- Tacmap / Norm2Tex for the sim-to-real half [2602.21625](https://arxiv.org/abs/2602.21625), [2609.25398](https://arxiv.org/abs/2609.25398)

**C6. Few-shot / gradient-free adaptation to an unseen sensor.** Cheap target-side adaptation instead of zero-shot.
- BIDETA (frozen encoder + few labeled contacts) [arXiv 2609.08673](https://arxiv.org/abs/2609.08673)
- FTP-1 (train only the new sensor encoder) [arXiv 2606.13102](https://arxiv.org/abs/2606.13102)
- T3 (finetune per sensor–task pair) [arXiv 2406.13640](https://arxiv.org/abs/2406.13640)

**C7. Equivariance in the contact/action frame.** SE(3) equivariance from perception to force control, EE-frame actions.
- EquiContact [arXiv 2507.10961](https://arxiv.org/abs/2507.10961)
- TacSE3 (SE(3) motion from tactile) [arXiv 2605.17929](https://arxiv.org/abs/2605.17929)

**C8. Morphology-aligned graph / anatomical abstraction for held-out hands (no touch).** Already solved for grasping without tactile.
- DexGrasp-Zero (unseen hands, 85% zero-shot) [arXiv 2603.16806](https://arxiv.org/abs/2603.16806)
- DexFormer (procedural hand assets → zero-shot) [arXiv 2602.08278](https://arxiv.org/abs/2602.08278)
- Morphometric Imitation (contact-preserving retargeting) [arXiv 2609.28660](https://arxiv.org/abs/2609.28660)

**C9. Tactile world-action models at scale.** Predict future touch + vision + action jointly.
- ME-Dex 1.0 [arXiv 2609.21449](https://arxiv.org/abs/2609.21449)
- N0-TWAM [arXiv 2607.23783](https://arxiv.org/abs/2607.23783)
- (VT-WAM, TouchWorld, OmniVTA, ViTacWorld cited as prior art inside ME-Dex §2.3 — not independently fetched)

**C10. Cross-sensor benchmarks/datasets.** The evaluation substrate is being built now, so "we introduce a cross-sensor benchmark" is no longer a contribution by itself.
- TacVerse (7 sensors, 106.8k images) [arXiv 2606.25877](https://arxiv.org/abs/2606.25877)
- SITR dataset [arXiv 2502.19638](https://arxiv.org/abs/2502.19638); FoTa [arXiv 2406.13640](https://arxiv.org/abs/2406.13640); ToucHD [arXiv 2602.09617](https://arxiv.org/abs/2602.09617)

**Bottom line on crowding:** cross-sensor unification via per-sensor encoders + a shared latent (or a physical intermediate) is saturated in 2025–2026; "unified tactile representation" alone is no longer a defensible mechanism-level novelty.

---

## 3. OPEN — precisely stated unoccupied mechanism-level ideas

**O1. Contact-frame (gauge) canonicalization: hand-indexed → contact-indexed representation.**
Change the canonicalization anchor. Today every unified representation is *hand-indexed* (map sensor → finger/palm region or MANO UV). Instead, make the representation a function of the contact stress/deformation field expressed in a **self-estimated local contact frame**: normal n from the net force/deformation field; tangent basis from shear orientation / principal curvature; patch centered on the contact centroid. Encode only the field in that frame (invariant to which finger, which sensor, its mounting, and resolution) and carry the contact pose in the hand/object frame as a **separate equivariant token** fed to the policy.
- Invariant to: hand identity, finger identity, sensor model, resolution, taxel ordering, mounting offset.
- Equivariant to: contact pose SE(3), shear direction; force magnitude must remain *variant* (a representation that loses force magnitude is broken).
- Test: train on K hands × M sensors; evaluate **zero-shot on a held-out hand AND a held-out sensor simultaneously** (double shift), with no target encoder and no target fine-tuning. Baselines: ME-Dex, FTP-1, UniForce, T3, SITR.
- Why open: ME-Dex/UniTacHand/FTP-1 canonicalize to hand anatomy; UniForce/GenForce canonicalize to force magnitude but keep contact *location* implicit in the sensor layout, so they still require a per-sensor encoder/adapter.

**O2. Active sensor-operator identification instead of passive adaptation.**
A new sensor's unknown measurement operator (raw reading → local force/deformation field) is identified *online* from a small budget of **self-chosen probe actions on a known-geometry touchstone**, selected to maximize identifiability (e.g., maximize Fisher information / predicted disagreement across candidate operators), then the frozen policy runs. Distinct from BIDETA (passive, needs labeled target contacts), FTP-1 (trains a new encoder), SITR (needs calibration), and classical weighted calibration.
- Test: unseen sensor, fixed probe budget N ∈ {5, 20, 100}; report success vs N and compare with BIDETA/FTP-1 at matched target data.

**O3. A stated *held-out-hand* tactile transfer protocol.**
No verified work reports zero-shot tactile transfer to a morphology never seen in training ([DexGrasp-Zero](https://arxiv.org/abs/2603.16806) does held-out hands but without touch; [ME-Dex](https://arxiv.org/abs/2609.21449) maps many hands but evaluates seen ones; [UniTacHand](https://arxiv.org/abs/2512.21233)/[TactAlign](https://arxiv.org/abs/2602.13579) cross the *human→robot* gap only). A protocol "train 4 hands, hold out 2, no target data" plus the measured morphology gap is itself a missing artifact and a natural home for O1.

**O4. A tactile invariance/equivariance test suite with a formal spec.**
Define the transformation group and, for each element, whether the representation *must* be invariant (sensor substitution, resolution resample, taxel permutation, mounting offset/rotation, gain/bias, dead-zone nonlinearity) or *must* be equivariant (contact-frame rotation/translation, shear direction), plus a metric that penalizes **wrong invariance** (e.g., a representation invariant to force magnitude). Existing cross-sensor benchmarks ([TacVerse](https://arxiv.org/abs/2606.25877)) measure transfer accuracy but do not specify or test these algebraic properties.

**O5. One encoder, no sensor-specific parameters: implicit tactile fields from point samples.**
Represent any sensor as a set of (surface position, surface normal, measurement) samples; encode with a permutation-invariant set transformer with Fourier positional features; decode at arbitrary query resolution. Goal: zero-shot on an unseen sensor **with no new encoder at all** — the gap left by FTP-1 (new encoder), T3 (sensor-specific encoders), UniForce/UniTouch (per-sensor adapters/tokens).
- Risk: [CTSRL](https://icml.cc/virtual/2026/poster/66793) ("Cross-Sensor Modulator" eliminating sensor-specific bias; sensor-agnostic) is close in spirit; needs its full text checked before claiming this.

**O6. Mounting/extrinsic invariance as a first-class transfer axis.**
All canonical models assume the sensor's pose on the finger is known. Learn the sensor→hand extrinsic jointly with the representation so that a sensor *re-mounted* (rotated/shifted, or swapped to a different finger) still transfers. Test: permute/rotate sensor mounts at test time; measure degradation. Not covered by any fetched work.

---

## 4. Best unoccupied opportunity + its biggest prior-art risk

**Best opportunity: O1 + O3 — contact-frame (gauge) canonicalization of touch, evaluated by simultaneous zero-shot transfer to a held-out hand *and* a held-out sensor.**

Why this one: (i) it is an *interface/representation change*, not a new scale or benchmark; (ii) it moves the anchor of canonicalization from the hand to the contact, which is the one axis none of ME-Dex / FTP-1 / UniTacHand / UniForce / SITR / CTSRL occupies; (iii) it is crisply testable — double held-out shift, zero target data, ablate the frame estimator and the equivariant pose token; (iv) it directly attacks the strongest remaining failure mode, since [TacVerse](https://arxiv.org/abs/2606.25877) shows direct cross-sensor transfer still degrades substantially even for force regression.

**Biggest prior-art risk: ME-Dex 1.0's Canonical Hand Model + Unified Tactile Autoencoder** ([arXiv 2609.21449](https://arxiv.org/abs/2609.21449), Li Auto, Sep 2026). It already claims "a shared spatial and latent space" across embodiments and sensing layouts and is from a well-resourced team; a reviewer may read a contact-centered variant as an incremental re-anchoring unless the paper shows the *held-out* double shift that ME-Dex does not report. Secondary risk: [UniForce](https://arxiv.org/abs/2602.01153) / [GenForce](https://arxiv.org/abs/2503.01058) force-grounded latents may behave contact-frame-like in practice, and [CTSRL](https://icml.cc/virtual/2026/poster/66793) already targets "sensor-agnostic" bias removal — so the contribution must be positioned specifically as *frame canonicalization + double held-out protocol*, not as "unified touch representation".

**Physical risk to plan for:** estimating a contact frame from a single tactile patch is ill-posed for near-normal, low-shear contact. A credible design needs multi-finger/whole-hand contact plus object-shape priors, and should report the frame-estimation error separately from downstream success.

---

## 5. What I could not verify

- **ME-Dex held-out-hand claim.** Only abstract, intro and §3 method were retrieved; the experiments (§4.4 "Unified Tactile Representation") and limitations were truncated. Whether any hand is genuinely held out is unconfirmed.
- **FTP-1's held-out *embodiment*.** Read the abstract + project page (incl. result tables). "5 hardware configurations" and "unseen platforms" are reported, but the technical-report PDF was not fetchable, so I could not separate held-out sensor from held-out embodiment.
- **N0-TWAM transfer setup.** Abstract only; the 6 embodiments / 450 tasks are described but no held-out morphology or sensor statement was available.
- **CTSRL full text.** Only the ICML 2026 poster abstract (no arXiv, no PDF) — no data on whether the Cross-Sensor Modulator leaves any sensor-specific parameters, which matters for O5.
- **AnyTouch 2, TacVerse (Wiley), GenForce (Nature Comms) full texts.** ICLR PDF rejected (`application/pdf`); Wiley and Nature returned 403. Verified via the arXiv versions [2602.09617](https://arxiv.org/abs/2602.09617), [2606.25877](https://arxiv.org/abs/2606.25877), [2503.01058](https://arxiv.org/abs/2503.01058) only.
- **Sparsh (the pretrained touch backbone referenced by BIDETA).** Not fetched from primary source; used only as reported inside [BIDETA](https://arxiv.org/abs/2609.08673).
- **RSS 2026 tactile completeness.** The accepted-papers page truncated after paper 56; I verified individual tactile/contact papers by URL (papers 4, 5, 6) but could not enumerate all tactile sessions. [LightTact (RSS 2026 p193)](https://roboticsproceedings.org/rss22/p193.html) was seen by title only.
- **Novelty exhaustiveness for O1/O4/O6.** I searched ~20 phrasings across arXiv, Semantic Scholar, ICML/ICLR/RSS/CoRL pages. The absence of a contact-frame-canonicalization paper is a *search result*, not a proof; a full-text pass over ME-Dex, FTP-1, CTSRL and the RSS 2026 proceedings is still owed before committing.
