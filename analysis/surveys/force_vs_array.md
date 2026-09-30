# Focused novelty check: within-subject FORCE vs TACTILE-ARRAY comparison (as of 2026-09-30)

Question: has anyone published a **controlled, same-hand / same-fingertip, same-physical-interaction** comparison of 3/6-axis force sensing vs a high-resolution tactile array (image / optical / dense depth) for **task-level** quantities (slip, failure onset, grasp stability, outcome)?

## 1. Table of found works

| work | venue/year | what compared | same hardware? | task-level or sensor-level? |
|---|---|---|---|---|
| [Tactile Genesis](https://arxiv.org/abs/2606.22332) | arXiv 2606.22332 (Jun 2026) | Ablates **sensor type**: binary contact, contact depth, **per-taxel kinematic F/T**, marker displacement, proximity, audio, temperature — one interface, 3 dexterous tasks on one robot (XHand). Finds per-taxel F/T "consistently the most useful sensor type"; placement > type; resolution << coverage | **Yes, in simulation** (identical robot/physics/tasks); real transfer only for the winning config | **Task-level** — but modality signal is simulated per-taxel, not a real 3-axis F/T vs a real tactile camera |
| [Data-Driven Optimization of Tactile Sensor Configurations](https://arxiv.org/abs/2409.20473) | arXiv 2409.20473 v3 (May 2026) | Sensor **count/placement** sufficiency on Shadow Hand: 92→21→14 taxels keep >90% performance; middle-finger taxels hurt; cross-platform Allegro/Leap | **Yes** (same hand, same tasks) | Task-level, but **within one modality (tactile taxels only)** — no force-vs-array contrast |
| [TacVerse](https://arxiv.org/abs/2606.25877) | arXiv 2606.25877 (Jun 2026) | 7 vision-based tactile sensors: shape, grating, force regression; within-sensor, zero-shot cross-sensor, few-shot | **No** — one sensor per collection; sensor shift confounded with object/interaction variation | Task-level but **no force modality at all** |
| [ME-Dex 1.0](https://arxiv.org/abs/2609.21449) | arXiv 2609.21449 (Sep 2026) | Heterogeneous tactile layouts + sim force sensors mapped into one latent space (world-action model) | **No** — heterogeneity is *engineered away*, never ablated as an experimental variable | Task-level |
| [Gel-OPTOFORT](https://www.semanticscholar.org/paper/30a6c63f2edb1538bcb5310aa61d1a452ec6e2ee) / [PDF](https://bura.brunel.ac.uk/bitstream/2438/29357/4/FullText.pdf) | Noh et al., ~2024, IEEE Sensors-style | **One sensor** = GelSight geometry + optoelectronic **multi-axis F/T** simultaneously | **Yes — co-located in one fingertip device** | **Sensor-level** (design + characterisation); no task-level contest |
| [uSkin (XELA Robotics)](https://www.xelarobotics.com/) | product | Dense array in which **each taxel is a 3-axis force** measurement | One device that is simultaneously "3-axis force" and "array" | Sensor-level; the proposed contrast is not expressible on this hardware |
| [PaXini PX-6AX-GEN3 + DexH13](https://www.koreaherald.com/article/10651060) | vendor PR, CES Jan 2026 | Claims one fingertip sensor measures **6-axis force + texture + elastic response** (15 dimensions); hand has 1,140 ITPU units | **Yes (vendor claim, single package)** | No published controlled task-level comparison |
| [Fingertip-scale six-axis tactile interface](https://link.springer.com/article/10.1038/s41378-026-01292-3) | Microsyst. Nanoeng. 2026 | 6-axis force + contact position localisation at fingertip | Yes (one device) | Sensor-level |
| [Making Sense of Vision and Touch](http://ai.stanford.edu/~yukez/papers/icra2019.pdf) | ICRA 2019 | Modality ablation vision-only / touch-only / both, peg insertion + cloth smoothing | Same robot, but "touch" is a **low-dimensional force/contact signal, not an array** (sensor type not re-verified here) | Task-level ablation — closest *methodological* ancestor, wrong touch modality |
| [Hoi! force-grounded dataset](https://openaccess.thecvf.com/content/CVPR2026/papers/Engelbracht_Hoi_-_A_Multimodal_Dataset_for_Force-Grounded_Cross-View_Articulated_Manipulation_CVPR_2026_paper.pdf) | CVPR 2026 | Force-grounded cross-view articulated manipulation data | Force only | Task-level; no tactile array |
| [F-TAC Hand / high-res touch across hands](https://arxiv.org/abs/2412.14482) | Nat. Mach. Intell. 2025 | 17 high-res tactile sensors, ~70% hand coverage, human-like grasping | Same hand | Task-level; **no force modality, no ablation** |

## 2. ALREADY DONE (URLs)
- **Co-located 6-axis F/T + optical tactile in one fingertip sensor**: [Gel-OPTOFORT](https://bura.brunel.ac.uk/bitstream/2438/29357/4/FullText.pdf) — but used for sensor characterisation, not task-level comparison.
- **Dense array whose taxels ARE 3-axis force**: [XELA uSkin](https://www.xelarobotics.com/) (and PaXini's claimed [15-dimension fingertip sensor](https://www.koreaherald.com/article/10651060)).
- **Controlled sensor-type ablation, task-level, everything else held fixed**: [Tactile Genesis](https://arxiv.org/abs/2606.22332) — in simulation, and its own motivation states the real-world version is impossible today: *"each sensor effectively defines a new robot, and no lab can replicate the same learning experiment across all of them."* That sentence is the confound the proposed study removes.
- **Sensor-sufficiency within one modality on real hardware**: [arXiv 2409.20473](https://arxiv.org/abs/2409.20473) (how many taxels are enough, where).
- **Cross-sensor tactile benchmark**: [TacVerse](https://arxiv.org/abs/2606.25877).
- **Modality ablation incl. force**: [Lee et al. ICRA 2019](http://ai.stanford.edu/~yukez/papers/icra2019.pdf).

## 3. NOT DONE (absence evidence)
- No paper found that mounts a **3/6-axis F/T sensor and a dense tactile array on the same fingertip** and reports which one predicts **task-level** events (slip onset, failure, stability, outcome) on **identical contacts** — and none reports their *complementarity* (what one sees that the other cannot).
- Cross-sensor benchmarks ([TacVerse](https://arxiv.org/abs/2606.25877)) collect **separate data per sensor**, so sensor identity is confounded with objects/trajectories/contacts.
- Heterogeneous-tactile papers ([ME-Dex 1.0](https://arxiv.org/abs/2609.21449)) treat modality differences as a nuisance to be normalised, not as the experimental variable.
- Sufficiency work is either simulation-only ([Tactile Genesis](https://arxiv.org/abs/2606.22332)) or single-modality ([2409.20473](https://arxiv.org/abs/2409.20473)).
- Vendor/blog-side framing exists ([Force-Torque Sensors vs Tactile Arrays for Grasp Feedback](https://inferensys.com/differences/physical-ai-and-humanoid-robotics-software/sensor-fusion-and-perception-pipelines/force-torque-sensors-vs-tactile-arrays-for-grasp-feedback)) — engineering commentary, not a controlled study.

## 4. Closest study and the required differentiation
**Closest overall**: [Tactile Genesis](https://arxiv.org/abs/2606.22332) (same robot, same tasks, sensor-type ablation, task-level, "F/T per taxel wins") — **simulation**, per-taxel signal families, no real tactile camera and no real 3-axis F/T.
**Closest hardware**: [Gel-OPTOFORT](https://bura.brunel.ac.uk/bitstream/2438/29357/4/FullText.pdf) (real co-located 6-axis F/T + optical tactile, sensor-level only).
**Closest methodology**: [Lee et al. ICRA 2019](http://ai.stanford.edu/~yukez/papers/icra2019.pdf) (modality ablation on one robot, but touch = low-dim force, no array).
A new study must therefore contribute: **(i) real hardware**, both modalities **streaming simultaneously from the identical contact** on one dexterous hand/fingertip; **(ii) task-level targets** (slip/failure onset, stability margin, outcome), not sensor metrics; **(iii) a sufficiency/complementarity claim** ("which sensor is enough for task X; what does each add") instead of a ranking; **(iv) the within-subject design itself as the result** — quantifying how much of the reported cross-hardware modality gap in the literature is sensor confound.

## 5. What I could NOT verify
- Whether Gel-OPTOFORT or any successor ever ran manipulation tasks and reported task-level comparisons across its two channels (paywalled IEEE/Scilit record; only abstract/PDF snippet seen).
- Any 2024-2026 ICRA/CoRL/RA-L paper doing a force-vs-array ablation on real hardware — IEEE PDFs were not fetched (paywall); absence here is search-based, not exhaustive.
- The exact touch modality in Lee et al. ICRA 2019 (no PDF fetch); treat "low-dim force, not array" as provisional.
- Whether PaXini's fingertips truly deliver a *dense high-resolution array* co-located with 6-axis force (vendor PR only), and whether XELA's uSkin offers a separate dense-geometry channel alongside its 3-axis taxels.
- Semantic Scholar API returned no rows for the force-vs-tactile query (rate-limit/empty), so citation-graph coverage of this question is incomplete.
