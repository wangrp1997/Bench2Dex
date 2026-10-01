# Borrowed playbooks for tactile dexterous manipulation (survey6, 2026-09-30)

Scope: which playbooks from adjacent fields are PROVEN elsewhere but NOT (or only partly) imported into tactile dexterous manipulation.
Occupancy = how crowded the touch literature already is. Value = expected ceiling of a paper that imports it properly.

## 1. Table

| Playbook | Applied to touch? | What is missing | Cost to do properly |
|---|---|---|---|
| (a) FM pretraining, heterogeneous data + unified interface ("web-scale touch") | **Partial.** [Sparsh](http://arxiv.org/pdf/2410.24090) (self-supervised touch reps), [Touch100k](http://arxiv.org.ezproxy.obspm.fr/pdf/2406.03813) (touch-language-vision), [TouchWorld](https://www.semanticscholar.org/reader/393609ffbd1eda3811cf97cdc04e3ce8e2c36bb5), [EgoTac](https://arxiv.org/abs/2608.15060) (5.7M image-tactile pairs from egocentric human video), [TouchThinker](https://arxiv.org/pdf/2606.11637v4), Sparsh-X (CMU): image+audio+motion+pressure backbone | No *policy* FM pretrained across sensor modalities then evaluated on dexterous contact tasks; no held-out-sensor generalization benchmark. Largest corpus is 5.7M *pairs* mined from video, i.e. a distillation proxy — not 10^9 real contacts, and no crawlable substrate. | 1-2 GPU-months + 3-5 sensor rigs; ~6 person-months of plumbing (the hard part is a canonical touch token, not the transformer) |
| (b) Data engine / self-improvement loop (rollout → filter → retrain) | **Partial & rising fast.** [TAMEn](https://ar5iv.labs.arxiv.org/html/2604.07335) (OpenDriveLab: tactile-aware closed-loop data *collection* engine), [DexPIE](https://arxiv.org/abs/2606.09615) (real-world dexterous policy improvement, +37.3% success; verified: DAgger-style intervention + optimality conditioning, **not** tactile-specific), [arXiv 2609.14633](https://export.arxiv.org/pdf/2609.14633) (iterative real-world collection) | TAMEn closes the *collection* loop (autonomy/reset), but nobody closes the *curation* loop: no touch-specific filtering criterion ("keep rollouts with the richest contact-event diversity / highest slip information"), no value-of-information sampling over contact states, loops are 1-2 iterations not hundreds. | High engineering: TAMEn shows the collection half is already built, so the remaining cost is the filter + retrain half — ~2 person-months on top of TAMEn |
| (c) Scaling laws (perf vs data / sensors / compute) | **NOT DONE for control; one precedent for perception.** [EgoTac](https://arxiv.org/abs/2608.15060) reports "both data diversity and volume improve performance steadily" for vision→tactile *prediction* (5.7M pairs) and OOD transfer vs training-data scale; [Zhao MIT thesis](https://dspace.mit.edu/bitstream/handle/1721.1/158785/zhao-alanzhao-phd-meche-2025-thesis.pdf) notes diminishing absolute gains with less data | Everything for *policies*: no controlled sweep over tactile data volume at fixed architecture, none over **sensor count/placement**, none over model size, none fit to an exponent. No tactile Chinchilla; no marginal-value curve to decide "buy sensors vs collect hours". | Cheap and decisive: 3 data tiers x 3 sensor counts x 3 model sizes x 3 seeds, ~1 week of GPU + a fixed contact-metric suite; EgoTac gives the protocol template to copy |
| (d) Preference / ranking learning (RLHF-style) | **Yes, thinly.** [VTLA](https://www.arxiv.org/pdf/2505.09577) (preference learning for insertion), [Touch-R1](https://export.arxiv.org/pdf/2605.27154) (GRPO w/ output-side rewards + input grounding) | No reward model over *contact quality*; no human/AI preference dataset over tactile trajectories; DPO/RLHF on touch is 1-2 papers deep | Medium: needs a preference elicitation interface (pairwise contact clips are easy for humans to rank) + reward model; ~2 person-months |
| (e) Test-time compute / search / verification | **Partial, brand new (one paper).** [ViTaL](https://yilin-wu98.github.io/vital_website/) (CMU, [arXiv 2606.14981](https://arxiv.org/pdf/2606.14981), CoRL 2026): bi-level steering of a diffusion policy — visual mode selection + tactile refinement; first language-conditioned tactile reward in a world-model latent space; +51% over base policy | No *verifier* trained for contact correctness; no tree/graph search over contact-mode switches; no test-time-compute scaling curve (does 8x more samples help touch as it does LLMs?); no tactile self-consistency/reranking | Medium; the verifier is the contribution. ~2-3 person-months, and ViTaL gives the scaffolding |
| (f) Privileged distillation from simulation | **Saturated (table stakes).** [PTLD](https://ar5iv.labs.arxiv.org/html/2603.04531) (privileged tactile latent distillation for dexterity), [HapticVLA](https://arxiv.org/pdf/2603.15257v1), ablation tables everywhere | Conceptually nothing — only better teachers / higher sim fidelity. Do not claim novelty here. | Low |
| (g) Curriculum / automatic task generation | **Partial.** [arXiv 2609.25887](https://export.arxiv.org/pdf/2609.25887) ("What is the Better Curriculum": controller-shaped grasping behaviour for contact-force tasks); domain-randomization curricula for contact | No *automatic generation* of contact-rich task families (LLM task synthesis + tactile feasibility filter in sim); curricula remain hand-designed and single-task | Medium; LLM proposal + tactile feasibility filter is a clean, self-contained paper |
| (h) Cross-embodiment / cross-sensor transfer via canonical interface | **Crowded.** [ME-Dex 1.0](https://export.arxiv.org/pdf/2609.21449), [UniTacHand](https://arxiv.org/pdf/2512.21233v1), [TactX](https://ar5iv.labs.arxiv.org/html/2606.31236) + HTT (convergent papers on a shared tactile latent across heterogeneous sensors), Sharpa+Tsinghua (21 sensors, one policy), Sparsh-X | Still no *sensor-agnostic API + conformance test* a new lab can adopt in an afternoon; each paper defines a private latent; no zero-shot transfer across transduction principles (capacitive→optical) with published numbers | Medium-high, hardware-bound. The unclaimed win is a spec + conformance suite, not another model |
| (i) Benchmark-driven competition / leaderboards | **Yes but shallow.** [ManiSkill-ViTac 2025 challenge](http://www.arxiv.org.ezproxy.obspm.fr/pdf/2411.12503) | No persistent leaderboard, no held-out sensor track, no standardized contact-metric reporting; challenges are one-off | Low-medium (mostly logistics + eval harness); highest field-level ROI |
| (j) Haptics/HCI/biomechanics/sensor physics imports | **Barely.** [MoiréTac](https://ieeexplore.ieee.org/document/11696001) (dual-mode visuotactile), mechanoreceptor-inspired sensor design (ACS Nano Lett.) | Psychophysics results (e.g., vibrotactile masking, rate coding, adaptation) are not used as *supervision targets* or inductive biases for policies; no "tactile illusions as data augmentation" | Low cost, high surprise value: ~1 person-month for an augmentation/pretraining-text objective grounded in skin mechanics |

## 2. Ranked by value x (1 - occupancy)
1. **(c) Tactile scaling laws for control** — zero occupancy on the control axis, decisive, cheap. EgoTac proved the protocol is publishable; nobody has done it for policies or sensor count.
2. **(b) The curation half of the data engine** — TAMEn built collection; filtering/curation for contact richness is unclaimed and the field's real bottleneck.
3. **(e) Test-time compute / tactile verifier** — exactly one paper (ViTaL, CoRL 2026); enormous headroom in verifier design and compute-scaling curves.
4. **(j) Psychophysics/biomechanics imports** — near-zero occupancy, high surprise, ~1 person-month.
5. **(d) Preference/RLHF over contact quality** — thinly occupied (VTLA, Touch-R1); humans can rank contact clips easily, so the data is cheap.
6. **(a) Touch FM with unified interface** — highest ceiling but 5+ groups racing; only win by owning the interface *and* the held-out-sensor protocol.
7. **(i) Leaderboard** — field-level ROI, not a paper.
8. **(g) Auto task generation** — clean but smaller ceiling.
9. **(h) Cross-embodiment** — crowded (TactX/HTT/ME-Dex/UniTacHand all within months).
10. **(f) Privileged distillation** — table stakes.

## 3. Single best playbook to import + first experiment
**Import: empirical scaling laws (c) — and let the fitted curve dictate the data-engine design (b).**
First experiment (1 week, no new hardware): fix one tactile policy architecture and a three-task contact suite (in-hand rotation, peg insertion, one tool-use/force task). Sweep **data volume** {1, 5, 20, 80 h of contact-rich teleop} x **sensor count** {1, 3, 8 sites} x **policy size** {3M, 30M, 300M}, 3 seeds. Report contact-sensitive metrics (slip rate, force-tracking error, contact-event F1), not bare success rate. Fit power laws per axis; test whether data and sensor coverage are substitutes or complements (the tactile analogue of a Chinchilla tradeoff). Deliverable: the first tactile scaling plot plus a marginal-value curve answering "buy another sensor or collect another 20 hours?"
Why this wins: it is cheap, it is a *measurement* (hard to scoop once published), it reframes the field's budget question, and EgoTac already showed reviewers accept scaling analyses — but only for perception, so the control version is open.

## 4. Now table stakes (do not propose as novel)
- Privileged/teacher-student sim-to-real distillation for touch (PTLD, HapticVLA, and many more).
- Domain randomization + curriculum for contact tasks.
- SSL representation pretraining for a single vision-based tactile sensor (GelSight/DIGIT-style).
- Diffusion/flow policies conditioned on tactile; action chunking with touch.
- Cross-sensor latent alignment papers (4+ in 2026 alone).
- Sim benchmarks (ManiSkill-ViTac-style) as the evaluation.
- Single-arm insertion as the demo task.

## 5. Could NOT verify
- Any explicit tactile scaling law for *policy* performance (three differently-phrased searches). Closest: EgoTac's perception scaling analysis. Absence of evidence, not proof.
- Whether the Sharpa/Tsinghua "21 sensors, one policy" claim has a peer-reviewed numbers table (only Chinese press summaries surfaced).
- Whether Touch-R1's GRPO rewards reflect human preference or purely synthetic rewards.
- Exact corpus sizes for most "web-scale touch" candidates; the 10^5 figure is inferred, not read off a datasheet.
- TAMEn's full text (GitHub page truncated before the README body); the closed-loop claim rests on title + ar5iv snippets.
- DexPIE's abstract never mentions tactile sensing — it is dexterous-hand post-training, so its inclusion in row (b) is as a *template*, not as tactile prior art.

