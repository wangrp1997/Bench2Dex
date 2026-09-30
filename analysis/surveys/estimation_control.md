# Contact-State Estimation and Control-Theoretic Use of Touch

Focused literature survey for a mechanism-level, theory-carrying contribution at the
intersection of **tactile sensing** and **robot manipulation**.
Angle: **contact-state estimation + control-theoretic use of touch**.
Survey date: **2026-09-30**. Scope: tactile servoing / tactile pose & contact-geometry
estimation; slip / incipient slip / contact-mode / jamming detection; touch-informed
force–impedance–compliance control incl. learning from demonstration; observability,
uncertainty quantification and filtering over contact state; guarantees (calibration,
conformal, certificates, reachability, formal monitoring); and the question of whether a
learned tactile monitor is known to fail against a one-line contact statistic.

## 0. Method and verification legend

Tools: `web_search` over arXiv / Semantic Scholar / IEEE / RSS / CoRL / ICRA / RA-L / T-RO /
CDC / HAL / MERL, plus the **arXiv Atom API** (`export.arxiv.org/api/query`) for systematic
title/abstract retrieval, plus direct fetches of arXiv abstract pages, RSS proceedings pages,
project pages and HAL landing pages.

**Important tooling limitation:** `web_fetch` rejects `application/pdf`
(`unsupported content type "application/pdf"`). Therefore **no PDF full text was read this
session**. Claims about method internals come from HTML abstracts, proceedings HTML pages and
project pages; everything else is flagged.

Verification tags used in the table:

| tag | meaning |
|---|---|
| ✅ | abstract / proceedings HTML / project page **fetched this session** |
| ◐ | title + URL seen verbatim in search results (with usable snippet), abstract **not** fetched |
| ✗ | could not verify |

The arXiv API stopped responding mid-survey (`unsupported content type "unknown"`) and the
Semantic Scholar API was intermittently HTTP 429, so arXiv-only 2026 works are covered
**incompletely** — see §5.

---

## 1. Table

### (a) Tactile servoing, tactile pose and contact-geometry estimation

| work | venue/date | what is estimated / controlled | guarantee | evaluation | tag |
|---|---|---|---|---|---|
| [Floriano Vazquez & Lepora, Uncertainty-aware deep learning for robot touch: Bayesian tactile servo control](https://arxiv.org/abs/2104.14184) | ICRA 2021 | edge pose + per-prediction uncertainty (Gaussian-density NN), Kalman-filtered | **none** — calibrated-style variance only, no coverage claim | TacTip optical sensor, 2D contour following; error halved vs deterministic NN | ✅ |
| [Lepora & Lloyd, Pose-Based Tactile Servoing (PBTS)](https://arxiv.org/abs/2012.02504) | arXiv 2020 (later IEEE) | sensor pose relative to object feature (edge/surface) → servo law; formalises PBTS vs visual servoing | stability argued by analogy to visual servoing; **no formal proof in abstract** | TacTip, regular + irregular 3D objects | ✅ |
| [Lloyd & Lepora, Pose and shear-based tactile servoing](https://arxiv.org/abs/2312.08411) | **IJRR 2023** | contact pose **+ shear**; Gaussian-density NN + discriminative Bayesian filter on **SE(3)** (Lie-group re-derivation) | **none** (probabilistic filter; no bound) | object tracking, surface following, single/dual-arm pushing | ✅ |
| [Fan et al., Tac-VGNN](https://arxiv.org/abs/2303.02708) | ICRA 2023 | tactile pose (depth/shear) via Voronoi GNN | none | TacTip surface following; +28.6% depth accuracy over plain GNN | ✅ |
| [Wang et al., TacRefineNet](https://arxiv.org/abs/2509.25746) | arXiv 2025-09 (v2 2026-07) | goal-conditioned **corrective wrist-pose increments** from cur/target multi-finger tactile images; "external-dexterity tactile servoing loop" | none | 156,007 sim samples → zero-shot to 11-DoF 5-finger hand; 80.7%/59.3% under 10°/10 mm | ✅ |
| [Freud, Lin & Lepora, SimShear](https://arxiv.org/abs/2508.20561) | **CoRL 2025** | shear-conditioned sim-to-real tactile **control** (shPix2pix GAN); tactile tracking + co-lifting | none | 1–2 mm contact error on varied trajectories | ✅ |
| [Kleff, Joseph & Padois, Geometrically Consistent Tactile Servoing via Hybrid Force–Position Control at the Center of Pressure](https://hal.science/hal-05441031) *(DOI 10.1109/LRA.2026.3699162)* | **RA-L 2026**, 11(7):8928–8935 | **contact pose + force at the Center of Pressure**; hybrid force–position law from the **CoP Jacobian**, decoupling force/motion subspaces | "geometric consistency"; **no stability/robustness theorem** | real manipulator, physical interaction regulation; beats image-based and pose-based tactile controllers | ✅ |
| [Zheng et al., Bayesian Active Object Recognition and 6D Pose Estimation from Multimodal Contact Sensing](https://arxiv.org/abs/2603.21410) | arXiv 2026-03 | joint belief over **object class + 6D pose**; customised **particle filter**; active touch selection under reachability constraints | **none** (Bayes posterior, no coverage guarantee) | sim + Franka Panda, **11 YCB objects**; beats F/T-only baselines | ✅ |
| [Van der Merwe, Ota, Berenson, Fazeli & Jha, TacGraph — Simultaneous Extrinsic Contact and In-Hand Pose Estimation via Distributed Tactile Sensing](https://arxiv.org/abs/2512.23856) ([project](https://tacgraph.github.io/)) | **RA-L 2026** | **in-hand pose + extrinsic contact**; learned tactile modules → **factor graph** enforcing geometric consistency, non-penetration, contact kinematics, force balance | **none** (point estimate; no error bound) | tactile-only beats ICP / CHSEL / SCOPE(v2); open-loop insertion at ~3 mm tolerance | ✅ |
| [Ota et al., Tactile Estimation of Extrinsic Contact Patch for Stable Placement](https://arxiv.org/abs/2309.14552) | ICRA 2024 | extrinsic **contact patch** from force + tactile → place-and-release stability | none | Jenga-like object pairs | ✅ |
| [Tian et al., Deep Tactile MPC](https://arxiv.org/abs/1903.04128) | ICRA 2019 | learned tactile dynamics model + MPC for tactile servoing to a goal **tactile image** | none | GelSight; ball, analog stick, 20-sided die | ✅ |
| [Sutanto et al., Learning Latent Space Dynamics for Tactile Servoing](https://arxiv.org/abs/1811.03704) | ICRA 2019 | latent-space dynamics over tactile manifold; contact-point tracking from demo | none | tactile finger, contact tracking | ✅ |
| [Yu & Rodriguez, Realtime State Estimation with Tactile and Visual Sensing for Inserting a Suction-held Object](https://arxiv.org/abs/1803.08014) | arXiv 2018 (IROS submission) | **pose + contact formation**; **iSAM factor graph** fusing force, vision, kinematics, contact geometry; data-driven contact-formation inference | none | 3D insertion, instrumented ground truth | ✅ |
| [M-VTOP: Modular Visuo-Tactile Object Pose Estimation](https://www.merl.com/publications/docs/TR2026-070.pdf) | MERL TR 2026 | visuo-tactile object pose | — | — | ◐ |

### (b) Slip, incipient slip, contact-mode / contact-event / jamming detection

| work | venue/date | what is estimated / controlled | guarantee | evaluation | tag |
|---|---|---|---|---|---|
| [Howe & Cutkosky, Sensing skin acceleration for slip and texture perception](https://cir.nii.ac.jp/crid/1360016869742020992) | **ICRA 1989** | slip via **acceleration threshold** on a sensing skin ("one-line statistic" ancestor) | threshold heuristic | classical demonstration | ◐ |
| [Dong, Ma, Donlon & Rodriguez, Maintaining Grasps within Slipping Bound by Monitoring Incipient Slip](https://arxiv.org/abs/1810.13381) | arXiv 2018 | **dense slip field** = deviation of the tactile motion field from a 2D rigid transform (analytic, *no training*) | none (86.25% detection accuracy) | GelSight, 10 objects × 240 trials, 24 Hz; closed-loop grip-force regulation | ✅ |
| [Taylor, Dong & Rodriguez, GelSlim 3.0](https://arxiv.org/abs/2103.12269) | arXiv 2021 (RA-L) | analytic shape, 3D force distribution, **incipient slip** in one compact finger | analytic, open source | design + demos | ✅ |
| [James, Redmond & Lepora, A Biomimetic Tactile Fingerprint Induces Incipient Slip](https://arxiv.org/abs/2008.06904) | IROS 2020 | engineered concentric ridges **induce** partial slip, making it observable longer before gross slip | mechanism (no guarantee) | multiple object shapes, drop prevention | ✅ |
| [Muthusamy et al., Neuromorphic Event-Based Slip Detection and Suppression](https://arxiv.org/abs/2004.07386) | IEEE Access 2020 | incipient slip via **autonomously sampled noise threshold** (baseline) + feature-based detectors; fuzzy grip-force suppression | none | **2 kHz (Δt = 500 µs)**; explicit threshold baseline and a slip metric | ✅ |
| [Lu, Deng, Redmond, Psomopoulou & Ward-Cherrier, A Neuromorphic Incipient Slip Detection System using Papillae Morphology](https://arxiv.org/abs/2509.09546) | arXiv 2025-09 (RA-L under review) | 3-class slip state (no / incipient / gross) via spiking CNN on NeuroTac | none (94.33% acc) | detects incipient slip ≥ **360 ms** before gross slip in all trials | ✅ |
| [Li et al., Modeling, Simulation and Application of Spatio-Temporal Characteristics Detection in Incipient Slip](https://arxiv.org/abs/2502.17335) | arXiv 2025-02 | stick–slip region **spatial distribution + temporal dynamics**; ties characteristic **strain-rate extreme events** to local slip; also friction estimation | physics-based model, no formal guarantee | simulation + prototype, varied geometry/friction/load | ✅ |
| [Komeno & Matsubara, Incipient Slip Detection by Vibration Injection into Soft Sensor](https://arxiv.org/abs/2402.11879) | **RA-L 2024** | **stick ratio** compressed into a 1-D pressure signal via injected white-noise vibration + spectral change; stick-ratio **stabilisation control** | none | beats conventional methods on estimation error and control | ✅ |
| [Jian et al., SlipSense](https://arxiv.org/abs/2609.15910) | **CoRL 2026** | multimodal slip detection: 32×32 piezoresistive array @240 Hz + 3-axis MEMS accel @8 kHz, cross-modal attention, causal prediction | none (but reports FPR < 1.6%) | **1.4 M frames, 37 objects**; 96.7% Macro F1; 76% of slips within **23.1 ms**; **zero-shot UMI→Tesollo hand** transfer | ✅ |
| [Robust Slip Detection and Material Classification via Spatiotemporal Transformers](https://arxiv.org/abs/2608.24162) | arXiv 2026-08 | slip + material class | none | visuo-tactile, spatiotemporal transformer | ◐ |
| [Wang et al., Robust Learning-Based Incipient Slip Detection using PapillArray](https://arxiv.org/abs/2307.04011) | arXiv 2023-07 | incipient slip + data augmentation for transfer | none | 95.6% offline / 96.8% on transferred gripper | ✅ |
| [Pang et al., Viko 2.0](https://arxiv.org/abs/2204.10082) | arXiv 2022 | contact area, shear force, **incipient slip at 24 Hz** from a visuotactile gecko gripper | none | adaptive grasping over wide geometry range | ✅ |
| [CoPRE: Improving Sensitivity in Proprioceptive Contact Detection for Low-Cost Robot Arms](https://arxiv.org/abs/2609.27381) | arXiv 2026-09 | **contact onset** from proprioception | none | low-cost arms | ◐ |
| [Debus & Dupont, Contact State Estimation Using Multiple Model Estimation and Hidden Markov Models](https://www.semanticscholar.org/paper/Contact-State-Estimation-Using-Multiple-Model-and-Debus-Dupont/bf75f4b0117c1aede5842e2af9093c04caf73649) | widely cited as **IJRR 2004** (venue/year not independently confirmed here) | **discrete contact state** via multiple-model estimation + HMM (classical mode classification) | probabilistic (model-based), no coverage guarantee | assembly-style contact transitions | ◐ |

### (c) Touch-informed force / impedance / compliance control, incl. learning from demonstration

| work | venue/date | what is estimated / controlled | guarantee | evaluation | tag |
|---|---|---|---|---|---|
| Whitney 1977 force feedback; Salisbury 1980 stiffness control; Raibert & Craig 1981 hybrid force/position; Mason 1981 compliance; Hogan 1985 impedance control | 1977–1985 | the entire classical force/compliance/impedance canon | stability results for linear impedance/force loops under model assumptions | classical | ✗ (canonical citations; not fetched) |
| [Aydinoglu & Posa, Contact-Aware Controller Design for Complementarity Systems](https://www.ieee-ras.org/images/Technical_Committee_Images/model-based-optimization/2020_ICRA_Posters/ICRA2020_Poster_Alp_-_Alp_Aydinoglu.pdf) | ICRA 2020 | LQR/controller synthesis directly on **complementarity (contact) dynamics** | theory: controller design for complementarity systems | simulation | ◐ |
| [Karacan, Zhang, Sadeghian, Wu & Haddadin, VA-UFIC](https://arxiv.org/abs/2408.14219) | IROS 2024 | visuo-tactile exploration of unknown 3D curvature; online **contact-alignment monitoring** (tactile error + curvature + orientation) → adaptive stiffness/force | **passivity via virtual energy tanks** (safety/stability guarantee) | Franka, point/flat/concave/convex/loss-of-contact | ✅ |
| [Zhang, Karacan, Sadeghian, Wu, Wu & Haddadin, Tactile-Morph Skills](https://arxiv.org/abs/2408.12285) | arXiv 2024-08 | unified force–impedance control driven by a **predicted energy budget** (TCN estimates energy for a motion+force profile) | **energy-budget safety** (stops when allotted energy is exhausted); passivity-based | real polishing-like tasks; zero-shot transfer to new geometry with similar friction | ✅ |
| [Bilaloglu, Löw & Calinon, Tactile Ergodic Coverage on Curved Surfaces](https://arxiv.org/abs/2402.04862) | arXiv 2024 (v3 2025) | geometric-algebra task-space impedance controller tracking a line while exerting a **prescribed force along it**; ergodic coverage objective on point clouds | none (controller formulation) | kitchenware cleaning/inspection | ✅ |
| [Shukla et al., DPA-FTG (Diffusion Policy Augmented by Fast Trajectory Generation)](https://arxiv.org/abs/2608.03103) | arXiv 2026-08 (RCIM) | 5 Hz diffusion picks a strategy/primitive; 60 Hz **force-conditioned neural impedance controller** | none | bimanual battery disassembly (compliant sheet separation); beats RDP | ✅ |
| [Li et al., M2-ResiPolicy](https://arxiv.org/abs/2603.15152) | arXiv 2026-03 | 10 Hz diffusion "Master-Guidance" with **tactile-intensity-driven adaptive vision/touch fusion**; 60 Hz GRU "Micro-Residual Corrector" on TCP wrench; force-mixed PBIC layer | none | fragile grasping, precision insertion; 93% damage-free | ✅ |
| [Gao et al., IMPACT](https://arxiv.org/abs/2606.10818) | arXiv 2026-06 | internal-model **predictive control** for forceful manipulation, decoupled planning/control | none | sim + real; generalises across unseen object weights | ✅ |
| [ForceMimic](https://arxiv.org/abs/2410.07554) | ICRA 2025 | **force-centric imitation learning** with a force–motion capture system; force as first-class modality | none | contact-rich manipulation, incl. deformable/force-critical tasks | ✅ |
| [Huang et al., 3D-ViTac](https://pmlr.com.cn/v270/huang25e.html) | NeurIPS 2024 | visuo-tactile policy with 3D tactile representation | none | fine-grained bimanual manipulation | ✅ |
| [Wang, Luo, Zhang, Chen, Pan & Zhang, POMDP-Guided Active Force-Based Search for Robotic Insertion](https://arxiv.org/abs/2404.03943) | IROS 2023 | **POMDP over contact configuration** (partially observable contact state) → search strategy in an FSM + Cartesian impedance controller; **proprioception only** | none (POMDP policy) | simulation; beats blind search in success/time/arc length | ✅ |
| [Shirai & Jha, Contact-Aware Covariance Control of Stochastic Contact-Rich Systems](https://www.semanticscholar.org/paper/Contact-Aware-Covariance-Control-of-Stochastic-Shirai-Jha/4b98536eebbc6a309ea96cabf875d7ead3b39890) | Semantic Scholar (venue not confirmed) | distribution/covariance of the state under contact | covariance-control theory | — | ◐ |
| [Covariance Steering for Uncertain Contact-rich Systems](https://ar5iv.labs.arxiv.org/html/2303.13382) | arXiv 2023-03 | steer state distribution through contact | distributional guarantee | — | ◐ |
| [A Control Framework With Tactile Diffusion Policy and Variable Impedance for Unknown Surface Tracking](https://www.semanticscholar.org/paper/A-Control-Framework-With-Tactile-Diffusion-Policy-Li-Deng/960651e7654a39eebcde381354ef5180b5166283) | Semantic Scholar (venue not confirmed) | tactile diffusion policy + variable impedance for surface tracking | — | — | ◐ |

### (d) Observability, uncertainty quantification, filtering over contact state

| work | venue/date | what is estimated | guarantee / uncertainty type | evaluation | tag |
|---|---|---|---|---|---|
| [Gadeyne, Lefebvre & Bruyninckx, Bayesian Hybrid Model-State Estimation Applied to Simultaneous Contact Formation Recognition and Geometrical Parameter Estimation](https://journals.sagepub.com/doi/10.1177/0278364905056196) | **IJRR 2005** | **contact formation (discrete) + geometry (continuous) jointly** | Bayes posterior; no coverage guarantee | compliant-motion tasks | ◐ |
| [Lefebvre, Bruyninckx & De Schutter, Task Planning With Active Sensing For Autonomous Compliant Motion](https://journals.sagepub.com/doi/10.1177/0278364904050079) | **IJRR 2005** | contact state + parameters, with **active sensing** to improve estimation | Bayes + active-sensing criterion | compliant motion | ✅ (title/venue page) |
| Jia & Erdmann (Montana's equations → nonlinear system → **observability of contact**) | CMU tech report 1997–98 (~[PDF](https://www.ri.cmu.edu/pub_files/pub1/jia_yan_bin_1997_1/jia_yan_bin_1997_1.pdf)) | observability of contact kinematics/dynamics in a manipulation task | **structural observability analysis** (classical; the closest existing "observability of contact" mechanism) | analysis | ◐ |
| [Koval, Pollard & Srinivasa, Manifold Representations for State Estimation in Contact Manipulation](https://www.ri.cmu.edu/pub_files/2013/12/koval2013manifold_isrr.pdf) | ISRR 2013 | pose under contact via **manifold particle filters** (pose is low-dimensional given contact) | Bayes posterior; **reports a structural negative result**: particle filter gets *worse* as sensor resolution / update rate increases | contact manipulation | ✅ (snippet with the negative finding) |
| [Yu & Rodriguez, iSAM contact-formation state estimation](https://arxiv.org/abs/1803.08014) | arXiv 2018 | pose + contact formation, **factor-graph (iSAM)** | none (MAP point estimate) | 3D insertion | ✅ |
| [Sanan, Tully, Bajo, Simaan & Choset, CARE: Simultaneous Compliance and Registration Estimation](https://www.roboticsproceedings.org/rss10/p51.html) | **RSS 2014** | **compliance map + registration** under forceful palpation, KF with force-balanced spring model ("SLAM for stiffness") | none (KF covariance) | continuum robot, 2 flexible bench-top structures | ✅ |
| [Maravgakis et al., Probabilistic Contact State Estimation for Legged Robots using Inertial Information](https://arxiv.org/abs/2303.00538) | ICRA 2023 | **probability of stable contact** from end-effector IMU via kernel density estimation | probability, no coverage guarantee | ATLAS, TALOS, Unitree GO1; real + sim | ✅ |
| [Rotella, Schaal & Righetti, Unsupervised Contact Learning for Humanoid Estimation and Control](https://arxiv.org/abs/1709.07472) | arXiv 2017 | per-DoF contact probability via fuzzy clustering (proprioception only) | probability; explicitly **beats a purely normal-force-threshold baseline** | simulated rough low-friction terrain | ✅ |
| [Zheng et al., particle-filter belief over class+pose](https://arxiv.org/abs/2603.21410) | arXiv 2026-03 | joint class+pose belief, active touch | Bayes posterior | 11 YCB, Franka | ✅ |
| [TacGraph factor graph](https://arxiv.org/abs/2512.23856) | RA-L 2026 | pose + extrinsic contact with physical-constraint factors | none | tactile-only insertion | ✅ |

### (e) Guarantees: certificates, conformal / calibrated risk, reachability, formal monitoring

| work | venue/date | what is certified | guarantee | evaluation | tag |
|---|---|---|---|---|---|
| [Li & Chou, Certifiable Gradient-Based Contact-Rich Manipulation via Smoothing-Error Reachable Tubes](https://www.roboticsproceedings.org/rss22/p190.html) *(DOI 10.15607/RSS.2026.XXII.190)* | **RSS 2026** | gradient-based policy synthesis through smoothed contact dynamics, with the smoothing error characterised as a **set-valued deviation** | **formal guarantees of constraint satisfaction AND goal reachability on the true hybrid dynamics**; claims "first certifiable gradient-based policy synthesis for contact-rich manipulation" | planar pushing, object rotation, in-hand dexterous manipulation | ✅ |
| [Tang & Althoff, Formal Verification of Robotic Contact Tasks via Reachability Analysis](https://arxiv.org/abs/2307.13977) | IFAC WC 2023 | hybrid automaton with contact + time delays; guard-intersection reachability | **formal verification of all reachable states vs specification** | safe human–robot interaction with constrained collisions | ✅ |
| [Tang & Althoff, Efficiently Obtaining Reachset Conformance for the Formal Analysis of Robotic Contact Tasks](https://arxiv.org/abs/2410.10391) | **IROS 2024** | reachset-conformant hybrid model (linear dynamics + injected non-determinism) | **reachset conformance** — abstract model encloses all measurements, transferring safety properties | two 3-DOF robots | ✅ |
| [Marques, Popov & Berenson, CaPTURe: Particle-Based Conformal Prediction for Contact-Aware Uncertainty Calibration](https://arxiv.org/abs/2608.09166) | **COPA 2026** (PMLR 329:813–843) | **prediction regions over future robot configuration**, with particle models of arbitrary fidelity; captures contact-rich vs contactless uncertainty | **distribution-free conformal coverage guarantee** (user-set probability) | marble-in-labyrinth + **tight-tolerance peg-in-hole**; up to +30% absolute success over best baseline. **No tactile sensing** | ✅ |
| [Liu, Zhang & Lin, Robust Operational Space Control with Conformal Disturbance Bounds for Safe Redundant Manipulation](https://arxiv.org/abs/2607.00424) | **IROS 2026** | **lumped disturbance bound** (ESO) calibrated online by sliding-window conformal prediction, used in a robust CBF | **practical probabilistic safety guarantee** via calibrated disturbance bound | 7-DoF Franka Research 3, 1 kHz, mm tracking under disturbances. **Disturbance, not contact state; no touch** | ✅ |
| [Ramesh & Prakash, HJ-SafeDMP](https://arxiv.org/abs/2606.28995) | arXiv 2026-06 | Control-Barrier-**Value Function** learned via finite-difference HJ recursion + expectile objective + conformal calibration | **provably safe** forward-invariant safe set + finite-sample probabilistic coverage | 7-DOF manipulator; closed-form filter, no QP | ✅ |
| [Li, Fang, Polisetti, Song & Chou, CORD-SLS](https://arxiv.org/abs/2606.14188) | arXiv 2026-06 | output-feedback **robust MPC** with contact smoothing; conformal calibration of perception-error bounds → **reachable tubes** | high-probability safe control | rope/cloth (contact-rich deformables), sim + hardware | ✅ |
| [Ward et al., Foundational World Models Accurately Detect Bimanual Manipulator Failures](https://arxiv.org/abs/2603.06987) | **ICRA 2026** | runtime **failure monitor**; world-model uncertainty used as the **nonconformity score** in conformal prediction | conformal (distribution-free) false-alarm control | Push-T + new bimanual cable dataset; "considerably outperforms statistical techniques"; 1/20 the parameters of next-best learned baseline | ✅ |
| [Zhang et al., Foresight](https://arxiv.org/abs/2606.23085) | arXiv 2026-06 | failure detection for long-horizon manipulation using **action-conditioned world-model latents** + **functional conformal prediction** | conformal calibration of detection thresholds | LIBERO-Long, ManiSkill-Long, BEHAVIOR-1K + real ReactorX-200 / Franka | ✅ |
| [Zheng et al., Rewind-IL](https://arxiv.org/abs/2604.16683) | arXiv 2026-04 | training-free monitor: **TIDE** inter-chunk discrepancy, **split conformal** calibration, + state respawning | conformal (coverage) + recovery mechanism | real + simulated long-horizon IL, flow-matching policies | ✅ |
| [Chen, Lyu & Beksi, ReconVLA](https://arxiv.org/abs/2604.16677) | arXiv 2026-04 | conformal prediction on **VLA action tokens** + robot-state-space outlier detection | calibrated (conformal) uncertainty + failure detection | sim + real manipulation | ✅ |
| [Learnable Conformal Prediction with Context-Aware Nonconformity Functions for Robotic Planning and Perception](https://arxiv.org/abs/2509.21955) | arXiv 2025-09 | learnable nonconformity functions | conformal coverage | robotic planning/perception | ◐ |
| [ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models](https://arxiv.org/abs/2608.13438) | arXiv 2026-08 | **pre-contact** execution monitoring; scores the consequence of the pending action | — (not verified) | — | ◐ |
| [Contact-Aware Controller Design for Complementarity Systems](https://www.seas.upenn.edu/~posa/DynamicWalking2020/734-1114-1-RVP.pdf) | ICRA 2020 | control synthesis with contact complementarity | theory | balancing | ◐ |
| [Energy-tank passivity controller tested before/after contact loss](http://crlab.cs.columbia.edu/humanoids_2018_proceedings/media/files/0104.pdf) | Humanoids 2018 | passivity under contact loss | passivity | 3 experiments | ◐ |

### (f) "Does a learned tactile monitor fail to beat a one-line contact statistic?"

**No published statement of this observation was found.** See §4(f) for the closest evidence and
why the claim is structurally plausible but unestablished.

---

## 2. CROWDED

Ranked by how saturated the space is, with the specific thing that is saturated.

**C1. Slip / incipient-slip *detection accuracy* is saturated.**
This is now a benchmark-and-latency engineering field, not an open scientific question.
Evidence: [SlipSense (CoRL 2026)](https://arxiv.org/abs/2609.15910) reports 96.7% Macro F1 with
FPR < 1.6% at 240 Hz on **1.4 M frames / 37 objects**, characterises detection latency explicitly
(23.1 ms for 76% of events), and does **zero-shot** transfer from UMI data to a Tesollo hand —
i.e. even cross-platform generalisation is now claimed. Alongside:
[Papillae-morphology neuromorphic incipient slip (94.33%, ≥360 ms lead time)](https://arxiv.org/abs/2509.09546),
[PapillArray learned incipient slip (95.6%→96.8% transfer)](https://arxiv.org/abs/2307.04011),
[spatiotemporal transformers](https://arxiv.org/abs/2608.24162),
[GelSlim 3.0 analytic incipient slip](https://arxiv.org/abs/2103.12269),
[Dong et al. dense slip field](https://arxiv.org/abs/1810.13381),
[Viko 2.0](https://arxiv.org/abs/2204.10082).
**Do not propose "better slip detection".**

**C2. Tactile servoing (perception→servo loop) is saturated, including the classical-theory revival.**
[Lloyd & Lepora IJRR 2023](https://arxiv.org/abs/2312.08411) already put a Gaussian-density NN plus a
**Lie-group (SE(3)) Bayesian filter** in the tactile servo loop;
[PBTS](https://arxiv.org/abs/2012.02504) already formalised the tactile/visual servoing analogy;
[SimShear (CoRL 2025)](https://arxiv.org/abs/2508.20561) already closed the sim-to-real shear gap; and
critically [Kleff et al., RA-L 2026](https://hal.science/hal-05441031) already derived **hybrid
force–position control at the Center of Pressure** as the "geometrically consistent" tactile feature
space. A "tactile + impedance/hybrid force control with a Jacobian" contribution is now anticipated.

**C3. Tactile contact-geometry / pose estimation with filters and factor graphs is saturated (2026).**
[TacGraph (RA-L 2026)](https://arxiv.org/abs/2512.23856) already combines *learned tactile modules +
a factor graph with contact kinematics, non-penetration and force-balance factors*, tactile-only,
and already beats ICP/CHSEL/SCOPE. [Bayesian active 6D pose + class with a particle filter over 11
YCB objects on a Franka](https://arxiv.org/abs/2603.21410) already occupies the active-Bayesian
version. [iSAM contact-formation estimation](https://arxiv.org/abs/1803.08014) occupies the
classical factor-graph version.

**C4. Conformal / calibrated-uncertainty machinery for robot manipulation safety is a 2026 stampede.**
Within months: [CaPTURe (COPA 2026, contact-aware conformal prediction regions)](https://arxiv.org/abs/2608.09166),
[conformal disturbance bounds + robust CBF at 1 kHz (IROS 2026)](https://arxiv.org/abs/2607.00424),
[HJ-SafeDMP (HJ reachability + conformal)](https://arxiv.org/abs/2606.28995),
[CORD-SLS (conformal-calibrated reachable tubes for deformable contact-rich MPC)](https://arxiv.org/abs/2606.14188),
[world-model failure monitor with conformal nonconformity scores (ICRA 2026)](https://arxiv.org/abs/2603.06987),
[Foresight (functional conformal)](https://arxiv.org/abs/2606.23085),
[Rewind-IL (split conformal + TIDE)](https://arxiv.org/abs/2604.16683),
[ReconVLA (conformal on VLA tokens)](https://arxiv.org/abs/2604.16677).
**"Apply conformal prediction to a tactile monitor" is no longer novel on its own.**

**C5. Formal guarantees for contact-rich manipulation now exist — but without touch.**
[RSS 2026 Li & Chou](https://www.roboticsproceedings.org/rss22/p190.html) claims the first
**certifiable gradient-based policy synthesis** with guarantees of constraint satisfaction *and*
goal reachability on the **true hybrid dynamics**, and already covers planar pushing, object
rotation and **in-hand dexterous manipulation**. Reachability-based verification of contact tasks
also already exists ([IFAC 2023](https://arxiv.org/abs/2307.13977),
[IROS 2024 reachset conformance](https://arxiv.org/abs/2410.10391)). So "guarantees for contact-rich
manipulation" *per se* is taken; the unoccupied part is guarantees **closed through a tactile
estimator**.

**C6. Tactile / force-conditioned policy learning (the "tactile diffusion policy" bandwagon) is saturated.**
[ForceMimic (ICRA 2025)](https://arxiv.org/abs/2410.07554), [3D-ViTac (NeurIPS 2024)](https://pmlr.com.cn/v270/huang25e.html),
[DPA-FTG](https://arxiv.org/abs/2608.03103), [M2-ResiPolicy](https://arxiv.org/abs/2603.15152),
[IMPACT](https://arxiv.org/abs/2606.10818), [Touch2Trace (CoRL 2026, tactile-only cable tracing,
93% success, 60 Hz)](https://arxiv.org/abs/2609.15921), plus
[controller-shaped demonstrations as a collection-time teacher](https://arxiv.org/abs/2609.25887)
which explicitly probes *the boundary* of tactile-free policies. Adding "tactile tokens to ACT/DP"
is exactly the crowded move.

**C7. Classical contact-state estimation (mode + geometry) is a well-developed older field.**
[Bayesian hybrid model-state estimation (IJRR 2005)](https://journals.sagepub.com/doi/10.1177/0278364905056196),
[active-sensing compliant motion (IJRR 2005)](https://journals.sagepub.com/doi/10.1177/0278364904050079),
[multiple-model + HMM contact state](https://www.semanticscholar.org/paper/Contact-State-Estimation-Using-Multiple-Model-and-Debus-Dupont/bf75f4b0117c1aede5842e2af9093c04caf73649),
[Koval manifold particle filters](https://www.ri.cmu.edu/pub_files/2013/12/koval2013manifold_isrr.pdf),
[CARE compliance+registration KF (RSS 2014)](https://www.roboticsproceedings.org/rss10/p51.html).
Re-deriving this with tactile arrays and no new theory is not a contribution.

---

## 3. OPEN

Five mechanism-level, theory-carrying openings. I state each as a **claim + a theorem you would
have to prove + the experiment that would falsify it**.

### O1. **Identifiability theory of contact state from touch (the "identifiable contact subspace")**
**Claim.** For a hybrid contact system (discrete mode *m* ∈ {free, sticking, incipient-slip,
gross-slip, jamming/wedged, …}) with contact kinematics in the sense of Montana's equations and a
*distributed* tactile observation map `y = h_m(q, θ) + ε`, the practically reachable estimation
accuracy of `(m, θ)` is governed by a **mode-dependent Fisher information / observability rank
condition**, and in many regimes the information is concentrated in a **low-dimensional sufficient
statistic** (total normal force, CoP displacement, or a single strain-rate extremum).
**Theorems to prove.** (i) A hybrid observability rank condition per mode for the tactile output
map, giving which mode pairs are *distinguishable* and which are *weakly unobservable*;
(ii) a Cramér–Rao lower bound on contact-geometry and mode-estimation error as a function of
tactile resolution/placement; (iii) a **sufficiency theorem** identifying the minimal statistic
that attains the bound (and hence a *no-free-lunch* statement for learned monitors: no estimator
can beat that bound, and a monitor whose accuracy equals the bound's attained value is *not
extracting more information*, merely re-encoding it).
**Falsification experiment.** On an existing public slip/contact dataset (e.g. SlipSense-scale
piezoresistive+accelerometer data), compute the empirical CRB for mode/geometry from a calibrated
sensor-noise model, then show that a learned monitor's accuracy coincides with the accuracy of a
one-line statistic at the bound. Conversely, exhibit a contact mode where the bound *requires*
multi-channel spatial data and where the one-line statistic provably fails — that is the regime
worth learning in.
**Why it is unoccupied.** Searches on tactile+observability returned no tactile observability
analysis; the classical observability-of-contact work (Jia & Erdmann) is force/point-wrench based
and pre-dates tactile arrays; [TacGraph](https://arxiv.org/abs/2512.23856) uses physical
constraints but proves no bounds.

### O2. **Coverage-to-invariance: from a certified contact-state *set* to a provable contact *invariant***
**Claim.** A set-valued tactile contact-state estimator with a distribution-free coverage guarantee
(conformal or set-membership) is sufficient to certify a **contact invariant** — e.g. "the grasp is
never lost", "normal force ≤ F_max", "no gross slip occurs", "the object stays in the assembly's
insertion basin" — provided the controller is Lipschitz in the contact-state estimate.
**Theorem to prove.** *Coverage-to-invariance lemma*: if P(contact state ∈ estimated set) ≥ 1−α and
the controller's safe-set condition is *L*-Lipschitz in the contact state with margin δ, then the
closed loop keeps the invariant with probability ≥ 1−α whenever the estimator's set radius
r ≤ δ/L. This is the missing bridge between the estimator literature (which stops at coverage) and
the certified-control literature (which assumes a known disturbance bound).
**Guarantee type:** distribution-free probabilistic invariance, with an explicit radius–margin
budget. Worth also deriving the *tight* r vs δ trade-off (a control-theoretic "sensing budget").
**Why it is unoccupied.** [Liu et al. IROS 2026](https://arxiv.org/abs/2607.00424) calibrates a
**disturbance magnitude**, not a hybrid contact state; [CaPTURe](https://arxiv.org/abs/2608.09166)
calibrates **future configuration regions**, not contact state, and does not close a control
certificate through the estimate; [RSS 2026 tubes](https://www.roboticsproceedings.org/rss22/p190.html)
certifies but has **no sensing in the loop**.

### O3. **Active tactile excitation as a *persistent-excitation* condition for contact modes**
**Claim.** Some contact modes — pre-eminently **incipient slip** (partial slip of the contact patch)
and jamming/wedging — are **not identifiable from quasi-static tactile images**; they become
identifiable only when the contact is actively excited (vibration injection, micro-motion, probing
waveform), and there is a **minimum excitation amplitude × bandwidth** for identifiability.
**Theorems to prove.** (i) Unidentifiability of the stick-ratio from the quasi-static
strain/pressure map alone (a rank/invariance argument: many stick-ratio values give the same
image); (ii) a persistent-excitation condition under which the stick-ratio becomes identifiable,
plus a CRB for stick-ratio estimation versus excitation energy; (iii) an adaptive excitation
policy minimising injected energy subject to an identifiability constraint — this is the
*dual* of classical active sensing, with the excitation as the decision variable.
**Prior art to beat.** [Komeno & Matsubara, RA-L 2024](https://arxiv.org/abs/2402.11879) is an
*existence proof* of the mechanism (white-noise injection + spectrum change → stick ratio, then
stick-ratio stabilisation) but offers **no identifiability theory and no minimal-excitation
result**. [Biomimetic fingerprint ridges](https://arxiv.org/abs/2008.06904) is a *passive*
structural version of the same idea.

### O4. **Contact-mode-switching control that is certified under mode-estimation error**
**Claim.** Mode-dependent impedance/force controllers for contact-rich tasks can be certified when
the *discrete mode estimate itself is wrong with probability ≤ α*, using a **multiple-Lyapunov /
hybrid-systems certificate with a dwell-time condition expressed in terms of the monitor's error
rate**.
**Theorem to prove.** Exponential stability (or a bounded ultimate bound) of the switched contact
system when the controller is chosen from a finite family indexed by a mis-estimated mode, under
(a) a conformal per-mode false-classification rate, and (b) a minimum dwell time. Corollary: the
maximum tolerable monitor error rate for a given task, which turns monitor accuracy into a
*control specification*.
**Why it is unoccupied.** Mode-switching hybrid force/position control has existed since
Raibert & Craig (1981) and Mason (1981) — see the classical-canon row in table (c) — but stability
guarantees *degraded by an estimator's error rate*
for a tactile contact-mode monitor do not appear in the literature surveyed. This is the natural
"control-theoretic use of touch" that the field is missing between C1 (detectors) and C5
(certificates).

### O5. **Minimal-tactile-sensing design: how much touch is needed for observability?**
**Claim.** For a specified set of contact modes and geometries, there is a **minimum number and
placement of independent tactile measurement channels** for (local) observability / attaining a
target CRB, computable as a combinatorial/continuous sensor-placement problem with the O1
information matrix as the objective.
**Theorem to prove.** A lower bound on channel count for observability of a given mode set (with
an explicit construction achieving it), plus monotonicity/submodularity structure that makes greedy
placement near-optimal.
**Why it is unoccupied.** Tactile-hardware papers choose taxel counts and layouts by design
intuition; no work found ties layout to an observability or estimation bound. Directly useful to
anyone designing the next sensor — and it gives a *theoretical* reason for a hardware choice,
which is rare and publishable.

---

## 4. Single best unoccupied opportunity + biggest prior-art risk

### 4.0 The single best opportunity

> **"Identifiability-limited contact estimation and certified contact control": a
> Fisher-information/observability theory of what touch can and cannot know about contact state
> (O1 + O3 + O5), wired into a control certificate that provably preserves a contact invariant from
> a coverage guarantee (O2 + O4).**

One sentence of content: **prove that the tactually identifiable part of contact state is a
low-dimensional subspace, characterise it with a mode-dependent Cramér–Rao/observability bound,
show that a one-line contact statistic is the sufficient statistic on part of that subspace (hence
learned monitors there are re-encodings, not information extractors), and then certify a contact
controller on the identifiable subspace via a coverage-to-invariance lemma.**

Why this is the right shape for an ACT/Diffusion-Policy-ambition contribution:
- it is a **mechanism** ("what information does touch carry about contact, and what control
  guarantees does that buy?"), not a benchmark delta;
- it **carries theorems** (hybrid observability rank condition, CRB, sufficiency, persistence of
  excitation, coverage-to-invariance, dwell-time stability under mis-estimation);
- it **explains an empirical anomaly the field keeps re-encountering** (learned monitors saturating
  near simple statistics) and turns it into a *prediction*, then a *design rule*;
- it is **robust to the 2026 stampede**: even if a dozen more tactile diffusion policies appear,
  none of them produce a bound on what the tactile channel can know. The conformal/CBF papers
  calibrate disturbances or future configurations; the tactile-estimation papers (TacGraph,
  particle-filter pose) are point estimators with no bounds; the certified-contact paper has no
  sensing in the loop.
- it is **measurable on existing infrastructure** — you can compute empirical CRBs and sufficiency
  gaps on public datasets (SlipSense-scale, GelSight marker data), so the theory is falsifiable
  without new hardware, and the certificate can be demonstrated on a Franka/insertion task.

### 4.1 Biggest prior-art risk (and the "this is just classical X" charges)

| risk | the charge | why it is not fatal / how to answer |
|---|---|---|
| **R1 (highest).** Classical *observability of contact* / *Bayesian hybrid model-state estimation* already did "contact state + geometry estimation with a formal model". [Gadeyne, Lefebvre & Bruyninckx, IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364905056196); [Lefebvre, Bruyninckx & De Schutter, IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364904050079); [Jia & Erdmann, Montana-based observability of contact](https://www.ri.cmu.edu/pub_files/pub1/jia_yan_bin_1997_1/jia_yan_bin_1997_1.pdf); [Debus & Dupont HMM contact state](https://www.semanticscholar.org/paper/Contact-State-Estimation-Using-Multiple-Model-and-Debus-Dupont/bf75f4b0117c1aede5842e2af9093c04caf73649); [Koval manifold particle filters](https://www.ri.cmu.edu/pub_files/2013/12/koval2013manifold_isrr.pdf) | "**This is just classical contact-formation observability with tactile arrays swapped for F/T.**" | The classical line assumes (a) *a known finite catalogue of ideal contact formations*, and (b) *point-wrench* observations, for which observability is largely combinatorial. Tactile arrays give a **spatially distributed, mode-dependent, high-dimensional nonlinear output map**; the classical framework has no notion of *rank deficiency as a function of spatial resolution*, no CRB on contact geometry, no *sufficient-statistic* result, and no coverage guarantee. State this explicitly as the delta. Concretely: reproduce one classical result (e.g. Debus HMM contact-state) as a special case of your rank condition — that is the strongest possible defence. |
| **R2.** "This is just conformal prediction + CBF." | [Liu et al. IROS 2026](https://arxiv.org/abs/2607.00424) (ESO + sliding-window conformal → robust CBF) and [CaPTURe COPA 2026](https://arxiv.org/abs/2608.09166) (conformal prediction regions for contact-rich configuration space) | They calibrate a **continuous disturbance magnitude** or a **future configuration region**. Neither calibrates a **hybrid discrete contact mode** jointly with continuous geometry, and neither *derives* the control certificate from the estimator's coverage radius (the coverage-to-invariance lemma is the new step). Your object of estimation is structurally different (a mode, i.e. a discrete latent). |
| **R3.** "This is just TacGraph plus a certificate." | [TacGraph RA-L 2026](https://arxiv.org/abs/2512.23856) already does learned tactile modules + physics-constrained factor graph for pose + **extrinsic contact** | TacGraph is a **MAP point estimator with no error bounds and no controller**. If you only attach conformal calibration to a TacGraph-style estimator, the contribution is thin; the theory must produce a *bound* (observability/CRB) that explains **where** TacGraph-style structure is necessary versus where a scalar suffices. |
| **R4.** "This is just certified contact-rich synthesis." | [Li & Chou, RSS 2026](https://www.roboticsproceedings.org/rss22/p190.html) already guarantees constraint satisfaction + goal reachability on the true hybrid dynamics for in-hand dexterous manipulation | That work assumes the model is known and **no tactile sensing in the loop**; it quantifies *model smoothing error*, not *sensing error*. Your set-valued deviation is the **estimator's** set, and the coupling to the controller certificate is the contribution. Cite it as the complementary half. |
| **R5.** "Passivity/energy-tank tactile control already gives guarantees." | [VA-UFIC](https://arxiv.org/abs/2408.14219), [Tactile-Morph Skills](https://arxiv.org/abs/2408.12285) | Energy tanks guarantee *passivity of the interaction*, not *correctness of contact-state knowledge*. Different guarantee class; state the distinction. |
| **R6.** "The one-line-statistic phenomenon is folklore." | no publication found (§4(f)); [Rotella et al. 2017](https://arxiv.org/abs/1709.07472) report the *opposite* direction (learning beats a normal-force threshold) | Do **not** assert it as known. Present it as a **prediction of your sufficiency/CRB theorem**, and report the empirical test as a result. This converts a possible "everyone knows that" objection into a contribution. Also note the classical ancestry: [Howe & Cutkosky 1989 acceleration-threshold sensing skin](https://cir.nii.ac.jp/crid/1360016869742020992) and [Muthusamy et al. 2020 threshold method at 2 kHz](https://arxiv.org/abs/2004.07386) show threshold statistics were the *original* mechanism. |

### 4(f) Explicit answer to the "one-line statistic" question

**Is it a known published observation? No — I could not find any paper asserting it.**
What exists instead is evidence pointing both ways:

*Against the claim (learning wins):*
- [SlipSense (CoRL 2026)](https://arxiv.org/abs/2609.15910): 96.7% Macro F1, FPR < 1.6%, with explicit baselines.
- [Foundational world models / conformal monitor (ICRA 2026)](https://arxiv.org/abs/2603.06987): "considerably outperforms statistical techniques".
- [Rotella et al. 2017](https://arxiv.org/abs/1709.07472): contact probability estimator beats a "purely measured normal force" contact determination.

*For the claim (statistics are enough / learning has structural failure modes):*
- The classical slip mechanism **is** a one-line statistic: [Howe & Cutkosky's acceleration threshold (ICRA 1989)](https://cir.nii.ac.jp/crid/1360016869742020992); [Muthusamy et al.](https://arxiv.org/abs/2004.07386) use an *autonomously sampled noise threshold* and still reach 2 kHz incipient-slip detection with 500 µs resolution and successful suppression.
- [Koval et al. (ISRR 2013)](https://www.ri.cmu.edu/pub_files/2013/12/koval2013manifold_isrr.pdf) report the opposite-direction structural anomaly that "the particle filter *performs worse as sensor resolution or the update rate increases*" — i.e. more tactile data did not help a contact-state estimator.
- [Dong et al.'s dense slip field](https://arxiv.org/abs/1810.13381) is a **training-free analytic** criterion beating many learned detectors' reported accuracies (86.25% vs typical 94–97% on *different* benchmarks, so not directly comparable — flag this).

**Verdict:** unverified as a literature fact; best treated as a **hypothesis your theory predicts**
(O1's sufficiency/CRB statement). A clean negative/structural result here would be a genuine
mechanism-level contribution, because it converts an anecdote into a bound.

---

## 5. What I could not verify

**Tool limitations (affect the whole survey):**
1. **No PDF full texts were read.** `web_fetch` returns `unsupported content type "application/pdf"`,
   so RSS 2026 p190, the HAL tactile-servoing PDF, all IEEE PDFs, MERL TRs and the CMU tech report
   were *not* read. Method details in the table come from abstracts/proceedings HTML only.
2. **The arXiv Atom API broke mid-survey** (`unsupported content type "unknown"`), so systematic
   arXiv coverage stops partway. Works that exist only on arXiv and were not surfaced by web search
   may be missing entirely.
3. **Semantic Scholar API was intermittently HTTP 429**, and the S2 paper page returned HTTP 202 with
   no body. `huggingface.co` is unreachable from this sandbox (`non-public IP address`).
4. **The ICRA 2026 tactile/force topic wiki could not be fetched** (`fetch failed`), and IEEE Xplore
   proceedings cannot be enumerated here. **Consequence: the novelty check against accepted-but-
   unindexed 2026 papers is incomplete.** This is the single largest risk to the "unoccupied"
   claims in §3–§4.

**Specific unverified items:**
5. **Debus & Dupont, "Contact State Estimation Using Multiple Model Estimation and Hidden Markov
   Models"** — title and URL verified; the widely cited venue/year (IJRR 2004) was **not**
   independently confirmed (S2 and Dimensions pages both failed to render).
6. **Gadeyne, Lefebvre & Bruyninckx (IJRR 2005)** — title + SAGE DOI page title verified; the
   abstract and the exact volume/pages were not read.
7. **Jia & Erdmann** — only seen as a 1997 CMU PDF snippet invoking Montana's equations for a
   nonlinear contact system; the exact title, venue and the *content* of its observability result
   were not verified. Do not cite it as a specific theorem without reading it.
8. **Howe & Cutkosky 1989** — title/venue verified via search snippets only; page numbers and the
   exact statistic used were not read.
9. **ContactGuard (arXiv 2608.13438)** — only the title, one snippet ("scores the consequence of the
   specific pending action") and a French-language summary were obtainable. Its actual monitoring
   mechanism and guarantee status are unverified.
10. **The "one-line statistic" claim (f)** — no publication found either asserting or refuting it.
11. **Classical canon** (Whitney 1977; Salisbury 1980; Raibert & Craig 1981; Mason 1981; Hogan 1985)
    — cited from standard knowledge; **not fetched this session**, so no URLs are given.
12. **Existence of a dedicated "tactile servoing" survey** — search surfaced
    [a 2026 review on adaptive intelligence in tactile sensing robotic hands](https://openurl.ebsco.com/EPDB%3Agcd%3A5%3A39940181/detailv2?sid=ebsco%3Aocu%3Arecord&id=ebsco%3Adoi%3A10.1007%2Fs44245-026-00275-y)
    and [a tactile multimodal-fusion survey (arXiv 2605.17336)](https://export.arxiv.org/pdf/2605.17336)
    but I could not confirm a survey dedicated specifically to tactile servoing
    (Lepora's IJRR 2023 paper contains a tactile-servoing review section, which is the strongest
    artefact I verified).
13. **Coverage completeness for (e):** the conformal-robot list is certainly **incomplete** — query
    results were bounded at 8–10 items per search and arXiv API enumeration failed. Treat §2 C4 as
    "crowded" rather than "exhaustively enumerated".
14. **MERL TR2026-070 (M-VTOP)**, **arXiv 2608.24162**, **arXiv 2609.27381**, **arXiv 2512.23856** —
    verified by title/URL/project page but abstracts not all fetched (TacGraph's project page *was*
    fetched and is trustworthy).

**Searches that returned nothing useful** (relevant negative evidence for §3):
`tactile + observability`, `tactile + Cramér–Rao / Fisher information for contact`, `tactile + set
membership / interval estimation`, `risk-controlling prediction sets for contact monitors`,
`learned tactile monitor vs one-line statistic`. These all came back empty or with only tangential
hits, which is why O1/O2/O5 are presented as open — but see limitation 2/4.

---

## Appendix: reproduction list of the highest-value URLs

- https://www.roboticsproceedings.org/rss22/p190.html — RSS 2026 certifiable contact-rich manipulation
- https://arxiv.org/abs/2608.09166 — CaPTURe, conformal prediction for contact-aware uncertainty
- https://arxiv.org/abs/2607.00424 — conformal disturbance bounds + robust CBF, 1 kHz
- https://arxiv.org/abs/2603.06987 — conformal world-model failure monitor (ICRA 2026)
- https://arxiv.org/abs/2606.14188 — conformal-calibrated reachable tubes, contact-rich MPC
- https://arxiv.org/abs/2410.10391 / https://arxiv.org/abs/2307.13977 — reachset conformance / formal verification of contact tasks
- https://arxiv.org/abs/2609.15910 — SlipSense (CoRL 2026), the current slip-detection frontier
- https://arxiv.org/abs/2509.09546 — neuromorphic incipient slip, ≥360 ms lead time
- https://arxiv.org/abs/2402.11879 — vibration-injection stick-ratio estimation + control (RA-L 2024)
- https://arxiv.org/abs/1810.13381 — dense slip field, training-free
- https://arxiv.org/abs/2512.23856 + https://tacgraph.github.io/ — TacGraph factor-graph contact estimation (RA-L 2026)
- https://arxiv.org/abs/2603.21410 — Bayesian active 6D pose + class, particle filter
- https://arxiv.org/abs/2312.08411 — IJRR 2023 pose+shear tactile servoing with SE(3) Bayesian filter
- https://hal.science/hal-05441031 — RA-L 2026 hybrid force–position tactile servoing at the CoP
- https://arxiv.org/abs/2508.20561 — SimShear (CoRL 2025)
- https://arxiv.org/abs/1803.08014 — iSAM contact-formation + pose state estimation
- https://journals.sagepub.com/doi/10.1177/0278364905056196 — Bayesian hybrid model-state estimation (IJRR 2005)
- https://journals.sagepub.com/doi/10.1177/0278364904050079 — active sensing for compliant motion (IJRR 2005)
- https://www.roboticsproceedings.org/rss10/p51.html — CARE compliance+registration KF (RSS 2014)
- https://arxiv.org/abs/2408.14219 + https://arxiv.org/abs/2408.12285 — energy-tank / energy-budget tactile control
- https://arxiv.org/abs/2404.03943 — POMDP over contact configuration for insertion
