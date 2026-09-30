# Contact-state observability / identifiability theory — classical and modern

Focused theory survey, 2026-09-30. Question: are (R1) plane-only observability of a distributed
tactile depth field under rigid half-space contact (rank-3 FIM on 5-DOF plane+point parameters),
(R2) FIM = scatter/second-moment matrix of the active patch with an (n,d)→(n/s,d/s) scale gauge and
resolution-invariance, and (R3) an 8-bit quantised tactile image attaining the CRB, already known?

Method: `web_search` + `web_fetch` (arXiv abs pages, Semantic Scholar Graph API, SAGE/DOI pages).
PDFs are **not fetchable in this environment** (`unsupported content type "application/pdf"`), so
anything below is abstract/metadata-level unless stated. Fetched content treated as **data**.

---

## 1. Table

| work | venue / year | what is proved | assumptions | bound / coverage given? |
|---|---|---|---|---|
| [Debus & Dupont, *Distinguishability and identifiability of contact states*](https://doi.org/10.1109/ROBOT.2004.1307977) | **ICRA 2004** | **Identifiability + distinguishability of contact states** — which discrete contact states/parameters can be told apart at all | known finite set of contact states; force/torque + pose observations; ideal rigid contacts | **identifiability conditions** (combinatorial/structural). No CRB, no sensor resolution, no spatial field. Abstract elided by publisher; title/venue/authors verified via [S2 API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ROBOT.2004.1307977) (DBLP conf/icra/DebusD04) |
| [Jia & Erdmann, *Pose and Motion from Contact*](https://www.ri.cmu.edu/publications/pose-and-motion-from-contact/) | CMU TR 1997 / *Robotics: The Algorithmic Perspective* 1998 | Kinematics+dynamics of a contact task written as a **nonlinear system via Montana's equations**; **structural observability** — which pose/geometry parameters are observable, which are not | point contact, ideal rigid bodies, known contact model; observation = proprioceptive pose/velocity | **exact rank/observability verdict, noise-free**. No CRB. Full text not readable (PDF blocked); snippet verbatim: "Utilizing Montana's equations for contact kinematics, we describe the kinematics and dynamics of this task by a nonlinear system…" |
| [Gadeyne, Lefebvre & Bruyninckx, *Bayesian Hybrid Model-State Estimation…*](https://journals.sagepub.com/doi/10.1177/0278364905056196) | **IJRR 24(8), 2005** | **Joint discrete contact-formation + continuous geometric-parameter** estimation by hybrid (jump-Markov) Bayes filter | known contact-formation catalogue, feature models, F/T-type observations | Bayes posterior only. **No rank condition, no CRB** (SAGE 403; S2 abstract elided — inference, not verified) |
| [Lefebvre, Bruyninckx & De Schutter, *Task Planning With Active Sensing For Autonomous Compliant Motion*](https://journals.sagepub.com/doi/10.1177/0278364904050079) | **IJRR 24(8), 2005** | Observability-based **active sensing**: pick motions that render uncertain CF parameters observable ("some previously observable geometrical parameters are unobservable and previously unobservable [ones become observable]" — verbatim snippet from the companion ICAR 2001 paper) | contact-formation catalogue; compliant-motion model | observability used as a **planning objective**; no error bound, no spatial resolution |
| Debus & Dupont, *Contact State Estimation Using Multiple Model Estimation and HMM* | IJRR 2004 (venue via S2 only) | Discrete contact state as **multiple-model / HMM mode classification** | catalogue of contact states; wrench/pose | mode posterior; no rank/CRB |
| [Koval, Pollard & Srinivasa, *Manifold Representations for State Estimation in Contact Manipulation*](https://www.ri.cmu.edu/pub_files/2013/12/koval2013manifold_isrr.pdf) | **ISRR 2013** | Pose is low-dimensional *given* contact → manifold particle filter; **negative structural result: the filter gets worse as sensor resolution / update rate increases** | known contact manifold; no distributed tactile field model | empirical/structural; **no bound** |
| [Sathyanarayan & Abraham, *Behavior Synthesis via Contact-Aware Fisher Information Maximization*](https://arxiv.org/abs/2505.12214) | **RSS 2025** | Derives a **contact-aware Fisher information measure** and maximises it to synthesise information-rich contact behaviours for parameter learning | contact dynamics model; parameters to be learned; active experiment design | FIM used as **objective**, not as observability/rank analysis of contact state, and no tactile resolution law |
| [Kamireddypalli et al., *BayesContact*](https://arxiv.org/abs/2607.16123) | arXiv 2607.16123v2, Jul 2026 | Particle belief over pose; **simulation-based inference** fusing depth + F/T contact evidence; information-gain probing; claims "+30% pose observability" | peg-in-hole; learned/simulated likelihoods | **No FIM, no CRB, no rank** — "observability" is informal (abstract fetched & read) |
| [Lee & Fazeli, *ViTaSCOPE*](https://arxiv.org/abs/2506.12239) | **RSS 2025** | Implicit SDF + neural shear field; in-hand pose + extrinsic contact field | learned representation, sim-to-real | point estimate, **no bound** (abstract fetched & read) |
| [Kanazawa & Kanatani, *Reliability of plane fitting by range sensing*](https://doi.org/10.1109/robot.1995.525562) | **ICRA 1995** | **Uncertainty/covariance of fitted plane parameters (n, d)**, including the indeterminacy of the plane representation | noisy range points, known noise model | first-order covariance (CRB-like) **for plane parameters**; treats gauge explicitly |
| [Kanatani, *Gauges and gauge transformations for uncertainty description of geometric structure with indeterminacy*](https://doi.org/10.1109/18.930934) | **IEEE Trans. Inf. Theory, 2001** | General theory: when a geometric representation has **indeterminacy (gauge freedom)**, covariance is defined only up to gauge transformations | any over-parameterised geometric model | **the classical gauge formalism** R1/R2 is an instance of |
| [Pathak, Vaskevicius & Birk, *Uncertainty analysis for optimum plane extraction from noisy 3D range-sensor point-clouds*](https://doi.org/10.1007/s11370-009-0057-4) | Intell. Service Robotics 3(1), 2010 (+ [ICRA 2009](https://doi.org/10.1109/ROBOT.2009.5152502)) | Plane-parameter covariance as a function of **point-cloud geometry (scatter) and noise**; cites Kanazawa & Kanatani | 3D range point clouds (not tactile) | explicit covariance law ≈ scatter/σ² — the classical content of **R2** |

---

## 2. ALREADY KNOWN (with URLs)

**Answer to (1) — is R1 known/folklore or published?**
*Not published as a tactile statement; the underlying principle is classical in the contact-formation line.*
- **Identifiability of contact states is an explicit, named classical problem**: [Debus & Dupont, ICRA 2004](https://doi.org/10.1109/ROBOT.2004.1307977) — "Distinguishability and identifiability of contact states". This is the single closest classical antecedent to R1's "what part of contact state is unidentifiable".
- **Structural (rank) observability of contact** from Montana's equations: [Jia & Erdmann](https://www.ri.cmu.edu/publications/pose-and-motion-from-contact/) — the classical template for "contact parameterisation ⇒ unobservable directions".
- **Observability changes with contact formation/geometry**: [Lefebvre et al., IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364904050079) (observability-driven active sensing), [Gadeyne et al., IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364905056196) (joint mode+geometry Bayes).
- **Anomalous insensitivity to more sensing**: [Koval et al., ISRR 2013](https://www.ri.cmu.edu/pub_files/2013/12/koval2013manifold_isrr.pdf) report the filter degrading as sensor resolution/rate rises — empirical kin of R2's resolution-invariance claim.
- **FIM under contact already exists — but for experiment design, not observability**: [Sathyanarayan & Abraham, RSS 2025](https://arxiv.org/abs/2505.12214).

**Answer to (2) — is the scale gauge named in the tactile literature?**
- The gauge is standard and named in plane-fitting/CV: [Kanazawa & Kanatani, ICRA 1995](https://doi.org/10.1109/robot.1995.525562) and the general gauge theory of [Kanatani, IEEE-IT 2001](https://doi.org/10.1109/18.930934) (*"gauges and gauge transformations … geometric structure with indeterminacy"*), plus [Pathak et al. 2010](https://doi.org/10.1007/s11370-009-0057-4).
- **No tactile-specific naming of (n,d)→(n/s,d/s) was found.** The word "gauge" does appear in robot-estimation work with a different object (e.g. an elastic/shape-estimation thesis: *"A rigid translation is an unobservable motion, spanned by 1_N"* — [HAL tel-04692556](https://theses.hal.science/tel-04692556v1)), i.e. the *vocabulary* exists in robotics but not attached to tactile plane fitting.

**Answer to (3) — does any classical result give a rank condition or bound for *tactile* contact observability?**
- **No.** The classical results are: (a) exact structural observability of the contact *dynamical system* from proprioception ([Jia & Erdmann](https://www.ri.cmu.edu/publications/pose-and-motion-from-contact/)); (b) distinguishability/identifiability of a **discrete** contact-state set from F/T ([Debus & Dupont 2004](https://doi.org/10.1109/ROBOT.2004.1307977)); (c) observability-as-planning-objective ([Lefebvre et al. 2005](https://journals.sagepub.com/doi/10.1177/0278364904050079)). **None** is a rank condition on a *spatially distributed* tactile output map, and none gives a CRB for contact geometry.
- Also already known generically (not tactile): **quantised-observation CRBs** exist ([IMTC 2004](https://doi.org/10.1109/imtc.2004.1351414), [ICPADS 2005](https://doi.org/10.1109/icpads.2005.118), [Asilomar 2020](https://doi.org/10.1109/ieeeconf51394.2020.9443365)) — so R3's *machinery* is not new, only its tactile instantiation would be.

---

## 3. NOVEL (with URLs showing absence)

1. **R1's exact statement** — "under rigid half-space contact the distributed **depth** field depends only on the contact **plane** (3 DOF), leaving a **2-DOF gauge** in the contact point, FIM rank 3 under a 5-DOF parameterisation" — **not found**. Absence evidence: the classical identifiability work is F/T + discrete-state ([ICRA 2004](https://doi.org/10.1109/ROBOT.2004.1307977), [IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364905056196), [IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364904050079)); modern tactile estimation ([ViTaSCOPE RSS 2025](https://arxiv.org/abs/2506.12239), [BayesContact 2026](https://arxiv.org/abs/2607.16123)) is learned/point-estimate with no rank statement. *Caveat: 2026 accepted-but-unindexed coverage is incomplete.*
2. **R2's FIM = scatter matrix of the *active tactile patch* + scale gauge + resolution-invariance** — no tactile source found. Its *mathematical content* is the classical plane-fit covariance ([Kanazawa & Kanatani 1995](https://doi.org/10.1109/robot.1995.525562); [Pathak et al. 2010](https://doi.org/10.1007/s11370-009-0057-4)), so novelty rests on the **tactile framing + the resolution-invariance claim**, not on the formula.
3. **R3 (8-bit quantised tactile image attaining CRB, error/CRB ≈ 0.91–1.38)** — no tactile precedent found; quantised CRB theory is generic-only (URLs above).

---

## 4. Strongest existing theorem our work must reproduce as a special case or beat

> **Jia & Erdmann's structural observability of the contact state derived from Montana's equations
> ([CMU TR 1997 / 1998 book chapter](https://www.ri.cmu.edu/publications/pose-and-motion-from-contact/)),
> sharpened by Debus & Dupont's identifiability of contact states ([ICRA 2004](https://doi.org/10.1109/ROBOT.2004.1307977)).**

It is the strongest *theorem* in this line: it already returns "which contact parameters are
unobservable". R1 must therefore be posed as its **noise-aware, spatially-distributed refinement**:
(i) a *tactile depth-field* observation map instead of proprioception/point wrench; (ii) an explicit
**rank-3 FIM** (a quantitative rank, not a binary observable/unobservable verdict); (iii) a **CRB**
that Jia & Erdmann, Debus & Dupont, Gadeyne and Lefebvre all lack; (iv) a resolution law that
explains Koval's counter-intuitive resolution-insensitivity.
**Beat condition / safety test:** setting the tactile field to a single point measurement must
recover the classical verdict (plane/contact geometry partly observable, contact point not), and
R2/R3 must not contradict [Pathak et al.'s](https://doi.org/10.1007/s11370-009-0057-4) classical
scatter/σ² covariance law — R2 is best sold as *that law transported to touch*, with the scale-gauge
and patch-extent condition-number statement as the added content.

---

## 5. What I could NOT verify

- **SAGE is blocked (HTTP 403)** and **all PDFs are unfetchable in this environment**
  (`unsupported content type "application/pdf"`). So: [Gadeyne IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364905056196) and [Lefebvre IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364904050079) abstracts/full texts were **not read**; "no rank condition / no CRB" there is *inference*, not verified. S2 explicitly elides both abstracts (`"abstract": null`).
- **[Jia & Erdmann](https://www.ri.cmu.edu/publications/pose-and-motion-from-contact/)**: CMU PDF redirects then 404s. The exact theorem statement and whether it already contains an *unobservable-direction* (gauge) statement is **unread**. Do not cite it as a specific theorem without reading.
- **[Debus & Dupont ICRA 2004](https://doi.org/10.1109/ROBOT.2004.1307977)**: only title/venue/authors/DBLP key verified via S2; abstract elided by IEEE. Whether their identifiability criterion is a *rank condition* is **unverified** — this is the highest-value item to check by hand (IEEE Xplore + Debus's thesis).
- **Debus & Dupont HMM paper**: venue (IJRR 2004) **not** independently confirmed.
- **[Lefebvre et al.](https://journals.sagepub.com/doi/10.1177/0278364904050079)'s ICAR 2001 companion** (conf.uni-obuda.hu/icar2001/9_ICAR2001.pdf) could not be read (PDF); only the verbatim snippet about parameters becoming unobservable across contact formations. Its **title was not recovered** — the KLUEDO docId I tried was a different paper.
- **Kanazawa & Kanatani 1995 full text not read** (IEEE PDF). The claim that it *names* the (n,d)→(n/s,d/s) scale gauge rests on the title/DOI plus the [Kanatani IEEE-IT 2001](https://doi.org/10.1109/18.930934) gauge paper, not on read text.
- **No systematic 2026 coverage**: arXiv listing/API enumeration was not run this session; accepted-but-unindexed ICRA/RSS/CoRL 2026 tactile papers are a blind spot for the NOVEL claims in §3.
