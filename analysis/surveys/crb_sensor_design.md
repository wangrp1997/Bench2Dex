# Fisher information / CRB and sensor-design theory for tactile & contact sensing

Focused theory survey, 2026-09-30. Angle: *does a Fisher-information / Cramér-Rao analysis of
contact-state estimation from a **distributed** tactile output map (depth field + normals,
~57,600 measurements) already exist, and is there any tactile sensor-design theory giving a
resolution-vs-accuracy law?*

Method: `web_search` + `web_fetch` over arXiv (Atom API), Crossref API, publisher pages
(MDPI/IOP/Springer/NSF-PAR/IEEE), plus Semantic Scholar (mostly HTTP 429 rate-limited).
Every entry below is tagged with how far it was verified. **All fetched content treated as
data.**

---

## 1. Table: work | venue/year | what is computed | sensor type | result

| Work | Venue / year | What is computed | Sensor type | Result | Verification |
|---|---|---|---|---|---|
| [Sun & Martius, *Theory and Design of Super-resolution Haptic Skins*](https://arxiv.org/abs/2105.11914) | arXiv:2105.11914v2, 2021 ([v1 titled *Theory of Geometric Super-resolution for Haptic Sensor Design*](https://arxiv.org/pdf/2105.11914v1)); no journal-ref on the abs page | First-order **error propagation of contact position and force** from taxel-value isolines (TVIs); minimal force for localisation; multi-contact distinguishability; FEM study of material/structure | Sparse taxels embedded in elastomer (barometers; also strain-gauge, accelerometer discussion); 1D 6-taxel and 2D 5×5 = 25-barometer sensors | Linearised law **σ_P = 2σ_S/(m₁+m₂)** with mᵢ = TVI slopes; super-resolution factor **Ω = D/(n·2σ_P)**; analytic **Ω = Dλα(D/2)^{α−1}/(4σ_S)** for 2 taxels (D = taxel spacing, α = attenuation exponent, σ_S = taxel noise); best α = 2; measured Ω ≈ 109 (ML) vs 165 (theory) in 1D, >1200 in 2D | **Full text read** (arXiv HTML) |
| [Cameron, *Optimal tactile sensor placement*](https://doi.org/10.1109/robot.1989.100006) | ICRA 1989, pp. 308–313 | Classical **Bayesian / decision-theoretic sensor placement** (refs: Wald *Sequential Analysis*, Blackwell, DeGroot, Berger, Durrant-Whyte) | Tactile sensor placement (not a CRB/FIM derivation) | Earliest dedicated "optimal tactile sensor placement" work; information/decision-theoretic objective | Crossref metadata verified; **content not read** |
| [Cameron & Durrant-Whyte, *A Bayesian Approach to Optimal Sensor Placement*](https://doi.org/10.1177/027836499000900505) | IJRR 9(5), 1990 | Bayesian/information-based placement objective | generic sensing (incl. tactile motivation) | The classical "information-based sensor placement" reference reused here | **Partially verified** (DOI seen in search snippet + cited in the 1989 Crossref record) |
| [Pathak, Vaskevicius & Birk, *Uncertainty analysis for optimum plane extraction from noisy 3D range-sensor point-clouds*](https://doi.org/10.1007/s11370-009-0057-4) | Intelligent Service Robotics 3(1):37–48, 2010 | **Uncertainty/CRB of plane parameters (n, d) fitted to noisy 3D points**; cites Kanazawa & Kanatani 1995 *Reliability of fitting a plane to range data* | 3D range sensor point clouds (not tactile) | Closest existing analysis to our `(n, d)` block: plane-parameter covariance as a function of point noise and point-cloud geometry | Crossref verified; **full text not read** (paywalled) |
| [Pathak, Vaskevicius & Birk, *Revisiting uncertainty analysis for optimum planes extracted from 3D range sensor point-clouds*](https://doi.org/10.1109/ROBOT.2009.5152502) | ICRA 2009, pp. 1631–1636 | Same, conference version | range point clouds | idem | DOI verified via Crossref reference list of the 2010 paper |
| Quantised-observation CRB (three independent works): [quantized sinewaves](https://doi.org/10.1109/imtc.2004.1351414) (IMTC 2004); [quantized RSSI localization](https://doi.org/10.1109/icpads.2005.118) (ICPADS 2005); [quantized mmWave PMCW MIMO radar](https://doi.org/10.1109/ieeeconf51394.2020.9443365) (Asilomar 2020) | 2004 / 2005 / 2020 | **CRB under quantised observations** (finite bit depth) | generic (noise-shaping, radar, WSN) | The formal machinery for a bit-depth-limited CRB exists and is mature; **no tactile application found** | Crossref metadata verified; contents not read |
| [Kamireddypalli et al., *BayesContact*](https://arxiv.org/abs/2607.16123) | arXiv:2607.16123, Jul 2026 | **Particle belief over object pose**, simulation-based inference of depth + F/T contact likelihoods; information-gain probing | depth (vision) + force/torque contact evidence | "improves pose observability … by 30%"; **no FIM/CRB**; uncertainty is amortised/sampled, not analysed | Abstract read |
| [ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation](https://doi.org/10.15607/rss.2025.xxi.054) | RSS XXI, 2025 | Implicit neural representation; in-hand pose + extrinsic contact | vision-based tactile | contact-state estimation, no error bound | Crossref metadata only |
| *[Simultaneous Extrinsic Contact and In-Hand Pose Estimation via Distributed Tactile Sensing](https://doi.org/10.1109/lra.2026.3653324)* | IEEE RA-L 2026 | Extrinsic contact + in-hand pose from **distributed tactile** | distributed tactile array | **Most on-topic title found** — but I could not retrieve the abstract/body (IEEE blocked; [NSF PAR record](https://par.nsf.gov/biblio/10685566/media/xml) also failed to fetch). Whether it does a CRB is **unverified** — *must be checked by hand* | Crossref title/DOI only |
| [Bimbo et al., *Tactile Sensor-based Estimation of Grasp Force and Contact State with Soft Fingers*](https://doi.org/10.1109/lra.2025.3568616) | IEEE RA-L 2025 | grasp force + contact state estimation | soft tactile fingers | no CRB seen in metadata | Crossref only |
| [*A Novel Inverse Solution of Contact Force Based on a Sparse Tactile Sensor Array*](https://doi.org/10.3390/s18020351) | Sensors 18(2):351, 2018 | **Inverse problem** for contact force from a sparse array | sparse taxel array | A tactile inverse problem; plausibly discusses conditioning/regularisation | MDPI returned **403**; title/DOI from Crossref |
| EIT-based tactile sensing ([Eng. Res. Express, DOI 10.1088/2631-8695/acc515](https://doi.org/10.1088/2631-8695/acc515)) | IOP, 2023 | multi-contact experiments on a compliant EIT sensor | EIT (electrical impedance tomography) tactile | EIT inverse problems are the textbook ill-posed/regularised case in tactile sensing | Search snippets only, **fetch blocked by redirect** |
| [*Sub-Frame Contact-Onset Estimation in a Self-Calibrated BJT Thermal Pixel Array*](https://doi.org/10.3390/s26134074) | Sensors 26(13):4074, 2026 | Estimation-theoretic (SSR, Nelder–Mead) **contact-onset timing** from a 16×16 thermal pixel array | thermal pixel array (4 Hz) | Estimation theory applied to a tactile pixel array — but for **temporal onset**, not contact plane/pose | Search snippets only |
| [Six-axis F/T sensor design by **sensitivity isotropy**](https://www.sciencedirect.com/science/article/abs/pii/S026322412201065X); [isotropy of Stewart-platform six-axis F/T sensor](https://en.cnki.com.cn/Article_en/CJFDTOTAL-HKXB200203011.htm) | Measurement 2022; Chinese J. Aeronautics 2002 | **Condition number / isotropy** of the Jacobian (calibration) matrix as the design objective | strain-gauge 6-axis F/T sensors | Classical design theory: **choose geometry to make the measurement Jacobian well-conditioned** | Search-result titles only |
| [*On the Comparison of Gauge Freedom Handling in Optimization-Based Visual-Inertial State Estimation*](https://doi.org/10.1109/LRA.2018.2833152) | IEEE RA-L 2018 | **Gauge freedom** (unobservable directions) in nonlinear least squares; effect on covariance/conditioning | VIO / SLAM | The borrowable formalism for "rank deficiency / gauge directions" in our FIM | Search snippet only |
| Event-based optical tactile sensing ([Funk et al., CVPRW 2025 event-vision slides](https://tub-rip.github.io/eventvision2025/slides/2025CVPRW_Niklas_Funk.pdf)) | CVPRW 2025 | event-driven tactile sensing hardware | event-based optical tactile | Adjacent design line (event threshold vs latency/bandwidth); **no CRB found** | Search result only |
| Classical **observability of contact** (Jia & Erdmann 1997; Gadeyne/Lefebvre/Bruyninckx IJRR 2005; Debus & Dupont; Koval) | 1997–2013 | Observability/identifiability of *contact formation* with a catalogue of ideal contacts | point wrench / F/T | The closest "rank/observability of contact" theory — but force-based, combinatorial, no spatial resolution | Summarised from the parent's [survey2/estimation_control.md](../survey2/estimation_control.md); **not re-verified here** |

---

## 2. ALREADY KNOWN (with URLs)

1. **First-order accuracy theory for contact localisation + force from a sparse tactile skin
   exists, and gives a quantitative resolution law.** Sun & Martius derive
   σ_P = 2σ_S/(m₁+m₂) and Ω = Dλα(D/2)^{α−1}/(4σ_S), i.e. localisation accuracy improves with
   taxel spacing *D*, TVI slope, and attenuation exponent α, and degrades with taxel noise σ_S.
   They also derive the *minimal force* for super-resolution and a two-contact
   distinguishability condition. → <https://arxiv.org/abs/2105.11914>,
   <https://arxiv.org/html/2105.11914v2>
   **This is the single closest existing "resolution-vs-accuracy" theory for tactile sensing.**

2. **Optimal tactile sensor placement is a classical problem** (Cameron, ICRA 1989,
   <https://doi.org/10.1109/robot.1989.100006>; Cameron & Durrant-Whyte, IJRR 1990,
   <https://doi.org/10.1177/027836499000900505>) and was solved in a **Bayesian
   decision-theoretic** framework (expected information gain), not via a CRB on contact
   parameters.

3. **CRB / uncertainty analysis for fitting a plane (n, d) to noisy 3D points is mature** —
   Kanazawa & Kanatani 1995 (cited in), Pathak et al. ICRA 2009 / Intel Serv Robotics 2010
   (<https://doi.org/10.1007/s11370-009-0057-4>). This is the exact mathematical content of the
   `(n, d)` block of the group's FIM, just outside robotics-tactile.

4. **CRB under quantised observations is classical** (IMTC 2004, ICPADS 2005, Asilomar 2020 —
   URLs in the table). The group's σ = (h_max/255)/√12 assumption is a standard uniform-quantiser
   noise model with a known CRB treatment.

5. **Ill-posedness of tactile inverse problems is a known phenomenon in specific sensor
   families** — EIT-based tactile sensing (<https://doi.org/10.1088/2631-8695/acc515>) and
   sparse-array contact-force inversion (<https://doi.org/10.3390/s18020351>). This is
   *regularisation* practice, not a conditioning/rank theory of contact-state estimation.

6. **Gauge freedom / unobservable directions are standard in state estimation**
   (<https://doi.org/10.1109/LRA.2018.2833152>), and **classical observability of contact** is
   force/contact-formation based (Jia & Erdmann; Gadeyne et al.; Debus & Dupont; Koval).

7. **F/T sensor design has a genuine design theory** based on **sensitivity isotropy / condition
   number of the calibration Jacobian** (<https://www.sciencedirect.com/science/article/abs/pii/S026322412201065X>,
   <https://en.cnki.com.cn/Article_en/CJFDTOTAL-HKXB200203011.htm>). This is the clearest existing
   "make-the-Jacobian-well-conditioned" design methodology in touch sensing — but for a
   **6-DOF point wrench sensor**, not a distributed surface.

8. **Contact-state estimation from distributed / vision-based tactile sensing is an active
   2025–2026 topic** — ViTaSCOPE (RSS 2025), BayesContact (arXiv:2607.16123), RA-L 2026
   *Simultaneous Extrinsic Contact and In-Hand Pose Estimation via Distributed Tactile Sensing* —
   but these are **estimators and uncertainty-propagation systems, not information-theoretic
   bounds**.

---

## 3. NOVEL (with the URLs showing the absence)

**N1. A Fisher-information / CRB analysis of contact-state (plane + object pose) estimation
from a *distributed* tactile surface was not found anywhere.**
Evidence of absence:
- arXiv Atom API, searched metadata (title/abstract/authors/comments):
  `all:"Fisher information" AND all:tactile` → **0 results** while `all:"Fisher information"`
  alone → 4427 and `all:tactile` alone → 2205;
  `abs:"Cramer-Rao" AND abs:tactile` → **0 results** while `abs:"Cramer-Rao"` → 1005 and
  `abs:"Cramer-Rao" AND abs:robot` → 9 (conjunction is working, so the 0 is real).
  Query URLs: <https://export.arxiv.org/api/query?search_query=all:%22Fisher+information%22+AND+all:tactile&start=0&max_results=40>
  and <https://export.arxiv.org/api/query?search_query=abs:%22Cramer-Rao%22+AND+abs:tactile&start=0&max_results=40>
- Crossref title search `query.title=Cramer-Rao+tactile` (top 20) →
  <https://api.crossref.org/works?query.title=Cramer-Rao+tactile&rows=20> contains **no**
  tactile/contact paper (only generic CRB theory and signal processing). Same for
  `query.title=Fisher+information+tactile`:
  <https://api.crossref.org/works?query.title=Fisher+information+tactile&rows=20> → only
  "Tactile Information Processing" (1986) and "Tactile Information Coding by Electro-tactile
  Feedback" (2020), neither a CRB.
- The parent's own negative: tactile + observability and tactile + CRB searches returned no
  tactile observability analysis (see <../survey2/estimation_control.md>).
- The nearest things found (Sun & Martius; BayesContact; RA-L 2026 distributed-tactile paper)
  either do **not** use FIM/CRB, or their content could not be read.

**N2. No quantitative law connecting sensor *spatial sampling / surface extent* and *object
curvature* to achievable contact-localisation accuracy was found.**
Sun & Martius's Ω law depends on taxel spacing *D*, α and σ_S, but is derived for a **few
taxels** (n = 2 or a short line) and models a *point force*, with **no surface extent, no
contact patch size, and no object curvature term**. No tactile paper was found expressing
localisation variance ∝ 1/(spatial second moment of the active patch) or ∝ 1/(aperture × √SNR).
The only place a patch-extent law is known to exist is the **plane-fitting / surface-normal
estimation** literature (Kanazawa & Kanatani 1995, cited in
<https://doi.org/10.1007/s11370-009-0057-4>) — never transferred to tactile sensing.

**N3. "Ill-conditioning / rank deficiency of contact-state estimation from tactile data" is not
a characterised phenomenon.** Searches surfaced only (i) regularised EIT tactile inversion,
(ii) sparse-array force inversion, (iii) generic gauge-freedom theory in SLAM. No work states
the *rank/conditioning structure of the contact-estimation FIM as a function of spatial
resolution*.

**N4. A 57,600-measurement depth+normal map is a different regime from everything found.**
All tactile accuracy theory found is **few-taxel** (2–25 units). Nothing treats a *dense* field
where the number of scalar measurements (~5.8×10⁴) vastly exceeds the 3–6 contact parameters,
so the interesting question is no longer "how many taxels" but "**which directions in parameter
space are weakly excited by the active patch**" — i.e. the ill-conditioning is geometric
(spatial second moment / gauge directions), not information-count. **No paper was found that
poses this question.**

**N5. Gauge freedom of the contact-plane parametrisation.** Writing the measurement model as
h = x·n − d makes (n, d) → (n/s, d/s) a **scale gauge** (exactly the redundancy handled
explicitly in the group's T2/T3 code), and the plane↔pose split introduces further gauge
directions. This is standard in total-least-squares plane fitting, but **no tactile paper was
found that names or uses this gauge freedom.**

---

## 4. Closest existing analysis, and how our distributed-map CRB differs

**Closest overall: Sun & Martius (arXiv:2105.11914).**
They formalise exactly the intuition our FIM quantifies — accuracy of contact position and
force from an elastic medium with embedded taxels — and they *do* produce an accuracy-vs-geometry
law. Differences:

| | Sun & Martius 2021 | Our distributed-map CRB |
|---|---|---|
| Sensing model | few taxels (n = 2 … 25), each a scalar function f(F, d) with additive noise | ~57,600 depth scalars over a 240×240 surface *plus known normals* |
| Parameters | contact position + force magnitude (2–3 scalars) | contact **plane** (n_t ∈ ℝ², d) and **object pose** (6-DOF) |
| Method | isoline intersection + first-order error propagation (σ_P = 2σ_S/(m₁+m₂)) | full FIM **F = (1/σ²) Σ_{active} g gᵀ**, g = [x·t₁, x·t₂, −1]; eigenvalue/condition analysis |
| Geometry dependence | taxel spacing D, attenuation exponent α | spatial second moment (extent) of the *active contact patch*, object curvature |
| Multiplicity | distinguishes 2 simultaneous contacts (combinatorial) | rank deficiency / gauge directions for a single distributed contact |
| Ill-conditioning | not discussed | the central finding |

Note the formal relation: their 2-taxel formula σ_P = 2σ_S/(m₁+m₂) is the inverse of the
2×2 FIM trace for two scalar measurements — so **their law can be read as a special
(few-taxel, 1-D, position-only) case of the FIM we compute.** That is the strongest defence
against "this is just error propagation": we should state the inclusion explicitly.

**Second closest: plane-fitting uncertainty (Pathak et al. 2010 + Kanazawa & Kanatani 1995).**
This gives the `(n, d)`-block covariance for a *passive* point cloud, where the points are
free-standing measurements. Our case inverts the roles: the *measurement* is the depth along
each surface normal, so the Jacobian columns are g = [x·t₁, x·t₂, −1] and the FIM is a
*second-moment (scatter) matrix of the active patch*. Consequence: the condition number is
governed by patch extent and is **independent of the number of surface samples** once the
patch is fixed — a scaling statement the point-cloud literature does not make.

**Third closest: classical contact observability (Jia & Erdmann; Gadeyne; Debus & Dupont).**
These give *rank* statements (which contact formations are distinguishable) from point-wrench
measurements with a finite catalogue. Our contribution would be the *quantitative* spectral
version of the same question for a spatially distributed, high-dimensional output map.

**Fourth closest: F/T sensor isotropy design.** "Make the Jacobian well-conditioned" is already
the design principle for six-axis F/T sensors. Our result would generalise the objective from a
6-DOF point sensor to a contact-state observable subspace of a distributed surface — a
recognisably similar *design philosophy*, new *object*.

---

## 5. What I could NOT verify

1. **Prior art I could not read (highest risk to novelty):**
   *Simultaneous Extrinsic Contact and In-Hand Pose Estimation via Distributed Tactile Sensing*,
   IEEE RA-L 2026, DOI [10.1109/lra.2026.3653324](https://doi.org/10.1109/lra.2026.3653324).
   IEEE Xplore is not fetchable from this session and the
   [NSF PAR mirror](https://par.nsf.gov/biblio/10685566/media/xml) failed. **This title is the
   most dangerous collision risk** and must be read by hand. Also unread:
   [ViTaSCOPE](https://doi.org/10.15607/rss.2025.xxi.054),
   [Bimbo et al. RA-L 2025](https://doi.org/10.1109/lra.2025.3568616),
   [Sensors 18(2):351](https://doi.org/10.3390/s18020351) (MDPI 403),
   [Kanazawa & Kanatani 1995](https://doi.org/10.1007/s11370-009-0057-4) (cited, not read),
   the [six-axis isotropy design paper](https://www.sciencedirect.com/science/article/abs/pii/S026322412201065X),
   and the EIT tactile paper ([IOP redirect blocked](https://doi.org/10.1088/2631-8695/acc515)).
2. **Sun & Martius published venue**: arXiv v2 shows no journal-ref; I could not confirm
   whether/where it appeared (the HTML layout suggests a Science-family supplement). Its
   *content* is verified by full-text read regardless.
3. **Cameron & Durrant-Whyte IJRR 1990** DOI was seen only in a third-party snippet; the
   ICRA-1989 Crossref record corroborates the work's existence, not the exact pages.
4. **OpenAlex API was unreachable** (`fetch failed`) and **bash has no network** (curl exit 28),
   so no OpenAlex/Scopus-scale sweep was possible. **Semantic Scholar API returned HTTP 429**
   on all attempts, so no citation-graph expansion (e.g. "who cites Sun & Martius") was done.
5. **arXiv `all:`/`abs:` searches cover metadata only, not full text.** A paper that mentions
   Fisher information/CRB only in its body would be missed. The negative in §3 N1 is therefore
   strong but not absolute; a Google Scholar full-text phrase search
   (`"Cramér-Rao" "tactile"`, `"Fisher information" "contact localization"`) is still worth
   running by a human.
6. **No venue-by-venue sweep was completed** (T-RO, RA-L, IJRR, RSS/ICRA/CoRL, IEEE Sensors,
   T-IM) because publisher APIs/paywalls blocked systematic retrieval; coverage is via
   web-search surface + Crossref metadata.
7. **No formula from Pathak et al. 2010 was read**, so the claim that plane-normal uncertainty
   scales with patch extent is asserted as the known content of that literature (via its own
   reference to Kanazawa & Kanatani's "Reliability of fitting a plane to range data"), **not**
   quoted.
8. One fetched source was **spam/parked** ([syntouchinc.com](https://syntouchinc.com/) now a
   casino page) — the BioTac paper *Estimating Point of Contact, Force and Torque in a
   Biomimetic Tactile Sensor with Deformable Skin* (Lin et al. 2013) could not be read. This is
   a real candidate for prior CRB-style analysis of a biomimetic tactile sensor and should be
   retrieved from IEEE/another mirror.

---

### One-line bottom line
No published FIM/CRB analysis of contact-state estimation from a **distributed** tactile surface
was found; the closest prior art is a **few-taxel first-order error-propagation resolution law**
(Sun & Martius 2021), classical **optimal tactile sensor placement** (Cameron 1989/1990),
**plane-fit uncertainty/CRB** (Pathak et al. 2010; Kanazawa & Kanatani 1995), **quantised-CRB**
theory, and **F/T-sensor isotropy design** — none of which characterises the rank structure or
ill-conditioning of a 57,600-measurement depth+normal map.
