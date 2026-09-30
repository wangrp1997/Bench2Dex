# Task-dependence of observability / monitorability
Survey date: 2026-09-30. Focus: when does a task outcome depend only on the *identifiable* part of the state?
Driver: a robotics group found monitorability-from-touch depends on whether the outcome functional depends on contact
AMOUNT (scalar, monitorable) vs contact GEOMETRY/relations (not); their data confounds the two (no high-amount frames
where the relational condition fails), so no discriminating experiment was constructible.

## 1. Table

| work | venue/year | formalism | what it links | applicable to our question? |
|---|---|---|---|---|
| Functional observability / functional observers — Kravaris, arXiv:2501.00167; Fernando–Trinh–Jennings line | eess.SY 2024; IEEE TAC ~2010 | observability of a *functional* `z = L x`, not of `x`; sufficient conditions via Lie derivatives + an **invariance condition** | task functional ↔ reconstructibility from the measured output | **YES — closest exact formalization of Q1; its "invariance condition" is the embryo of Q2** |
| Sensor Observability Analysis / task-space observability — arXiv:2305.06403 | IEEE T-RO 2024 | observability of *task-space directions* from a sensor Jacobian; optimized over configuration/motion | sensor ↔ task-space quantity | YES for Q1 (a second, robotics-native notion) — but it is an optimization, not a monitorability iff-theorem |
| Gauge freedom in BA/SLAM/VIO — Triggs et al., "BA: A Modern Synthesis"; observability-constrained EKF-SLAM (Huang–Dissanayake) | 2000; T-RO 2008–10 | estimator state lives on a gauge orbit; only gauge-invariant quantities are estimable | unobservable directions ↔ estimable invariants | YES as machinery/analogy; stated for the *state*, never lifted to a task functional |
| Approximate Information State — Subramanian, Sinha, Seraj, Mahajan, arXiv:2010.08843 | JMLR 2022 | `ε`-sufficient statistic for predicting future rewards + observations under actions; value loss bounded by `ε` | representation sufficiency ↔ task value | **YES — best "task-relevant sufficiency" definition to build on** |
| Predictive State Representations — Littman–Sutton–Singh; Mixed-Obs PSRs (AAAI 8680) | UAI 2002 / AAAI | state = vector of predictions of future observations | sufficient statistic without latent state | YES, conceptual ancestor |
| Bisimulation metrics / MDP homomorphisms — Ferns–Panangaden–Precup; Castro et al.; task-aware predictive bisimulation | UAI 2004 / ICML | states equivalent iff same reward + bisimilar future | task-driven state *equivalence* | Partially (equivalence, not identifiability of a functional) |
| State abstraction theory — Li–Walsh–Littman; Abel et al. | ICML 2006 / 2019–20 | taxonomy of abstractions preserving optimal value / `Q*` | which state detail a task needs | Background taxonomy |
| PAC-Bayes failure prediction — Farid, Snyder, Ren, Majumdar, arXiv:2202.05894 | RSS/IJRR 2022 | guaranteed FP/FN bounds for a learned failure predictor from observation history; class-conditional tradeoff | monitor error ↔ info available | Partially — **no** sufficiency/observability condition; bounds degrade silently when info is missing |
| MI-based safety observability — Kim, Nakamura, Bajcsy, arXiv:2510.06492 | arXiv v2 2026 | "estimation gap" vs "prediction gap"; MI measure of safety observability + conformal calibration | latent representation ↔ safety-critical variable | **YES — the only task-relative observability *diagnostic* found**, but empirical/heuristic |
| Action-conditional visuo-tactile grasping — Calandra et al., "More Than a Feeling", arXiv:1805.11085 | RA-L 2018 | model predicts outcome of candidate grasp adjustment from raw visuo-tactile | touch ↔ grasp-adjustment outcome | Practical precedent, not theory; no amount-vs-geometry dissociation |
| Task-relevant information bottleneck / perception-constrained control | RLJ 2025; misc. | bottleneck on task-relevant MI | info budget ↔ control | Borrowable framing only |

## 2. ALREADY KNOWN (formalisms to build on / reproduce)
- **"Observability of a function of the state" already has a name: functional observability.** `z = Lx` may be reconstructible
  from the measured output even when `x` is not. For LTI the operative condition is essentially that the unobservable
  subspace be contained in `ker L` — i.e. *"the outcome depends only on the identifiable part of the state"* is already a
  stated condition, one level below task semantics. https://arxiv.org/abs/2501.00167
- **The nonlinear functional-observer design rests on an explicit invariance condition** (adapted from state-space invariance
  conditions in the observer literature) — the closest thing to a task-level gauge statement that exists. Same URL.
- **Task-space observability is defined relative to a task-space Jacobian** and can be maximized by motion/configuration:
  https://arxiv.org/abs/2305.06403 (IEEE T-RO 2024). Reproduce this as the "we already have observability-relative-to-a-task-space"
  citation; note it is a rank/optimization criterion, not a monitorability theorem.
- **Task-relative sufficiency is formalized** as Approximate Information States, with a value-loss bound: https://arxiv.org/abs/2010.08843
- **Safety-relevant information loss in learned latents is now measured**: estimation gaps (observation never reveals the
  quantity) vs prediction gaps (failure visible only after it occurs), argued with a mutual-information diagnostic.
  https://arxiv.org/abs/2510.06492 — reproduce this as the empirical baseline our theorem should sharpen.
- **Guaranteed FP/FN failure prediction exists**: http://arxiv.org/abs/2202.05894 — the guarantee we would *condition on monitorability*.
- **Gauge/observability toolkit**: BA gauge freedom and observability-constrained SLAM give the standard "unobservable
  directions exist; only invariants are estimable" reasoning. http://luthuli.cs.uiuc.edu/%7Edaf/courses/Optimization/Papers/bundleadjust.pdf

## 3. NOVEL (with URLs showing absence)
- **No theorem of the form "outcome `J` is monitorable from sensor `S` iff `J` factors through the `S`-identifiable
  quotient" was found.** Functional observability (arXiv:2501.00167) is the closest *statement* but (a) reconstructs `Lx`
  from I/O of a *known* model, (b) is never phrased relative to a task/outcome functional chosen by the designer,
  (c) never connects to a learned monitor or to FP/FN guarantees.
- **No task-level analogue of SLAM gauge reasoning was found**: gauge arguments stop at the state estimator. Nobody
  appears to lift "estimable = gauge-invariant" to "monitorable = invariant under the sensor's indistinguishability
  relation *and* sufficient for the task functional".
- **No controlled experiment dissociating contact AMOUNT from relational contact GEOMETRY was found** — i.e. no design
  matching contact magnitude across levels of a relational condition (e.g. thumb opposition vs total contact count at
  matched contact area/force) showing a scalar statistic fails while a relational monitor succeeds. Searches return only
  (i) correlational tactile→grasp-success learning (arXiv:1805.11085 and successors), (ii) contact-mode/pose estimation,
  (iii) sensor-placement/motion optimization. https://arxiv.org/abs/1805.11085
- **Consequence for the group's confounded data**: the confound is a *literature-level* gap, not just a local data defect.
  The matched-magnitude discriminating experiment is claimable as novel, and the theory that predicts when it is needed is
  also unclaimed.

## 4. Best formalism to build on, and what a new theorem must add
**Build on functional observability (arXiv:2501.00167) + AIS sufficiency (arXiv:2010.08843); use MI-based safety
observability (arXiv:2510.06492) as the empirical baseline to beat, and PAC-Bayes (arXiv:2202.05894) as the guarantee to
inherit.** A new theorem must add four things:
1. **A sensor-relative indistinguishability relation `~_S`** on states (the "gauge orbit" of the sensor: two states are
   `~_S`-equivalent when no history of `S` can separate them), generalizing the LTI unobservable subspace and the
   functional-observer invariance condition to a learned/empirical sensor.
2. **An iff statement**: `J` is monitorable from `S` **iff** `J` is constant on `~_S`-classes (equivalently, `J` factors
   through the quotient `State / ~_S`). The iff direction, at the level of a chosen outcome functional, is exactly what
   functional observability does not supply.
3. **An approximate version (`ε`-monitorability)** with a bound linking within-class variation of `J` to monitor FP/FN —
   this is the bridge that turns 2510.06492's MI diagnostic into a guarantee and plugs into 2202.05894's PAC-Bayes bounds.
4. **A constructive experiment-design criterion** for the confounded case: when the amount and relational components must
   be decorrelated by a matched-magnitude probe before the implication is falsifiable — directly addressing the group's
   inability to build a discriminating experiment from existing frames.

## 5. Could NOT verify
- Whether the functional-observability literature states the LTI condition in exactly the "unobservable subspace ⊆ ker L"
  form (abstract-level reading only; the nonlinear paper's condition is in terms of Lie derivatives + invariance).
- Whether the *original* functional-observer papers (Fernando/Trinh/Jennings, IEEE TAC ~2010) already contain the
  task-relative phrasing — only the 2024 arXiv paper was read.
- Whether any contact/tactile paper states an identifiability argument (amount vs geometry) *without* running the
  dissociation — not exhaustively searched; only correlational work surfaced.
- A canonical definition of "perception-constrained control" (candidate: perception-aware MPC / Salaris et al.) — unresolved.
- Formal properties (estimator, consistency, scaling) of the MI safety-observability measure in 2510.06492 — unverified.
