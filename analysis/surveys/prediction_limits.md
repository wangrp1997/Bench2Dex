# Theory of when a learned failure/anomaly predictor cannot beat a simple statistic
### Information-theoretic & statistical limits of failure / anomaly / event prediction
Survey date: 2026-09-30. Angle requested: **information-theoretic and statistical limits**.
All claims below were checked against a fetched source unless marked `UNVERIFIED (snippet-only)` or `UNVERIFIED (own derivation)`.
`bash` has **no network** in this session; only `web_search` / `web_fetch` reached the internet, and paywalled IEEE/Elsevier PDFs could not be fetched (see §5).

---

## 1. Table: result | source | statement | tightness / usable as a ceiling?

| # | Result | Source | Statement | Tightness / usable? |
|---|--------|--------|-----------|---------------------|
| 1 | **Fano's inequality** | Cover & Thomas, *Elements of Information Theory*, ch. 2 (standard; <https://en.wikipedia.org/wiki/Fano%27s_inequality>) | `H(P_e) + P_e log(M-1) ≥ H(Y|X)` → lower-bounds Bayes error from conditional entropy. `UNVERIFIED here (textbook-standard)`. | **Weak** as a ceiling: it is an *error* lower bound, requires `H(Y|X)` of a continuous sensor, and is loose except in the near-deterministic regime. Not usable to bound AUC. |
| 2 | **Hellman–Raviv bounds** | *Probability of error, equivocation, and the Chernoff bound*, IEEE Trans. IT 16(4), 1970 — <https://www.semanticscholar.org/paper/ae1c34e48989399c4faa90260647934f44c9d572> `UNVERIFIED (paywalled; snippet-only)` | Sharper upper/lower bounds tying `P_e` to `H(Y|X)` (upper bound on error, i.e. usable in the direction we want, but via entropy). | **Loose.** Bounds the *scalar* error, not the ROC/AUC; entropy-only so geometry-blind. |
| 3 | **Feder–Merhav tight entropy–error bounds** | IEEE Trans. IT 40(1):259–266, 1994 — <https://dlnext.acm.org/doi/10.1109/18.272494>; comment/correction <https://ieeexplore.ieee.org/document/746849> `UNVERIFIED (paywalled; snippet-only)` | Tightest known relation between entropy and error probability (recovers Fano/Hellman–Raviv as limits). | **Tight for error, still not an AUC bound**, and needs the true conditional entropy. |
| 4 | **Chernoff / Bhattacharyya upper bounds on Bayes error** | Generalized forms: *Pattern Recognition Letters* 2014 — <https://www.sciencedirect.com/science/article/abs/pii/S0167865514000166> `UNVERIFIED (paywalled; snippet-only)` | `P_e ≤ √(π₀π₁)·exp(−C(s))` (Chernoff) / `≤ √(π₀π₁)·ρ` (Bhattacharyya): fully computable from the class-conditional densities. | **Usable and computable**, but bounds *error/accuracy*, not AUC or TPR@FPR. |
| 5 | **Best-possible ROC/Neyman–Pearson ceiling from any f-divergence** ⭐ | Mullhaupt & Peng, *Bounding Neyman–Pearson Region with f-Divergences*, arXiv:2505.08899 (2025) — <https://arxiv.org/abs/2505.08899> | Let `D_f(P₊‖P₋)` be any f-divergence. Any test's `(FPR, FNR)` lies above a divergence-dependent boundary → a **lower bound on the NP boundary**, i.e. an **upper bound on achievable TPR at each FPR**, hence on AUC. For hockey-stick f (≈ total variation) the bound is *best possible*; yields a refined Chernoff-α upper bound in closed form, and improves Pinsker for KL. | **The best general AUC/ROC ceiling found.** It is one-sided (a necessary condition), so it is a valid but not exact ceiling — see §4 caveat. |
| 6 | **Arc length of optimal ROC is an f-divergence; lower bound on maximal AUC** | Song Liu, *Estimating the Arc Length of the Optimal ROC Curve and Lower Bounding the Maximal AUC*, arXiv:2110.09651 (NeurIPS 2022) — <https://arxiv.org/abs/2110.09651> | Optimal-ROC arc length = an f-divergence, estimable variationally with rate `O_p(n^{−β/4})`; yields a tractable lower bound on the maximal AUC. | **Usable and estimable from data** (positives+negatives). Gives a *lower* bound on the oracle AUC → an *upper* bound on how much any model is missing. |
| 7 | **Unified f-divergence / ROC / cost-curve theory + generalized Pinsker** | Reid & Williamson, arXiv:0901.0356 (JMLR 2011) — <https://arxiv.org/abs/0901.0356> | Unifies f-divergences, Bregman divergences, proper losses, ROC curves, cost curves, information; tight surrogate bounds + generalized Pinsker inequalities. | **The theoretical backbone**; not a sharp numeric ceiling by itself. |
| 8 | **Gaussian exact AUC↔divergence ceiling** | Classical detection theory: `AUC = Φ(d′/√2)` (e.g. Barrett & Myers, *Foundations of Image Science*, eq. 13.115 region) `UNVERIFIED (paywalled book; snippet-only)`. Our derivation: equal-variance Gaussians give `KL(P₊‖P₋) = d′²/2`, hence `AUC_max = Φ(√KL)` `UNVERIFIED (own derivation)`. | For equal-variance Gaussian sensor statistics the oracle AUC is an **exact closed form in KL**: `AUC_max = Φ(√(KL(P₊‖P₋)))`. | **Exact & tight in the Gaussian case; immediately usable.** Prior-free (AUC is prior-free). Requires Gaussianity of the monitored statistic. |
| 9 | **AUC bounds for Gaussian graphical models; 1−AUC decays exponentially with dimension** | Khajavi & Kuh, arXiv:1605.05776 — <https://arxiv.org/abs/1605.05776> | Analytic upper/lower AUC bounds in terms of KL/ROC; for tree approximations to Gaussian models `1−AUC` decays exponentially as dimension grows. | **Usable**, and directly relevant to high-dimensional (multi-sensor) robot observations. |
| 10 | **Fisher–Neyman factorization / sufficiency** | Standard: <https://en.wikipedia.org/wiki/Sufficient_statistic> `UNVERIFIED (textbook-standard)`. | `T` is sufficient for `Y` iff `P(Y|X) = g(Y,T(X))` for some `g`. | Foundation of the "learned model cannot beat `T`" argument **when `T` is sufficient**. |
| 11 | **Rao–Blackwell / Lehmann–Scheffé** | <https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem>, <https://en.wikipedia.org/wiki/Lehmann%E2%80%93Scheff%C3%A9_theorem> `UNVERIFIED (textbook-standard)` | Conditioning any estimator on a sufficient statistic cannot increase its risk; the (unique) function of a complete sufficient statistic is UMVU. | Rigorous "no gain from more computation" statement — **for risk, not for AUC**, and it requires a loss function. |
| 12 | **Blackwell comparison of experiments** | Blackwell, *Equivalent comparisons of experiments*, Ann. Math. Stat. 24(2), 1953 — <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full> `UNVERIFIED (page not fetched)`. Modern large-sample sharpening: Mu, Pomatto, Strack, Tamuz, *Econometrica* 2021 (referenced in <https://zenodo.org/records/20800031>) `UNVERIFIED (snippet-only)`. | Experiment A dominates B iff every decision problem's risk is no worse under A. | The correct decision-theoretic formalization of "**no method can extract more than this statistic**": dominance, not AUC. |
| 13 | **MASS learning (minimal achievable sufficient statistic)** | Cvitkovic & Koliander, ICML 2019, arXiv:1905.07822 — <https://arxiv.org/abs/1905.07822> | Trains deep nets to produce minimal sufficient statistics *w.r.t. a function class*, using Conserved Differential Information (works for deterministic nets where MI degenerates). | Modern neural treatment of sufficiency. **Usable**; note "minimal" is relative to the function class. |
| 14 | **Minimal-sufficient-statistic / "lossless information compression" for fault detection (control theory)** ⭐ | Li, Ding, Liang, Chen, Xue, arXiv:2208.01291 — <https://arxiv.org/abs/2208.01291> | For nonlinear dynamic systems, derives **existence conditions** for an autoencoder latent variable that is the *minimal sufficient statistic* (an "analog concept ... lossless information compression") for optimal fault detection; then builds the loss/training algorithm. | **Closest existing analogue of the parent's claim in a monitoring setting** (three-tank process, not a robot). Proves optimality *within a specified detection specification*, not "no model can ever do better". |
| 15 | **No-free-lunch for anomaly detection via over-expressive representations** ⭐ | Reiss, Cohen & Hoshen, arXiv:2306.07284 — <https://arxiv.org/abs/2306.07284> | Theoretical toy model giving a fundamental trade-off between **representation sufficiency and over-expressivity**: beyond a point, increasing representation expressivity *degrades* anomaly-detection performance. Stated as a no-free-lunch theorem for AD. | Directly supports "a bigger learned monitor will not help". **Empirical + toy-theoretic**, not a bound for a specific task. |
| 16 | **NL principle / limits of representation learning in AD** | Hoshen, arXiv:2307.11085 (CVPR'23 VAND keynote) — <https://arxiv.org/abs/2307.11085> | Argues the dominant AD paradigm cannot scale indefinitely due to a no-free-lunch principle; priors are needed. | Position paper — **not a usable bound**. |
| 17 | **V-usable information / "usable information under computational constraints"** ⭐ | Xu, Zhao, Song, Stewart, Ermon, ICLR 2020, arXiv:2002.10689 — <https://arxiv.org/abs/2002.10689> | Variational extension of Shannon information relative to an observer/model family: predictive `V`-information `I_V(X→Y)`; **unlike MI it can be created by computation** (violates DPI), and is PAC-estimable in high dimensions. | **The right formalism to make "no learning gain" architecture-relative.** Combined with row 5 it gives an architecture-specific AUC ceiling (apparently not yet done — see §3). |
| 18 | **PAC-Bayes failure-prediction bounds on a real robot** ⭐ | Farid, Snyder, Ren, Majumdar, arXiv:2202.05894 — <https://arxiv.org/abs/2202.05894> | Synthesizes a failure predictor with **guaranteed bounds on false-positive and false-negative rates** via PAC-Bayes, including **class-conditional** bounds trading FP vs FN; validated on a grasping manipulator with a wrist RGB-D camera; bounds closely match empirical error. | **The only robotics failure-prediction work found that carries an actual guarantee.** Bounds *error rates*, not AUC, and is not information-theoretic. |
| 19 | **Mutual-information-based safety observability + "prediction gaps" for robot world models** ⭐ | Kim, Nakamura, Bajcsy, arXiv:2510.06492 — <https://arxiv.org/abs/2510.06492> | Identifies "estimation gaps" (safety quantity not in the observation) and "prediction gaps" (failure not anticipatable); proposes an **MI-based measure of safety observability** and a rollout-based predictability measure; mitigates with privileged multimodal supervision and conformal risk calibration; asks "**when are world-model state representations sufficient for reliable robot control**". | The nearest robotics instantiation of an information-theoretic *observability* diagnostic. It is a **diagnostic, not a proof/ceiling**, and does not prove any hand statistic sufficient. |
| 20 | **VLA failure prediction with a logistic regression on progress features** | Zhu, Liu, Liu, arXiv:2609.21246 (Sept 2026) — <https://arxiv.org/abs/2609.21246> | Shift-aware failure prediction for OpenVLA on LIBERO: ROC-AUC 0.8497 after 60 actions vs 0.7906 without execution-progress features; beats ActProbe and SAFE-MLP baselines. | Counter-evidence for (e): here the *simple* learned model (shared logistic regression) wins. Shows the ceiling is often reached by low-capacity models. |
| 21 | **Deep AD models effectively learn linear mappings; complexity buys little** ⭐ | Sarfraz et al., ICML 2024 position, arXiv:2405.02678 — <https://arxiv.org/abs/2405.02678> | Critical audit: SOTA deep time-series AD "effectively learn linear mappings"; added complexity gives very little improvement; simpler baselines recommended. | **Empirical, no theory.** Directly the phenomenon the parent measured. |
| 22 | **Random guess beats all algorithms under the point-adjust protocol; PCA beats many deep detectors** ⭐ | Sehili & Zhang, TPCTC 2023, arXiv:2308.13068 — <https://arxiv.org/abs/2308.13068> | Under the popular "point-adjust" protocol a **random guess systematically outperforms all algorithms developed so far**; a simple PCA baseline outperforms many recent deep-learning methods on popular benchmarks. | Empirical + evaluation-methodology explanation only. |
| 23 | **"Apparent progress may be illusory" — flawed benchmarks** | Wu & Keogh, arXiv:2009.13807 (IEEE TKDE 2021) — <https://arxiv.org/abs/2009.13807> | Majority of exemplars in the standard Yahoo/Numenta/NASA AD benchmarks suffer from four flaws; many published comparisons are unreliable. | Empirical; explains *evaluation* failure, not a fundamental limit. |
| 24 | **Distribution-free conditional coverage is impossible** ⭐ | Gibbs, Cherian & Candès, arXiv:2305.12616 — <https://arxiv.org/abs/2305.12616> (see also Chen & Li, ICML 2026, arXiv:2512.24139 — <https://arxiv.org/abs/2512.24139>; Bian & Foygel Barber, arXiv:2205.03647 — <https://arxiv.org/abs/2205.03647>) | Conformal prediction gives **marginal** coverage; **exact distribution-free conditional coverage is impossible** in finite samples. Exact coverage is attainable only over a *finite-dimensional* class of shifts/subgroups. | **Provable impossibility** — the sharpest "what is NOT possible" statement for calibrated failure probabilities. |
| 25 | **ECE is not a well-behaved provable quantity** | Nixon et al., arXiv:1904.01685 — <https://arxiv.org/abs/1904.01685> | ECE "has numerous flaws"; conclusions about which recalibration method is best are **drasticallly** changed by binning/norm/class-conditionality choices. | Demonstrates ECE has **no meaningful provable ceiling**; treat ECE claims as metric-dependent. |
| 26 | **Verified calibration: sample complexity of ECE estimation** | Kumar, Liang & Ma, NeurIPS 2019 spotlight, arXiv:1909.10155 — <https://arxiv.org/abs/1909.10155>; code <https://github.com/p-lambda/verified_calibration> | Popular recalibration methods (Platt/temperature scaling) are *less* calibrated than claimed under a debiased, verified ECE estimator; estimating ECE to precision ε needs many samples (binned ECE is biased with few samples). | **Provable-ish**: gives correct-confidence intervals for calibration error, i.e. tells you how many samples you need before an ECE number means anything. |
| 27 | **Calibration ≠ correct posteriors (grouping loss)** ⭐ | Perez-Lebel, Le Morvan, Varoquaux, ICLR 2023, arXiv:2210.16315 — <https://arxiv.org/abs/2210.16315> | "Even a **perfectly calibrated classifier with the best possible accuracy** can have confidence scores that are **far from the true posterior probabilities**" — the *grouping loss*; modern nets exhibit it, notably under distribution shift. | The key "what is NOT provable" result: calibration gives you no per-input guarantee. |
| 28 | **Chow's rule / optimal reject option** | Chow, *On optimum recognition error and reject tradeoff*, IEEE Trans. IT 16(1), 1970 — <https://doi.org/10.1109/TIT.1970.1054406> `UNVERIFIED (paywalled; snippet-only)` | The Bayes-optimal selective predictor thresholds the true posterior; its risk–coverage curve is the **lower envelope** for the task, and the optimal ROC is that of the likelihood ratio. | **Usable as an oracle ceiling** for failure prediction with abstention. Not estimable without the true posterior → an oracle, not a practical bound. |
| 29 | **Selective prediction / risk–coverage in practice (AURC)** | Geifman & El-Yaniv, ICML 2019, arXiv:1901.09192 — <https://arxiv.org/abs/1901.09192>; AURC objective in arXiv:2607.03528 — <https://arxiv.org/abs/2607.03528> | SelectiveNet optimizes classification + rejection jointly; area under the risk–coverage curve (AURC) is the standard metric. | Consistently improves the trade-off but gives **no guarantee** it reaches the Chow oracle. |
| 30 | **Domain-adaptation lower bound: invariant representations provably fail under label shift** ⭐ | Zhao, Tachet des Combes, Zhang, Gordon, ICML 2019, arXiv:1901.09453 — <https://arxiv.org/abs/1901.09453>; Ben-David et al., *A theory of learning from different domains*, <https://doi.org/10.1007/s10994-009-5152-4> `UNVERIFIED (page not fetched)` | An **information-theoretic lower bound on the joint error of *any* domain-adaptation method** that learns invariant representations, when marginal label distributions differ — a fundamental trade-off. Also: a counterexample showing small source error + invariance is *not* sufficient. | **Provable ceiling under distribution shift.** The strongest "you cannot fix this by learning better" result in the shift setting. |
| 31 | **Domain-shift sample-complexity / H-divergence bounds** | Ben-David et al. (above); importance-weighting bounds (Cortes, Mansour, Mohri, NeurIPS 2010) `UNVERIFIED (not fetched)` | Target risk ≤ source risk + `d_HΔH(𝒟_S,𝒟_T)/2` + λ (ideal joint error); λ is not estimable from source data alone. | **Usable bound shape**, but the unestimable λ is exactly the failure mode in robotics. |
| 32 | **Anomaly detection needs an assumption, and rare events need `Ω(1/π)` data** | Rare-event sampling / minimum-volume-set framing (Scott & Nowak; Polonik; Chandola et al. survey) `UNVERIFIED (not fetched)`; standard Bayes-threshold consequence (own analysis) | (i) AD is ill-posed without a notion of "normal region". (ii) With failure prior `π`, the Bayes rule under 0–1 loss predicts "no failure" unless the likelihood ratio exceeds `(1−π)/π` — a provable threshold, and the expected count of positives under `n` samples is `nπ`, so resolving a rate `π` needs `n = Ω(1/π)` per unit precision. | (ii) is **provable and directly usable**: it explains why rare-failure learned monitors cannot be distinguished from a constant predictor with few failures. |

---

## 2. ALREADY KNOWN (established theory we must cite or reproduce)

**Sufficiency ⇒ no information gain (the parent's Proposition 1, but stated correctly).**
Fisher–Neyman factorization: `T` sufficient for `Y` iff `Y ⟂ X | T`; then `P(Y|X) = P(Y|T)`.
Consequently the *best possible* predictor using all of `X` is a function of `T`; the ceiling is the
oracle AUC of `P(Y|T(X))` (<https://en.wikipedia.org/wiki/Sufficient_statistic>). This must be paired with
the **data-processing inequality** (Cover & Thomas) and, for decision-theoretic rigor, with **Blackwell
dominance** (<https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full>).
Guaranteed no-gain *for a given loss* comes from **Rao–Blackwell** and **Lehmann–Scheffé**
(<https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem>,
<https://en.wikipedia.org/wiki/Lehmann%E2%80%93Scheff%C3%A9_theorem>).
⚠️ **Correction to our own theory note:** sufficiency does **not** imply `AUC(f) ≤ AUC(c)` for the *raw* statistic `c`.
It implies `sup_f AUC(f) = AUC(P(Y|c))`, i.e. the ceiling is the best *transform* of `c`. If `P(Y|c)` is
non-monotone in `c` (e.g. a U-shaped risk), a learned model can legitimately beat `AUC(c)` while still
being unable to beat `AUC(P(Y|c))`. Our claim should be "no model beats the likelihood ratio in `c`",
not "no model beats `c`".

**Sample-complexity / no-free-lunch.**
No-free-lunch for anomaly detection with an explicit *sufficiency vs over-expressivity* trade-off:
<https://arxiv.org/abs/2306.07284>; limits-of-AD position: <https://arxiv.org/abs/2307.11085>.
Under distribution/label shift, an information-theoretic **lower bound on the joint error of any
invariant-representation method**: <https://arxiv.org/abs/1901.09453> (+ H-divergence bounds,
<https://doi.org/10.1007/s10994-009-5152-4>). Rare-failure sample cost scales with `1/π` (Bayes-threshold
argument above; standard in rare-event estimation).

**Calibration: what is provable.**
Marginal coverage from conformal prediction; **exact conditional coverage is impossible** distribution-free
(<https://arxiv.org/abs/2305.12616>, <https://arxiv.org/abs/2512.24139>, <https://arxiv.org/abs/2205.03647>).
ECE is not a stable provable object (<https://arxiv.org/abs/1904.01685>) and needs careful, unbiased
estimation with quantified sample cost (<https://arxiv.org/abs/1909.10155>). Even a perfectly calibrated,
maximally accurate classifier need **not** output true posteriors (grouping loss,
<https://arxiv.org/abs/2210.16315>). The optimal selective/risk–coverage frontier is Chow's posterior
threshold rule (<https://doi.org/10.1109/TIT.1970.1054406>), approximated in practice by
<https://arxiv.org/abs/1901.09192>.

**Bounds on binary prediction / AUC.**
`f`-divergence → Neyman–Pearson-region (ROC) lower bound, best possible for hockey-stick f:
<https://arxiv.org/abs/2505.08899>. Optimal-ROC arc length is an `f`-divergence and lower-bounds the maximal
AUC, estimable at rate `O_p(n^{−β/4})`: <https://arxiv.org/abs/2110.09651>. Unified
divergence/ROC/cost-curve theory with generalized Pinsker: <https://arxiv.org/abs/0901.0356>.
Gaussian/average-case AUC bounds: <https://arxiv.org/abs/1605.05776>. Classical entropy–error bounds:
Hellman–Raviv (<https://www.semanticscholar.org/paper/ae1c34e48989399c4faa90260647934f44c9d572>) and
Feder–Merhav (<https://dlnext.acm.org/doi/10.1109/18.272494>); Bhattacharyya/Chernoff upper bounds on Bayes
error (<https://www.sciencedirect.com/science/article/abs/pii/S0167865514000166>).

**Modern treatment of sufficiency for neural predictors.**
MASS / minimal *achievable* sufficient statistics w.r.t. a function class: <https://arxiv.org/abs/1905.07822>.
Predictive sufficiency as a principle for state-space models: <https://arxiv.org/abs/2508.03158>.
`V`-usable information — information relative to a model family, PAC-estimable:
<https://arxiv.org/abs/2002.10689>. Minimal-sufficient-statistic latent for fault detection (proved
existence conditions): <https://arxiv.org/abs/2208.01291>. Representation sufficiency vs over-expressivity:
<https://arxiv.org/abs/2306.07284>. The information-bottleneck lineage (Tishby et al.) is the other standard
reference for "the minimal sufficient statistic is the IB-optimal representation" `UNVERIFIED (not fetched)`.

**Robotics side.**
Failure prediction with PAC-Bayes FP/FN guarantees on a real manipulator: <https://arxiv.org/abs/2202.05894>.
MI-based safety-observability and "prediction gap" diagnostics for robot world models, with a sufficiency
question posed explicitly: <https://arxiv.org/abs/2510.06492>. Recent VLA failure prediction with an
*explicitly low-capacity* (logistic-regression) predictor that beats neural baselines:
<https://arxiv.org/abs/2609.21246>.

---

## 3. NOVEL (what appears ABSENT, with the URLs that show the absence)

**N1. No usable information-theoretic AUC ceiling instantiated for a robot sensor.**
The ingredients exist (rows 5, 6, 8, 17, 18) but the composition does not. Evidence of absence:
* arXiv all-field search `"usable information" "AUC"` → **no results**: <https://arxiv.org/search/?searchtype=all&query=%22usable+information%22+%22AUC%22>
* arXiv all-field search `"information-theoretic" "failure prediction" "robot"` → **no results**: <https://arxiv.org/search/?searchtype=all&query=%22information-theoretic%22+%22failure+prediction%22+%22robot%22>
* arXiv `"sufficient statistic" "failure prediction"` → **no results**: <https://arxiv.org/search/?searchtype=all&query=%22sufficient+statistic%22+%22failure+prediction%22>
  (arXiv's web search covers title/abstract/comments, not full text — see §5.)
* The only robotics failure-prediction guarantee found is PAC-Bayes on FP/FN (error, not AUC):
  <https://arxiv.org/abs/2202.05894>. The only MI-based robot safety diagnostic is observability, not a
  ceiling: <https://arxiv.org/abs/2510.06492>.
**Novel contribution available: instantiate the NP-region `f`-divergence bound (<https://arxiv.org/abs/2505.08899>)
for a robot's class-conditional sensor distributions, and repeat it with the JSD/hockey-stick divergence so
the bound is stated directly in terms of the sensor's mutual information with the failure label.**

**N2. No derivation that a specific one-line robotic monitoring statistic is (minimal) sufficient.**
`"sufficient statistic" "anomaly detection"` returns only 3 papers (<https://arxiv.org/search/?searchtype=all&query=%22sufficient+statistic%22+%22anomaly+detection%22>),
of which the closest is a **process-industry** result proving existence conditions for a minimal-sufficient
autoencoder latent for optimal fault detection (<https://arxiv.org/abs/2208.01291>). Nothing does this for a
tactile/contact-count statistic in manipulation. The neural-sufficiency methods
(<https://arxiv.org/abs/1905.07822>, <https://arxiv.org/abs/2508.03158>) do not instantiate to a robotics
monitoring task, and <https://arxiv.org/abs/2510.06492> *asks* the sufficiency question but proves nothing.
**Novel contribution available: prove `Y ⟂ o | c` (or its failure) for a concrete contact statistic by
exhibiting two observations with equal `c` and different `P(Y|o)` — the parent's Proposition 3 — and state the
consequence as an oracle-AUC ceiling, `sup_f AUC(f) = AUC(P(Y|c))`.**

**N3. "Learned monitor == sufficient statistic" is NOT a named phenomenon.**
No standard name. Search for the phrase returns nothing
(<https://arxiv.org/search/?searchtype=all&query=%22learned+monitor%22+%22sufficient+statistic%22> → no results).
The nearest established names/concepts are: **minimal (achievable) sufficient statistic**
(<https://arxiv.org/abs/1905.07822>), **lossless information compression** for fault detection
(<https://arxiv.org/abs/2208.01291>), **representation sufficiency vs over-expressivity**
(<https://arxiv.org/abs/2306.07284>), **predictive sufficiency** (<https://arxiv.org/abs/2508.03158>),
**`V`-usable information** (<https://arxiv.org/abs/2002.10689>), and **Blackwell dominance**
(<https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full>).
A defensible new coinage for our paper: **"monitor sufficiency"** or **"the sufficiency ceiling"** —
"if a hand statistic `c` is sufficient for the failure label, then no monitor can exceed the oracle AUC of
`P(Y|c)`, and a low-capacity monitor that matches `c` is behaving as the Bayes-optimal detector."

**N4. No published robotics case where a learned failure predictor provably could not beat a simple statistic.**
The negative/adversarial evidence is all in the general time-series AD literature and is explained
empirically or by evaluation methodology, never by a bound: random guess beats everything under
point-adjust (<https://arxiv.org/abs/2308.13068>); deep AD ≈ linear maps (<https://arxiv.org/abs/2405.02678>);
benchmarks flawed (<https://arxiv.org/abs/2009.13807>); expressivity hurts (<https://arxiv.org/abs/2306.07284>).
In robotics the reported direction is the *opposite* — low-capacity predictors win
(<https://arxiv.org/abs/2609.21246>), and failure-prediction claims are framed as improvements
(<https://arxiv.org/abs/2410.14868>). **This is the gap the parent's measurement fills, and pairing it with
N1/N2 is what makes it a theory paper rather than another negative result.**

---

## 4. The single best existing bound to use as OUR ceiling

**Primary candidate: the `f`-divergence → Neyman–Pearson-region bound of Mullhaupt & Peng
(<https://arxiv.org/abs/2505.08899>).** For any `f`-divergence between the failure/no-failure
class-conditionals, it lower-bounds the achievable `(FPR, FNR)` boundary, hence upper-bounds achievable
`TPR` at every `FPR` and therefore the achievable **AUC of *any* monitor**, learned or not. It is the best
general bound available: hockey-stick f makes it best-possible, KL yields a Pinsker improvement, and the
Chernoff-α form is closed-form. Because for balanced priors `I(X;Y) = JSD(P₊‖P₋)` (an `f`-divergence),
the same bound can be *re-expressed in terms of the sensor's mutual information with the failure label* —
which is exactly the "MI ceiling" the parent asked for, and which as far as this survey can tell **nobody has
written down for a robot sensor**.

**Closed-form companion when Gaussianity holds (very likely for a count/contact statistic or a
sum of per-site indicators under the CLT):** with equal-variance Gaussians
`KL(P₊‖P₋) = d′²/2` and `AUC = Φ(d′/√2)`, so
**`AUC_max = Φ(√(KL(P₊‖P₋)))`** — exact, tight, and directly estimable from a two-sample KL estimate
`UNVERIFIED (own derivation from standard detection theory: AUC = Φ(d′/√2))`.

**Honest caveat — no single scalar pins down the AUC (own analysis).**
Two problems with identical total variation can have different optimal AUC. Example: (A) `P₊=U[0,2]`,
`P₋=U[0,1]` gives `TV=1/2` and optimal AUC `0.75`; (B) `P₋=U[0,2]`, `P₊=U[1,3]` also gives `TV=1/2` but
optimal AUC `0.875`. So MI (and TV, and any single `f`-divergence) yields only a **one-sided, generally
loose** ceiling. The *tight* ceiling is the likelihood-ratio AUC, which requires the full class-conditional
pair, not one number. Anyone quoting "MI ⇒ AUC ≤ X" must also report how loose the bound is on their data.

**What would be needed to make it practical for the robotics group:**
1. **State the ceiling for the sensor, then for the monitor class.** Compute the bound once for the raw
   sensor stream (all functions of the observation history) — this is the absolute ceiling — and once with
   `V`-usable information (<https://arxiv.org/abs/2002.10689>) to get the ceiling for a *given model
   family*. The gap between the two is the honest "learning could still help" budget.
2. **Estimate the divergence from finite data with a known rate**, e.g. the variational arc-length estimator
   with rate `O_p(n^{−β/4})` (<https://arxiv.org/abs/2110.09651>), or a Gaussian/KL fit for the scalar
   statistic. Report confidence intervals; the ceiling is only useful with error bars.
3. **Handle imbalance and shift explicitly.** AUC is prior-free, so the `f`-divergence bound can be used at
   the measured failure rate; but under shift the class-conditionals move, so the bound must be recomputed
   per condition, and Zhao et al.'s lower bound (<https://arxiv.org/abs/1901.09453>) tells us that
   invariance-based fixes cannot close the gap when label marginals shift.
4. **Validate the ceiling against the measured `AUC(c)`.** The prediction to test: `AUC(c)` should be
   (near-)equal to `AUC_max` on the tasks where the learned models never win (parent's task 26/32; 0.94 vs
   0.90) and strictly below it on task 73 (0.719), which is exactly the pattern the parent already measured.

---

## 5. What I could NOT verify

* **Paywalled PDFs (could not fetch; `web_fetch` returned no usable text):** Hellman–Raviv 1970,
  Feder–Merhav 1994 + its comment, Chow 1970, the *Pattern Recognition Letters* Bhattacharyya/Chernoff
  bounds paper, the Barrett & Myers *Foundations of Image Science* AUC=`Φ(d′/√2)` passage, and
  Ben-David et al. 2010. These are cited from search snippets and are `UNVERIFIED`. `bash` has **no network**
  in this session, so no PDF could be downloaded by any other route.
* **arXiv API flakiness:** several `export.arxiv.org/api/query` calls returned
  `unsupported content type "unknown"` mid-session (and Semantic Scholar returned HTTP 429). Absence
  evidence for §3 therefore comes from arXiv's **web** search, which matches **title/abstract/comments only —
  not full text**. "No results" means no title/abstract match, not that no full text anywhere says it.
* **Theorem-level details not checked:** I verified abstracts, not proofs. In particular I did not verify
  (i) the exact functional form or tightness conditions of the arXiv:2505.08899 bound, (ii) whether it is
  stated explicitly for AUC, (iii) the large-sample Blackwell-dominance sharpening of Mu et al. 2021
  (only seen in a third-party PDF), (iv) the `Ω(1/π)` rare-event statement (I could not fetch a clean primary
  source; the Bayes-threshold part is elementary and marked as own analysis), and (v) the claim that
  `I(X;Y) = JSD(P₊‖P₋)` for balanced priors (standard, but I did not fetch a primary source for it).
* **The `AUC_max = Φ(√KL)` relation is my derivation**, not a fetched statement; it follows from
  `AUC = Φ(d′/√2)` (snippet-verified) and `KL = d′²/2` for equal-variance Gaussians (standard). Verify before
  publishing.
* **No robotics paper with a *quantified* simple-statistic-beats-learned result was found.** The parent's own
  task-26/32/73 measurements may be the first of their kind; I could not find a prior published instance, and
  the general AD literature's negative findings are explained empirically, not theoretically. That is a
  point in favour of the paper but also means there is no prior art to lean on for the empirical claim.
