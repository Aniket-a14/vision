# Decision Layer — Explainability, Prescription, Thresholds, Economics

**Status:** derived from deep research, 2026-07-31. Supersedes §§8–9 of `multimodal_defect_prediction_build_guide.md`.

The original guide's decision layer is: SHAP on XGBoost → brute-force grid counterfactual → validate against the simulator → fixed-cost COPQ. Three structural problems, each with a cheap fix that is also a strong report section.

---

## D1 — SHAP answers the wrong question

SHAP attributes to the **model's conditional expectation**. The prescriptive question is `E[Y | do(X = x')]`. These coincide only under conditions that do not hold in high-pressure die casting, where the process variables are heavily coupled (melt temp ↔ die temp ↔ cycle time ↔ fill time).

**Three concrete failure modes to design around:**

| Failure mode | What happens here |
|---|---|
| **Correlated-feature credit splitting** | Melt temp and die temp trade attribution between retrains. Operators see the "root cause" change week to week and stop trusting it. |
| **Off-manifold evaluation** | Interventional TreeSHAP evaluates the model at physically impossible combinations (700 °C melt + 300 °C die + 40 ms fill). Attributions become extrapolation artefacts. |
| **Confounded proxies ranked as causes** | Cycle time will show large SHAP for porosity — but it is *downstream* of die thermal state, not a lever. Prescribing a cycle-time change is wrong. (The canonical analogue: vasopressors rank top for mortality by SHAP; they mark the sickest patients, they do not cause death.) |

**Rules for the codebase:**
- If TreeSHAP is retained, use `feature_perturbation="interventional"` with a cached in-distribution background set of 100–1000 rows. Never present a SHAP value for a non-manipulable variable as a "cause".
- **Re-label the output.** "Model evidence", not "root cause". This one word change is honest and costs nothing.
- Add `monotone_constraints` to XGBoost wherever the metallurgy is unambiguous. This alone kills most nonsense explanations.

### The three-layer explanation stack

| Layer | Purpose | Library | Priority |
|---|---|---|---|
| **Glass-box shape functions** | "How does risk vary with intensification pressure, all else equal?" | **InterpretML `ExplainableBoostingClassifier`** (EBM — a GA²M) | **Essential.** The shape function *is* the explanation — no post-hoc approximation. Accuracy comparable to XGBoost on tabular data, restricted to pairwise interactions (vs. order-8 interactions in deep XGBoost trees, which are uninterpretable). Also better calibrated out of the box, which matters for §D3. |
| **Global effect check** | Sanity-check monotonicity against metallurgy | **ALE plots** (`alibi` / `PyALE` / `interpret`) | **Essential, cheap.** ALE is unbiased under feature correlation; it accumulates local differences over the conditional distribution. **PDP marginalises over physically impossible combinations and will lie here — ban PDP from this project.** ALE is also O(n) and faster. |
| **Local per-shot rule** | "Flagged because slow-shot < 0.24 m/s AND die temp < 190 °C" | **Anchors** (`alibi`) | **High value.** IF–THEN rules with stated precision/coverage are what an operator will actually act on. |
| **Global importance done right** | "Which sensors matter?" | **SAGE** (`iancovert/sage`) | Nice-to-have. SAGE distributes model *loss* across features and sums to the loss — the correct object. Mean-\|SHAP\| is the common but wrong substitute. |

**Overkill:** NODE-GAM, GAMformer, Causal-SHAP hybrids.

### What the operator HMI shows

1. Risk + **calibrated** probability + prediction interval.
2. An **Anchor rule** in process language, with its precision ("holds for 93 % of similar shots").
3. One or two prescribed setpoint deltas: magnitude, direction, expected risk reduction, **and cost of the change**.
4. The **shape function** for the variable being changed, with the current setpoint marked — so the operator sees the local slope and the safe window.
5. An **abstention state** when out-of-distribution. "I don't know — call the process engineer" is a trust feature, not a failure.

**Do not show a 20-bar SHAP waterfall.** It is the most common and least useful production XAI artefact.

---

## D2 — Replace the grid search with constrained optimisation

The grid search cannot express the constraints that make a recommendation executable.

### Problem formulation

```
minimise over x':
    p_defect(x') · C_scrap                    quality cost
  + Σ_j c_j · |x'_j − x_j|                    cost of change
  + λ · energy_or_cycle(x')                   throughput

subject to:
  x'_j ∈ [lo_j, hi_j]           machine physical limits
  |x'_j − x_j| ≤ Δ_j            ramp-rate / trust region (thermal inertia is real)
  x'_j = x_j  ∀ j ∈ immutable   alloy chemistry, geometry, die ID
  coupling constraints          gate velocity is a function of fast-shot velocity and gate area
  ‖x' − x‖₀ ≤ k,  k = 1..2      sparsity — operators will not execute a 6-knob change
```

### Cost of change — the part everyone forgets

| Parameter | Cost of change | Constraint |
|---|---|---|
| Intensification pressure | free, instant | box only |
| Fast-shot velocity / switch point | free, instant | box + coupling |
| Melt/holding temperature | slow (30–90 min), energy, **increases H pickup** | ramp-rate + asymmetric cost |
| Die temperature (thermolator/spray) | 10–30 min to stabilise | ramp-rate, strongly asymmetric |
| Alloy chemistry | lot-level, days | **immutable within a shift** |
| Die geometry / venting | tooling change, weeks | **excluded from the search space** |

### Solvers

| Tier | Approach | Library | Verdict |
|---|---|---|---|
| **Ship this** | **Exact MILP over the trained tree ensemble** — encode leaf/split structure as binaries, add all constraints above, solve to proven optimality in milliseconds | **`gurobi-machinelearning`** (supports `xgboost.Booster`, `XGBRegressor`, LightGBM, sklearn trees/NNs — ⚠️ `gbtree` boosters only, `reg:squarederror` / `reg:logistic` objectives only; commercial licence) | Replaces the grid entirely. Exact instead of exponential. |
| **Open-source equivalent** | Same formulation via Pyomo | **`OMLT`** + HiGHS/CBC/SCIP | Use if no Gurobi licence. Slower, free. **This is the right choice for this project.** |
| Uncertainty-aware search | LightGBM surrogate + distance-based uncertainty, deterministic global solve | **`ENTMOOT`** | Well-matched, under-used. Worth a look if time allows. |
| Physical DOE campaign | Constrained Bayesian optimisation | **Optuna ≥ 4.5** (`GPSampler` gained inequality constraints in 4.2, multi-objective in 4.4, **constrained multi-objective in 4.5**) | For the DOE stage, not per-shot recommendations. |
| UI-layer alternatives | Diverse counterfactuals | **DiCE** (`permitted_range`, `features_to_vary`, per-feature weights) | Use as a generator of *alternative* options ("3 different ways out of the risk zone"), **not** as the primary optimiser. Note DiCE-Extended (arXiv 2504.19027) addresses vanilla DiCE's instability under perturbation. |
| **Reject** | OR-Tools / CP-SAT | | Mismatched — CP-SAT excels at combinatorial scheduling; this is continuous setpoints + embedded tree ensemble. MILP is the natural fit. |
| **Reject** | `scipy.optimize` DE/SLSQP | | Fine as a 20-line baseline; beaten by MILP because the objective is piecewise-constant and gradient-free, and DE gives no optimality certificate. |

### Fit the surrogate on *interventional* data

**The strongest single recommendation in this document.** You have a simulator — that is an enormous advantage most causal practitioners lack, because **you can run true interventions.**

1. Generate a **randomised interventional dataset** — randomise setpoints within physically legal envelopes (a virtual DOE).
2. Fit the prescriptive surrogate on **that**, not on observational logs. A model fit on randomised data has **no confounding by construction**, and its partial derivatives *are* causal effects within the simulator.
3. Use the observational stream only to validate marginals and detect drift.
4. Reserve DoWhy/EconML for the reconciliation question: "does the measured effect match the simulator?"

This sidesteps the entire observational-causal-inference problem, is cheap, and is a genuinely sophisticated methodological move for a capstone. It also directly defuses the most dangerous failure mode: in observational plant data, operators already raise intensification pressure *when they see porosity*, which induces a **reversed sign** — a SHAP-driven recommender trained on such data will tell operators to *lower* pressure to reduce porosity.

**If observational causal inference is wanted anyway** (a strong optional report section): hand-build a DAG with domain input first, use `causal-learn` discovery only to *challenge* edges that disagree with it, then `DoWhy` for identification + **refuters** (placebo treatment, random common cause, data-subset, add-unobserved-confounder), and `EconML` for continuous-treatment dose-response CATE.

---

## D3 — Thresholds and cost-sensitivity

### Calibrate first, then threshold

Cost-optimal thresholds computed on **uncalibrated** XGBoost probabilities are meaningless. Order of operations:

1. `CalibratedClassifierCV` (isotonic or sigmoid) + reliability diagrams.
2. **`sklearn.model_selection.TunedThresholdClassifierCV`** (sklearn ≥ 1.5) with a custom `scoring` implementing the cost-gain matrix. This one in-tree class is the whole of what's needed.

The classic optimum:

```
t* = (C_FP − C_TN) / [(C_FP − C_TN) + (C_FN − C_TP)]
```

For a pressure-tight impeller, `C_FN ≫ C_FP`, so **t\* will plausibly land at 0.02–0.10**, far below 0.5. The original guide is right that 0.5 is wrong; it under-states how far wrong.

### Class imbalance — a warning

**Prefer `scale_pos_weight` / `class_weight` + threshold tuning over SMOTE.** Resampling distorts exactly the probability calibration the expected-value calculation depends on. This is the most common mistake in defect-prediction papers, and doing it correctly is a differentiator.

**Do not use `costcla`** — effectively unmaintained, no recent releases. Reimplement its example-dependent-cost ideas in ~40 lines over sklearn rather than taking the dependency.

### Neyman–Pearson as a secondary constraint

The NP paradigm minimises Type II error subject to a **user-specified upper bound on Type I error with finite-sample high-probability control**. This is the right frame when the spec says *"escape rate ≤ X PPM"* rather than *"minimise cost"*.

```
t = min( t*_EV , t_NP(α) )
```

Cost-optimal **and** guaranteed to meet the contractual escape rate. No maintained Python package exists (only R's `HNPclassifier`) — the umbrella algorithm is ~30 lines: order scores on a left-out class-0 sample, take the order statistic giving the desired violation-rate bound.

### Uncertainty in the cost parameters themselves

Nobody knows `C_FN` to better than a factor of 3. Three options:

1. **Threshold-sensitivity band** — essential, near-free. Sweep `C_FN/C_FP` over its plausible range; plot `t*` and realised cost. If `t*` is flat over 5×–50×, the argument is settled. **Decision curve analysis** (`dcurves`) is the packaged version and is excellent for stakeholder communication.
2. **Bayesian decision theory** — recommended. Put a prior on the cost ratio, minimise posterior expected loss integrating over both model and cost uncertainty. Output is management-legible: *"threshold 0.04, robust across our cost beliefs."*
3. **Distributionally robust optimisation** — overkill unless a regulator demands worst-case guarantees.

### Conformal prediction — the abstention state

Distribution-free prediction sets under exchangeability, via **`MAPIE`** or **`crepes`**. This is how the "abstain / call the engineer" state in §D1 gets a **coverage guarantee** instead of a heuristic threshold. There is direct 2025–26 precedent for conformal deferral policies that route the most-uncertain observations to a human.

**This is the highest-value ML-rigour addition to the whole project** and it is roughly a day's work.

---

## D4 — Economics, corrected

### Frameworks to name

- **PAF model** (Prevention / Appraisal / Internal Failure / External Failure) — the primary accounting structure. Two are investments, two are losses.
- **Taguchi quadratic loss** `L(y) = k(y − T)²` — **the most under-used idea available here.** Porosity severity is continuous; pass/fail costing systematically under-prices marginal parts. This is the correct bridge from "porosity 2.1 %" to "£". Prioritise it.
- **ASQ COQ benchmarks** — COPQ ≈ **10–20 % of revenue** typical, world-class **< 5 %**. Verified across multiple sources.
- ⚠️ **The 1-10-100 rule** — use as a *directional heuristic only*. **No primary ASQ publication stating this ratio could be located**; every source is secondary, and a 2016 study of 171 agile projects found modern practice flattens the curve. **Do not present it as a cited standard.**

### Cost model

```
C_scrap_internal = metal_cost × (1 − remelt_recovery)     # ~5–8 % melt loss on remelt
                 + energy_remelt
                 + machine_time × (cycle_time / OEE)      # opportunity cost — usually the LARGEST term
                 + direct_labour + handling
                 + appraisal_cost_already_sunk            # X-ray, leak test

C_escape = C_scrap_internal × M
```

Justify **M** from a built-up chain — customer sort/containment + line-stop charges + freight + PPAP re-submission + warranty accrual + probability-weighted recall — **not** from the 1-10-100 slogan. For a pressure-tight rotating part where porosity means a leak path or fatigue initiation, **M in the 10–50× range is defensible; > 100× only with an evidenced recall pathway.** State it as a range and run the sensitivity sweep.

### Figures that are safe to cite

| Quantity | Figure | Source quality |
|---|---|---|
| Porosity share of Al casting rejections | **~50 % of scrapped parts** | Widely repeated, secondary |
| Typical HPDC part porosity | ~0.5 vol% | Industry |
| Injection speed effect | 1.0 → 1.5 m/s raised porosity **7.49 % → 9.57 %** | **Peer-reviewed (PMC11509743) — good** |
| Foundry internal failure | 2 % scrapped at full cost + 3 % reworked at half = **3.5 % of production cost**; real COPQ typically **3–5× visible scrap cost**; steel-foundry remelt ≈ **7 % of casting cost** | **AFS / *Modern Casting* (2021) — best foundry-specific source** |
| Automotive warranty | **~$58 bn globally in 2024, ~2.2 % of industry sales**, nearly doubled since 2012; some OEMs > 4 % of revenue | **McKinsey (2025) — strong** |
| Al clean die-cast scrap price | ~**$0.38/lb** (Dec-2025) | ScrapMonster — good for the remelt credit |

⚠️ **No published die-casting-specific scrap-cost-per-part or escape multiplier exists** — these are proprietary. **Do not invent one.** Build from a stated machine-hour rate and declare it as an assumption.

The original guide's **prior-shift correction** to a realistic 2–4 % base rate remains correct and important. Keep it.

---

## D5 — SPC integration: two lanes, one arbiter

Classical SPC and ML risk scores answer different questions and **must not be merged into one alarm**.

| Layer | Method | Role |
|---|---|---|
| **L1** | Shewhart X̄-R / I-MR on the "Big Five" (metal temp, die temp, gate velocity, intensification pressure, cycle time) | System of record. Keep as-is. |
| **L2** | **EWMA (λ ≈ 0.1–0.2)** or CUSUM on the same | Catches small sustained drifts — die wear, thermolator degradation. |
| **L3** | **Hotelling T² + MEWMA** on the correlated parameter vector | The Big Five are strongly coupled; univariate charts cannot see interaction-driven shifts. Pair T² with **MYT decomposition or a PCA/SPE split** — an undiagnosable T² alarm is worse than no alarm. |
| **L4** | **EWMA chart on the ML risk score itself** | The key integration trick. Risk-adjusted EWMA is well-supported in the literature and gives better shift detection at *lower* false-alarm rates. **Chart the score, don't chart the alarms.** |
| **L5** | Cpk / Ppk | Capability, not detection. ≥ 1.33 general, ≥ 1.67 special characteristics. A Cpk/Ppk gap is the sub-grouping/stability diagnostic. |

### Alarm fatigue — borrow ISA-18.2's numbers wholesale

- **6–12 annunciated alarms/hour** per operating position is the standard's target; compliant systems average **< 6/hour**.
- **1–2 alarms per 10 minutes**; **< 1 %** of 10-minute periods should contain > 10 alarms.
- Operators saturate at roughly **one alarm per minute**.
- Rationalisation typically eliminates **30–60 %** of configured alarms.

**Design rules:**
1. **Set the alert budget first, then derive the threshold.** 120 shots/hour ÷ 6 alarms/hour ⇒ a 5 % alert-rate ceiling. Solve for the threshold that hits it, then check compatibility with the cost-optimal threshold. If they conflict, the answer is automation or batching — **not a louder alarm.**
2. **Never alarm on a single shot.** Alarm on the L4 EWMA crossing a limit, or on `k`-of-`n` consecutive high-risk shots.
3. Suppress redundant alarms — if T² and the ML score share a root cause, raise **one** alarm with both as evidence.
4. **Do not implement the full Nelson/Western Electric eight-rule set on the ML score.** All eight running together give a combined false-alarm rate several times the nominal 1/370 of a single 3σ rule — this is the classic alarm-fatigue generator. Use rule 1 (beyond 3σ) plus rule 2 (9-in-a-row one side) and stop.
5. Track **alarms/hour, top-10 bad actors, and operator acknowledge-and-act rate** as first-class metrics. If operators stop acting, the system has failed regardless of AUC.

---

## D6 — Validation ladder

Stage 0 is what the original guide does. It is necessary and **wildly insufficient**.

| Stage | Tests | Feasible in 4 weeks? |
|---|---|---|
| **0. Simulator inversion** | Can the optimiser invert the surrogate? | ✅ already planned — **label it as circular** |
| **1. Simulator-perturbation robustness** | Perturb coefficients ±20–50 %, add unmodelled noise, shift out-of-distribution; recommendation **sign** must stay stable | ✅ **cheapest high-value test available — do this** |
| **2. Independent digital twin** | Agreement with a *different* physics model (MAGMASOFT / FLOW-3D CAST / ProCAST) | ❌ out of scope — cite as future work |
| **3. Shadow mode** | Log recommendations, execute none; compare to engineer's independent choice | ⚠️ simulate it — the twin makes this genuinely demonstrable |
| **4. Off-policy evaluation** | Doubly-robust / SNIPS estimators (`obp`) | ⚠️ **requires logged propensities.** Since the twin generates the data, **log them from day one** — it cannot be retrofitted, and it makes stage 4 free |
| **5. Physical DOE** | Taguchi orthogonal array screening → RSM confirmation runs | ⚠️ virtual DOE only — still a real report section |
| **6. Switchback line trial** | Randomise policy-on/off in time blocks, not per part | ⚠️ simulate — see below |

**Why switchback and not A/B:** consecutive shots share the die thermal state, so there is severe **temporal interference between units** — randomising per part is invalid. Randomise at the **shift or 2-hour block** level, block on die ID and shift crew, and discard a burn-in window at each switch (thermal state persists ~15–30 min). Theory reference: Bojinov & Simchi-Levi, *Design and Analysis of Switchback Experiments*, **Management Science** (2022).

**One control worth more than it costs: the operator-override log.** Every recommendation, every accept/reject, every reason code. It is simultaneously the OPE propensity source, the drift detector, the trust metric, and the audit trail. It costs almost nothing and it is the artefact most regretted when absent.

---

## D7 — Compliance framing (report section, ~free to write)

- **EU AI Act**: in-process quality inspection is **not** Annex III high-risk (none of the eight categories covers industrial QC). It reaches Annex I only if the AI is a *safety component* under harmonisation legislation such as the **Machinery Regulation (EU) 2023/1230** (applies Jan 2027). A porosity advisor recommending setpoints to a human is very likely out of scope; the same model auto-adjusting an injection profile on a safety-relevant machine is a different question. **⚠️ Hidden trap: Annex III category 4 covers employment and worker management.** If dashboards rank *individual operators* by scrap rate, the system may pull itself into high-risk through the back door — **aggregate override metrics at shift/cell level, never per operator.**
- **Article 4 AI-literacy obligations already apply** regardless of risk tier.
- **ISO/IEC 42001** — Clause 7.4 (transparency, documenting models and decision processes) and Clause 6.1.3 (AI system impact assessment) are the practical hooks. Adopt the clause structure as the documentation skeleton; do not chase certification.
- **⚠️ Largest identified compliance gap: no IATF 16949 / AIAG guidance on ML-based quality decisions could be found.** The right framing is to treat the model as a **measurement system** and run an **attribute agreement analysis / MSA analogue** — Cohen's kappa against a reference standard. AIAG MSA-4 puts **κ > 0.75** at "good to excellent"; some organisations demand **> 0.95**. **Target κ ≥ 0.90.** This is cheap to compute, unmistakably industrial, and almost no student project does it.
- **Staged autonomy** with the correct asymmetry: shadow → advisory → **autonomous-reject-only** → full. **Let the AI reject before you let it accept** — a false reject costs one casting, a false accept escapes to a customer. Specify **escape rate (false accept)** and **overkill rate (false reject)** separately; a single "accuracy" number is not an acceptable spec for a QC gate.
