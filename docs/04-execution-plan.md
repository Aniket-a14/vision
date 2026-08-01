# 4-Week Execution Plan

**Solo · CPU laptop · starts 2026-07-31 · freeze scope end of Week 3.**

Every week has a **gate**. If a gate fails, you do not proceed — you drop to the de-scope ladder in §6. This is the mechanism that stops a 4-week project becoming a 9-week one.

---

## Week 0 — one evening, before Week 1 counts

| # | Task | Done when |
|---|---|---|
| 0.1 | **Install real `uv`** (the `uv` on PATH is a broken Safety shim at `AppData\Local\safety\uv.bat`) | `uv --version` prints a version |
| 0.2 | `uv init`, Python 3.13, install stack; verify `import torch, xgboost, shap, cv2, timm, interpret, mapie` | all imports succeed |
| 0.3 | Download the Kaggle casting dataset; **verify the four folder counts exactly**: train 3758 def / 2875 ok, test 453 def / 262 ok | counts match |
| 0.4 | `git init`, push to GitHub, GitHub Actions running `ruff` + `pytest` on an empty suite | green badge |
| 0.5 | Time DINOv2 ViT-S/14 extraction on **200 images**, extrapolate to 7,348 | you have a real number |

**0.5 is the decision point for the whole project.** Do not discover this in Week 2.

### Measured on this machine (4C/8T i7-1195G7), 2026-08-01

Train (6,633) and test (715) together are the 7,348 images, so a *regime* costs one pass
over the whole dataset and there are only two regimes, not four.

| Backbone | Dim | Per regime (7,348 images) | Both regimes | Basis |
|---|---|---|---|---|
| `resnet18` | 512 | **6 min 41 s** | **12 min 46 s** | measured end to end |
| `dinov2_s` (ViT-S/14 @224) | 384 | **~10 min 30 s** | ~21 min | measured at 11.6 img/s |

The ResNet figure was an extrapolation from a 200-image slice and overstated the cost by
4x; sustained throughput is 17–25 img/s, and the second regime is *faster* than the first
because the images are already in the OS page cache and resolution loss shrinks the frame
before the resize.

The DINOv2 estimate was 5x too pessimistic for the same reason. A 96-image benchmark gives
7.8 img/s, but that still amortises model loading over too few batches; sustained rate is
11.6 img/s. **Benchmark on a slice large enough to reach steady state, or do not quote the
number.** A full ladder — lab plus five severities — is six passes, about an hour.

**Consequences, decided:**
- Headline caches use **`dinov2_s`**; `resnet18` is kept as a cheap fallback and a
  backbone-comparison row in the results table.
- ~~The degradation sweep uses `resnet18` on a fixed 1,500-image subsample.~~ **Superseded:**
  the subsample was sized against the 2.8 h estimate, which was ~4x too pessimistic. Four
  extra severities took **16 min** at full size, so the sweep runs on all 6,633 images and
  no subsampling needs stating in the report.
- Regenerating a cache is never on the critical path — the cache is keyed by
  `(split, backbone, regime, severity)` and reused by every downstream experiment.

---

## Week 1 — Data foundation and the twin

**Theme: the twin is the project's foundation. If it's wrong, everything downstream is confidently wrong.**

| Day | Work |
|---|---|
| 1 | Repo skeleton per `03-architecture-and-stack.md` §6. `docker-compose.yml`: Postgres+TimescaleDB, Mosquitto, Redis. Postgres schema: `parts`, `process_telemetry` (hypertable), `predictions`, `overrides`, `audit_log`. **Import-linter test asserting `twin/` never imports `models/`.** |
| 2–3 | **`twin/physics.py`** — the physics-based layer: H₂ solubility curve (0.69 → 0.036 mL/100 g at solidification, the ~19× cliff), `Fe_crit(Si) = 0.075·Si − 0.05`, sludge factor `SF = Fe + 2Mn + 3Cr < 1.8`, saturating intensification pressure with a knee ≈ 67 MPa, two-sided Fe (soldering below ~0.7 %, β-platelets above Fe_crit), split slow-shot / fast-shot velocities each with a critical-velocity optimum. All per `01-physics-twin-corrections.md`. |
| 3 | **`twin/scm.py`** — defect propensity from physics + irreducible noise. **Causal order is sacred**: `parameters → probability → sampled label → matched image`. Calibrate the intercept to the image-split prevalence. **Log the propensity for every row** (free off-policy evaluation later — it cannot be retrofitted). |
| 4 | **`twin/dynamics.py`** — stateful line: die thermal state integrating shot-to-shot, tool wear accumulating on shots *and* thermal-exposure integral, alloy lots stepping chemistry, sensor drift, shift changeovers. |
| 5 | **`imaging/degrade.py`** (Regime B) + save 6 before/after pairs to `figures/`. **`data/` schemas in `pandera`**; **lot-aware and die-aware splitters**. Freeze the Power BI export contract. EDA notebook. |
| 5 | **Offline demo bundle** script (`docker save` tarball). Build it now. |

### Gate 1 — all four must pass

- [ ] No single feature correlates with the label above **|0.35|**, checked as a mean across seeds — a one-seed reading of this is noise (see the Gate 2 note)
- [ ] Pour-temp histogram split by label shows the **U-shape** (defects at *both* tails), not clean separation
- [ ] `pytest` passes; import-linter confirms `twin/` ↛ `models/`
- [ ] `docker compose up` brings the stack up clean from scratch

---

## Week 2 — Features, models, and ML rigour

**Theme: this is your highest-scoring week. Do not let it slip.**

| Day | Work |
|---|---|
| 1 | Extract features: **4 caches** (train/test × lab/inline) with DINOv2 ViT-S/14. Start it, work on something else while it runs. Cache to `.npy`, never regenerate. |
| 2 | Baselines: vision-only, process-only, both regimes. **PCA ablation {8, 16, 30, 64}** + block scale-normalisation. Three tabular models: XGBoost, EBM, TabPFN-2.5. |
| 3 | **Fusion + the full 3×2 ablation.** Degradation-severity sweep (0.5 → 3.0) for all three models — *this single chart is your best figure*. |
| 4 | **Calibration** (isotonic + reliability diagrams) → **Mondrian conformal prediction** (MAPIE) → abstention state. **Cost-optimal threshold** via `TunedThresholdClassifierCV`. |
| 5 | **Grouped SHAP** (image block → one scalar), **ALE plots** (PDP is banned), Anchors rules. **Attribute MSA / Cohen's kappa.** MLflow logging everything. Drift baselines: Frouros on tabular, Evidently on embeddings. |

### Gate 2

- [ ] **Process-only ROC-AUC in 0.80–0.88, averaged over ≥5 twin seeds.** Above the band → raise `noise_sd` or lower `signal_gain` and regenerate. Real foundries cannot predict this well from process data alone; 0.97 means the noise term is too small and the result is not credible.
- [ ] **Fusion beats vision-only under inline imaging** by a margin you can state and defend, reported as a *paired within-seed* difference

> **Measured 2026-08-01 — read before trusting any single number here.**
> The effective sample size is the number of **alloy lots, not parts**. At
> `shots_per_lot=220` a 715-image test split contains only ~4 lots, and chemistry is drawn
> per lot, so process-only AUC swings **0.61–0.87 across seeds (sd 0.086)**. Seed 42 alone
> reads 0.93 — the highest of five, and out of band. Shortening lots tightens it
> (10 lots → sd 0.037, 20 lots → sd 0.022) but 40 shots per furnace charge is not
> physically defensible for HPDC, so the variance is **reported, not tuned away**.
> `max |corr|` is seed-noisy for the same reason, and its top feature moves between
> `fe_content_pct`, `pour_temp_c` and `tool_wear_shots` depending on the draw.
>
> Consequence: **single-seed gate readings are not evidence.** `defectlab ablate` runs
> across seeds by default and reports mean ± sd. Absolute AUC is noisy; the paired
> fusion-minus-vision difference is far tighter, because the image channel is identical
> across twin seeds and cancels.

### First measured 3×2 result — ResNet-18, 5 seeds

| modality | regime | mean AUC | sd |
|---|---|---|---|
| vision | lab | 0.9964 | 0.0000 |
| vision | inline | 0.9847 | 0.0000 |
| process | either | 0.8372 | 0.0558 |
| fusion | lab | 0.9955 | 0.0017 |
| fusion | inline | 0.9882 | 0.0046 |

Process-only sits **inside** the 0.80–0.88 band on the mean, so `signal_gain=3.0` needs no
re-tuning; only the seed-42 reading (0.9266) was out of band.

Paired fusion − vision, within seed:

| regime | mean | sd | seeds positive | t (n=5) | p |
|---|---|---|---|---|---|
| inline | **+0.0035** | 0.0046 | 4/5 | 1.71 | **0.162** |
| lab | −0.0009 | 0.0017 | 2/5 | −1.13 | 0.32 |

**The fusion advantage is not yet established.** The direction is right and consistent
(4/5 seeds), but the effect is under one standard deviation from zero. The single-seed
+0.0094 previously reported was the largest of the five.

The cause is headroom, not fusion: inline degradation costs vision only **0.0117** AUC
(0.9964 → 0.9847), so there is almost nothing for the process channel to recover. The
mechanism is nonetheless visible — the per-seed gain tracks how informative that seed's
process channel is, at **r = 0.74**. Testing the hypothesis properly needs a degradation
severe enough to actually damage vision, which is what the severity sweep is for.

### Degradation sweep — ResNet-18, 5 seeds x 5 severities

`severity` originally scaled only sensor noise; blur, lighting and resolution were fixed
regardless of it. That is why inline cost vision just 0.0117 AUC. Severity now scales all
four channels, and the sweep above was re-run against the corrected model.

| severity | vision | fusion | gain | seeds positive | p |
|---|---|---|---|---|---|
| 0.5 | 0.9985 | 0.9988 | +0.0003 | 3/5 | 0.431 |
| 1.0 | 0.9847 | 0.9882 | +0.0035 | 4/5 | 0.163 |
| 1.5 | 0.9671 | 0.9654 | **−0.0017** | 1/5 | 0.818 |
| 2.0 | 0.8984 | 0.9264 | +0.0280 | 4/5 | 0.111 |
| 3.0 | 0.8374 | 0.8991 | **+0.0617** | 5/5 | 0.025 |

**The primary test is the trend, not any single severity.** Five per-severity tests invite
cherry-picking, and the 0.025 at severity 3 does not survive a Bonferroni correction for
five comparisons. The pre-specified analysis is the dose-response slope: fit gain against
severity within each seed, then test the slopes against zero.

### Settled at 15 seeds

The five-seed run left the trend at p = 0.028. Image caches are seed-invariant, so widening
to 15 seeds cost model fits only — 13 minutes, no re-extraction.

| severity | vision | gain | sd | seeds + | p |
|---|---|---|---|---|---|
| 0.5 | 0.9985 | +0.0002 | 0.0007 | 9/15 | 0.263 |
| 1.0 | 0.9847 | +0.0051 | 0.0034 | 14/15 | <0.001 |
| 1.5 | 0.9671 | +0.0031 | 0.0111 | 8/15 | 0.291 |
| 2.0 | 0.8984 | +0.0352 | 0.0228 | 14/15 | <0.001 |
| 3.0 | 0.8374 | +0.0727 | 0.0290 | **15/15** | <0.001 |

> **Trend: mean slope +0.0303 per unit severity, t = 9.39, p < 1e-5, 15/15 seeds positive.**

**Fusion's benefit grows as imaging degrades.** At severity 3 the camera costs vision 0.161
AUC and fusion recovers 45 % of it, on every seed tested. The original null was not wrong —
it was measured at the one severity where vision has no headroom to lose.

**The severity 1.5 anomaly is real, not noise.** It survived tripling the seed count: the
gain there (+0.0031) is *lower* than at severity 1.0 (+0.0051) despite worse imaging, and
its spread is three times wider (sd 0.0111 vs 0.0034). Vision itself degrades smoothly
through that point, so the dip is in the fusion model, not the camera. Unexplained. The
neighbouring severities both cross a resolution boundary (96 px at 1.0, 54 px at 1.5, 31 px
at 2.0) and 1.5 is where the effective resolution first falls below the backbone's 64 px
receptive field, but that is a hypothesis, not a finding. **Reported as an open anomaly.**

One limit stays in the writeup: **vision sd is exactly 0.0000** at every severity, because
the images, their order and the label vector are identical across twin seeds. Only the
process channel varies, so all of the paired variance comes from fusion.

The severity ladder was fixed before any result was seen, and the whole curve is reported.

### Backbone replication — DINOv2-S, same seeds and same ladder

| backbone | vision 0.5 → 3.0 | AUC lost | slope | t | p | seeds + |
|---|---|---|---|---|---|---|
| `resnet18` | 0.9985 → 0.8374 | 0.161 | +0.0258 | 3.39 | **0.028** | 5/5 |
| `dinov2_s` | 0.9994 → 0.9157 | 0.084 | +0.0076 | 1.84 | 0.140 | 4/5 |

**The direction replicates; the significance does not.** Reporting only the ResNet row would
be selective — both were run, so both are shown.

The gap is headroom again, not architecture. DINOv2-S is the more robust encoder and loses
only half as much AUC to the same camera, so there is half as much for the process channel
to recover. Plotting the gain against **vision AUC** instead of against severity puts the two
backbones on one relationship:

> pooled gain = 0.348 − 0.351 x vision AUC, **r = −0.73**

Per-backbone slopes are −0.387 (ResNet) and −0.224 (DINOv2), which overlap at this n.

**Significance of that relationship, stated conservatively: t = −2.66, p = 0.056, n = 5.**
The unit of analysis is the twin seed, not the (backbone, seed) pair — both sweeps reuse the
same five seeds, so they share a process channel and a label vector. Treating the ten pairs
as independent gives p = 0.005 and is wrong; the honest number is the marginal one.

So the mechanism is coherent and consistent across two architectures, but with five
lot-limited seeds only the ResNet severity trend clears p < 0.05, and that is one of the two
backbones tested. **Claim the dose-response direction; do not claim it is established.**
Adding seeds is the cheapest way to settle it — the image caches are seed-invariant, so more
seeds cost model fits only, not extraction.
- [ ] Reliability diagram shows calibration is actually improved
- [ ] Mondrian CP achieves its nominal coverage on the defect class *specifically* (this is the whole point — check the class-conditional number, not the marginal one)
- [ ] Every number in the results table is regenerable by one command

**⚠️ Freeze the models at the end of Week 2.** Everything after this consumes them.

---

## Week 3 — The live system

**Theme: make it a running system, not a notebook.**

| Day | Work |
|---|---|
| 1 | `services/linesim/` — the twin publishing shots to MQTT on the **ISA-95 UNS topic tree**, with sequence numbers. `services/ingest/` — MQTT → Postgres, **zero business logic** so it restarts freely. Image refs only, never image payloads. |
| 2 | **FastAPI**: `/predict`, `/parts/{id}`, `/recommend`, `/spc`, **SSE `/stream`** (HTTP/2, heartbeat 15 s, Redis fan-out, `Last-Event-ID` reconnect). ONNX-exported model loaded once at startup. **Hash-chained audit log** with the DB trigger. |
| 3 | **`spc/`** — X-bar/R, I-MR, EWMA, Nelson rules 1–8, frozen Phase I limits as versioned rows. Validate against R `qcc` fixtures. **EWMA on the ML risk score** (layer L4). Apply the ISA-18.2 alert budget. |
| 4 | **`prescribe/`** — constrained recommender. Fit the surrogate on a **randomised interventional dataset** from the twin (no confounding by construction). Actionability constraints: box limits, ramp rates, immutable chemistry/geometry, ≤2 knobs. **Simulator-perturbation robustness test** (±20–50 % coefficient perturbation; report the fraction of recommendations whose direction survives). |
| 5 | **React app**: live line view (SSE), part inspector with Anchor rule + shape function + recommendation + cost of change, interactive sandbox. Mantine + uPlot. Operator override with reason codes. |

### Gate 3 — **HARD SCOPE FREEZE**

- [ ] `docker compose up` → live line streaming into a browser dashboard, end to end
- [ ] Recommendations respect all actionability constraints (a recommendation outside physical limits destroys credibility instantly)
- [ ] Override → audit-log row with model version **and the explanation shown at the time**
- [ ] SPC engine agrees with R `qcc` on fixtures

**From here, no new features. Week 4 is for writing.** This is the original guide's most important scheduling rule and it is correct.

---

## Week 4 — Economics, Power BI, deployment, report

| Day | Work |
|---|---|
| 1 | **`economics/`**: prior-shift correction to a 2–4 % realistic base rate, PAF cost model, **Taguchi quadratic loss** for continuous severity, escape multiplier **M as a stated range (10–50×)**, cost-vs-threshold curve, sensitivity sweep. Report escape rate and overkill rate separately. |
| 2 | **Power BI**: export Parquet/CSV per the Week-1 contract, build the four pages, threshold-slider parameter driving a live COPQ measure. |
| 3 | **Deploy**: Fly.io or Railway free tier. Live line + sandbox both reachable. Rebuild and test the **offline bundle** — your viva demo must work with no network. |
| 4–5 | **Report + slides.** Every figure regenerated from a script. Write the limitations section properly — a candid one signals you understand your work; a defensive one signals the opposite. |

### Measured economics — ResNet-18 fusion, severity 2, corrected to a 3 % line

`defectlab economics --severity 2` re-prices the model on the line rather than on the test set.
Two corrections are applied and both matter: probabilities are shifted from the training
prevalence (0.567) to 0.030 by the Elkan odds rescaling, and the two error rates are then
weighted by that same prior instead of by the test mixture. Skipping either would price a
factory that does not exist.

Cost per 1,000 shots, with scrap €12, inspection €3, M = 25×:

| Policy | Prevention | Appraisal | Internal failure | External failure | Total | €/shot |
|---|---|---|---|---|---|---|
| **Model gate** | 50.00 | 495.79 | 339.34 | 516.56 | **1,401.68** | **1.40** |
| Ship everything | 0 | 0 | 0 | 9,000.00 | 9,000.00 | 9.00 |
| Inspect everything | 0 | 3,000.00 | 360.00 | 0 | 3,360.00 | 3.36 |

Optimal threshold 0.015, escape rate 0.057, overkill rate 0.141.

**The saving that does not depend on M.** Savings against ship-everything are almost entirely a
restatement of the multiplier — they run €2,529 to €16,253 across M = 10–50×, which is a range
too wide to quote. Savings against *100 % manual inspection* move the other way and barely move
at all: €2,289 down to €1,613 over the same range, positive throughout. That is the defensible
headline, because it survives the worst case of the one input that was guessed.

**Alert rate lands inside the human-factors budget unforced.** At the cost optimum the gate flags
16.5 % of shots; on a one-minute cycle that is **9.9 alarms/hour**, inside the ISA-18.2 6–12/hour
band. The alert-budget constraint deferred from `models/pipeline.py` therefore never binds here —
it was disabled because a 57 % prevalence made it starve recall, and once the prior is corrected
the economic optimum satisfies it on its own. Worth stating plainly: this is a result, not a
constraint that was imposed and then reported as a success.

### Taguchi loss — read the ratio, not the euros

Absolute off-target loss comes out at €13.31/shot against a €12 part, which is not credible. The
cause is structural, not a bug: with Δ₀ = 3σ a parameter drawn at its own declared spread has
expected loss A₀/9 ≈ €1.33 *whatever A₀ is*, and there are ten parameters. The absolute figure is
an artefact of the tolerance assumption, so the table reports `baseline_ratio` against A₀/σ².

| Parameter | Baseline ratio | Reading |
|---|---|---|
| `pour_temp_c` | 2.64 | genuine drift away from target added by the line dynamics |
| `intensification_pressure_mpa` | 1.16 | slightly above an independent draw |
| `cooling_time_s`, `hold_time_s`, `slow_shot`, `fast_shot` | 0.98–1.02 | dynamics add nothing; these are the control |
| `fe_content_pct`, `si_content_pct` | 0.60–0.67 | lot-level, so constant within a lot |
| `die_temp_c` | 0.49 | damped below its declared spread by die thermal inertia |
| `mn_content_pct` | 0.43 | lot-level |

Six parameters sitting at exactly 1.0 is the useful part: it means the ratio is measuring the
dynamics and nothing else. `die_temp_c` at 0.49 and the chemistry columns below 0.7 are both
independently predicted by the twin's own physics — a slow thermal state variable and a
lot-level draw respectively — so this doubles as a cross-check on the dynamics.

`tool_wear_shots` is excluded: it has no nominal to sit on, and a quadratic penalty around 25,000
shots would be meaningless.

### Measured prescription — riskiest shot of 600, seed 7

`defectlab prescribe --seed 7`. The surrogate is fitted on 20,000 **interventional** shots
(`sample_uniform_envelope`), never on historian data: the line dynamics correlate pour
temperature with die temperature and both with time-in-shift, so a model fitted on production
data learns associations that do not survive intervention.

```
fast_shot_velocity_ms: 1.662  -> 2.062  (+0.4 m/s)
slow_shot_velocity_ms: 0.1816 -> 0.2416 (+0.06 m/s)
pour_temp_c:           654.2  -> 669.2  (+15 C)
risk 1.0000 -> 0.0174 (+0.9826), margin +16.613 logits
```

Physically right, and worth checking rather than assuming: that shot filled slowly (fast shot
1.66 m/s against a 1.60 limit and a 2.80 nominal) with cold metal (654 °C). That is the
cold-shut and misrun path. All three moves go back toward nominal, and each is capped exactly
at its ramp limit — the search cannot propose more than one shot of travel.

**Robustness — and its limit.** Perturbing every mechanism weight multiplicatively by ±20/35/50 %
leaves stability at **1.000 at all three scales**, median margin gain 16.07–16.10 logits, and the
same holds at the median, p75, p90 and p99 shots. That is a strong result, but it must be
reported with the reason it is strong, or it is worth nothing:

- It is **not vacuous**. Every recommendation trades mechanisms off — the worst shot improves
  three indices while worsening `gas_porosity` by 0.136, the p99 shot worsens `shrinkage`, the
  median shot worsens `flash`. Had each move improved every index it touched, any positive
  reweighting would have agreed *arithmetically* and the test would have proved nothing.
- It is strong because the improvement **dominates the worsening by roughly two orders of
  magnitude**, and ±50 % reweighting is at most a 3× swing in the ratio. It cannot flip that.
- So the honest claim is: *for shots this far from nominal, the advice does not depend on the
  mechanism weights being right*. It would be a genuinely hard test on a shot where the
  improved and worsened mechanisms were comparable in size, and no such shot appears in this
  run.
- Scope: perturbing weights asks "what if shrinkage matters more than we thought", not "what if
  the physics is wrong". The indices themselves are untouched. This is narrower than full model
  misspecification and is not a substitute for real process data.

Advice is suppressed below the 3 % line base rate. Before that gate the median shot — risk
0.0001 — was still collecting three setpoint changes worth 4.8 logits it had no use for.

### Gate 4

- [ ] Every figure in the report regenerable by one command
- [ ] Public demo URL live
- [ ] Offline bundle verified on a machine with networking disabled
- [ ] `.pbix` opens and refreshes against the exports

---

## 5. What makes this "production-grade" — the checklist an examiner will recognise

| Signal | Evidence in this build |
|---|---|
| Reproducible | uv lockfile, Docker Compose, seeded RNG, one-command figure regeneration |
| Tested | pytest + testcontainers + hypothesis (SPC) + import-linter + behavioural ML tests |
| CI/CD | GitHub Actions: lint, test, build, offline-bundle artefact |
| Observable | structured logs, MLflow lineage, drift monitors, SPC on the risk score |
| Uncertainty-aware | Mondrian conformal prediction with an abstention state |
| Auditable | hash-chained append-only log, model version + explanation stored per decision |
| Correctly-specified decisions | calibration → cost-optimal threshold → alert budget; escape/overkill reported separately |
| Domain-credible | ISA-95 topic tree, ISA-18.2 alert budget, Cohen's kappa MSA, Nelson rules, staged autonomy, EU AI Act positioning |
| Honest | declared simulation, stated claim boundary, perturbation robustness instead of circular validation |

### Behavioural ML tests (cheap, and the kind nobody writes)

- **Directional**: raising pour temperature past the known-bad threshold must not *decrease* predicted risk
- **Metamorphic**: two parts identical but for shot number get near-identical risk; the same part scored twice is bit-identical
- **Invariance**: shift and time-of-day must not move the prediction (also your fairness check)
- **Slice**: minimum performance per die and per alloy lot — a model that's great overall and useless on Die 7 loses operator trust permanently on Die 7
- **Recommendation validity**: every prescribed setpoint is inside physically achievable limits

---

## 6. De-scope ladder — cut in this order

Cut from the top when a gate slips. **Never cut from the bottom to save the top.**

1. LoRA backbone fine-tuning *(stretch only — never planned)*
2. TabPFN-2.5 *(keep XGBoost + EBM; the comparison survives with two models)*
3. OMLT/MILP prescriptive optimiser → fall back to constrained grid search *(document the tradeoff — it's still a report section)*
4. Anchors rules *(keep grouped SHAP + ALE)*
5. Keycloak auth → single hardcoded role + a stubbed auth seam *(document it as deliberately stubbed)*
6. Drift detection UI → drift computed in a notebook and reported as a figure
7. Public cloud deploy → local Docker Compose + a recorded demo video
8. **CUT LINE — everything below is load-bearing, do not cut**
9. SPC engine
10. Mondrian conformal prediction
11. Physics twin corrections
12. The 3×2 ablation and the degradation sweep
13. Power BI `.pbix` *(hard rubric requirement)*
14. Report

**The physics-based solidification layer is the highest-risk item in the plan.** If Week 1 day 3 is going badly, degrade it to the literature-calibrated causal SCM (still with all six corrections from `01-physics-twin-corrections.md`) and spend the saved time on Week 2 rigour. The architecture keeps these separable precisely so this swap costs nothing downstream. **Nothing else in the plan depends on the solidification equations being present.**

---

## 7. The five things most likely to sink this

1. **Label leakage into simulated parameters.** Project-ending. The import-linter test and the 0.35 correlation gate exist to catch it.
2. **Random train/test splits instead of lot-aware and die-aware splits.** Silent, subtle, inflates everything.
3. **Building the Power BI dashboard before models freeze.** You will build it twice.
4. **Losing Week 1 to the solidification physics.** Timebox it to 2 days and use the ladder.
5. **Reporting only lab-imaging results.** The inline regime is the entire argument — without it, fusion looks pointless and the thesis collapses.
