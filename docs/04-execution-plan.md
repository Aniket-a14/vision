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

**0.5 is the decision point for the whole project.** If extrapolated extraction exceeds ~45 min/regime × 4 regimes, fall back to ResNet-18 and note it. Do not discover this in Week 2.

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

- [ ] No single feature correlates with the label above **|0.35|**
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
| 3 | **Fusion + the full 3×2 ablation.** Degradation-severity sweep (0.0 → 2.0) for all three models — *this single chart is your best figure*. |
| 4 | **Calibration** (isotonic + reliability diagrams) → **Mondrian conformal prediction** (MAPIE) → abstention state. **Cost-optimal threshold** via `TunedThresholdClassifierCV`. |
| 5 | **Grouped SHAP** (image block → one scalar), **ALE plots** (PDP is banned), Anchors rules. **Attribute MSA / Cohen's kappa.** MLflow logging everything. Drift baselines: Frouros on tabular, Evidently on embeddings. |

### Gate 2

- [ ] **Process-only ROC-AUC in 0.80–0.88.** Above 0.88 → increase `NOISE_SD` and regenerate. Real foundries cannot predict this well from process data alone; 0.97 means your noise term is too small and the result is not credible.
- [ ] **Fusion beats vision-only under inline imaging** by a margin you can state and defend
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
