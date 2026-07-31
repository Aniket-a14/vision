# Architecture & Tech Stack Decisions

**Date:** 2026-07-31. **Constraints:** solo, ~4 weeks, CPU-only laptop (4C/8T, 15.7 GB), academic capstone engineered to production standards, Docker Compose + free public demo, Power BI mandatory deliverable.

---

## 0. The scoping reality, stated plainly

Independent research estimated a genuine industrial MVP of this system at **~51 person-weeks for a team of 3**. You have **4 person-weeks**. That is ~8 % of the estimate.

**You cannot build that system. You can build something better for your actual purpose:** a system that is *production-shaped* — correct architecture, real seams, real tests, real deployment — at demonstrator scale, with an explicit, documented statement of what is stubbed and why. Examiners reward a defensible boundary far more than they reward a half-finished sprawl.

Two rules follow, and everything below obeys them:

1. **Every component is either fully built or deliberately and visibly stubbed.** No half-built anything.
2. **Depth over breadth.** Three things done to genuine production standard beat ten things done partly. Your chosen investment axes — ML rigour, engineering craft, live system behaviour — pick the three.

---

## 1. The critique that matters most — and the fix

Research surfaced one finding that outranks every tech-stack choice:

> **The fusion premise, as originally specified, joins data from different physical objects.** The process parameters simulate Al-Si high-pressure die casting; the images are real submersible-pump impellers from a Kaggle dataset. No physical part possesses both. Therefore *any* cross-modal SHAP interaction between a tabular feature and an image component is an artefact of your pairing rule, not physics.

This is the first thing an examiner will find, and it cannot be fixed by better modelling.

**It can be fixed by reframing, and the reframing is stronger than the original claim.** The original guide already anticipated half of this with its declared-simulation paragraph. Go further and make the honest version the thesis:

> This system is a **validated simulation testbed for multi-modal fusion under realistic imaging degradation**. The vision branch operates on genuine industrial imagery; the process branch operates on physically-grounded simulated telemetry; **the two are coupled only through the shared defect label, and no cross-modal interaction is claimed to be physical.** The research question is therefore not "what causes defects" but "**when imaging quality degrades, how much predictive performance does an uncorrelated process-parameter channel recover?**" — a question the design answers validly, because the *independence* of the two channels is precisely what the coupling scheme guarantees.

That reframing is airtight. It converts the design's biggest weakness into the thing that makes the experiment clean. Write it in the abstract, not buried in limitations.

**Consequences to enforce in code:**
- Report modality ablations always (vision-only / process-only / fusion × lab / inline).
- **Never report a tabular×image SHAP interaction as a finding.** Aggregate the image block into one "vision contribution" scalar.
- State the claim boundary in the abstract, methodology, and conclusion.

### The saturation problem — already solved by the existing design

Research confirms the Kaggle casting set is **saturated**: published results include VGG-19 at 100.0 % accuracy and HiDraNet at 99.8 %. A saturated benchmark leaves fusion nothing to contribute.

**The original guide's Regime B (inline degradation) already fixes this** and is the single best idea in it. Degrading to 96×96 effective resolution with motion blur, lighting drift and sensor noise drops vision-only into the 0.86–0.91 band and creates the headroom the fusion argument needs. Keep it, and treat **Regime B as the primary result, Regime A as the reference baseline** — not the other way round.

This is now doubly justified: **MVTec AD 2** (2025) showed SOTA anomaly-detection methods fall from > 90 % to **58.7 % AU-PRO** under multi-lighting real-world capture. Your degradation study is a small-scale replication of a documented, published effect. Cite it.

---

## 2. Model architecture — the upgrade

The originally-specified pipeline (`frozen ResNet-18 → PCA-64 → concat with 9 tabular → XGBoost`) turns out to be, almost exactly, the **published control condition** in a 2026 benchmark (MulTaBench), where it is beaten by 4.0–5.8 points. Four changes, all cheap:

| Change | From | To | Cost | Why |
|---|---|---|---|---|
| **Backbone** | ResNet-18 (ImageNet) | **DINOv2 ViT-S/14** | ~1 line via `timm`; ~2–4× extraction time (≈ 20–40 min/regime on your 4 cores, one-off) | Current SOTA frozen dense features; the encoder MulTaBench itself used. ⚠️ DINOv3 is stronger but its weights are access-gated — request them, fall back to DINOv2. |
| **PCA width** | fixed 64 | **ablate {8, 16, 30, 64}, expect 16–30** | free | 64 image dims vs 9 tabular is a **7:1 token imbalance**, which is documented to suppress the tabular signal — i.e. it suppresses exactly the modality carrying your contribution. The ablation is itself a publishable figure. |
| **Block scaling** | raw concat | scale-normalise the two blocks before concat | free | Same imbalance problem, second lever. |
| **Tabular model** | XGBoost only | **XGBoost + EBM + TabPFN-2.5** | ~half a day | EBM (`interpret`) is a glass-box GA²M whose shape functions *are* the explanation. TabPFN v2 beat CatBoost across 261 datasets and is near drop-in at 9 features. Report all three — a model comparison is a free results table. |

**Keep XGBoost as the headline model** if it wins; the point is that you *compared*, and that you have a glass-box alternative to cross-check the physics against.

**Optional stretch (only if Week 3 is clean):** LoRA-tune the DINOv2 backbone on the inline regime. Research shows target-aware tuning beats frozen embeddings by 4.0–5.8 points with **GBDTs gaining most**. Your MX450 (2 GB VRAM) can just about do ViT-S LoRA at batch 8–16. This is the largest single accuracy lever available — but it is a stretch goal, not a plan item.

---

## 3. ML rigour additions (your #1 chosen investment axis)

Ranked by (impact × feasibility). The first three are the difference between a good project and a distinguished one.

### 3.1 Mondrian conformal prediction — do this first

The single most actionable number in the entire research:

> At an imbalance ratio of 1:345, **marginal conformal prediction achieves only 52.94 % anomaly coverage** at a nominal 90 % level. **Class-conditional (Mondrian) calibration restores 90.59 %.**

Defect detection lives in exactly this regime. This gives you:
- A finite-sample coverage guarantee instead of a bare score.
- A principled **abstention state** — "I don't know, call the engineer" — with a stated coverage level rather than a hand-tuned threshold.
- A production behaviour (deferral to a human) that almost no student project has.

**~50 lines with `MAPIE` or `crepes`. Half a day. Do it in Week 2.**

### 3.2 Calibration before thresholding

`CalibratedClassifierCV` (isotonic) + reliability diagrams, **then** `TunedThresholdClassifierCV` with the cost matrix. Cost-optimal thresholds on uncalibrated XGBoost probabilities are meaningless, and this ordering error is extremely common.

**Do not use SMOTE.** Use `scale_pos_weight` + threshold tuning. Resampling distorts precisely the calibration the expected-value calculation depends on.

### 3.3 Leakage discipline, hardened

Keep the original guide's checks (no feature correlating > 0.35 with the label; U-shape in pour temp) and add:

- **Split by alloy lot and by die, never randomly.** Chemistry is lot-level, tool wear is die-level; random splits leak group structure and inflate everything.
- Fit scaler and PCA on **train only** — the original guide flags this and it remains a top-5 pitfall.
- A CI test asserting the training-time and serving-time feature vectors for the same part ID are **byte-identical**. This one test buys ~90 % of a feature store's value at ~1 % of the cost.

### 3.4 Simulator-perturbation robustness

Covered in `01-physics-twin-corrections.md`. Perturb simulator coefficients ±20–50 %, re-run recommendations, report the fraction whose **direction** survives. This defuses the circularity critique and is a one-day job.

### 3.5 Drift detection

- **Tabular:** `Frouros` (ADWIN / KSWIN / Page-Hinkley) — the largest catalogue (31 methods) and actively maintained.
- **Image embeddings:** `Evidently` embedding-drift, or `Drift-Lens`. **Necessary because lighting drift barely moves pixel statistics but substantially changes model behaviour** — pixel-level tests miss it entirely.
- ⚠️ **Do not build on `deepchecks` (last release 2024-12), `whylogs` (dead since 2025-01), or `NannyML` (stalled post-acquisition).** All three are widely recommended and all three are effectively abandoned. `Evidently` is slowing but is the only healthy option — keep drift checks behind a thin interface.

### 3.6 Attribute MSA — Cohen's kappa

Treat the model as a **measurement system** and run an attribute agreement analysis against the reference labels. AIAG MSA-4 puts κ > 0.75 at "good to excellent"; **target κ ≥ 0.90**. Cheap to compute, unmistakably industrial, and essentially no student project does it. Research found **no published IATF/AIAG guidance on ML-based quality decisions**, so this framing is also the honest way to fill a real gap.

### 3.7 Report escape rate and overkill rate separately

Never a single "accuracy" headline. **Escape rate** (false accept — a bad part ships) and **overkill rate** (false reject — a good part is scrapped) have costs differing by ~60×, and a QC gate specified by one number will not survive scrutiny.

---

## 4. Tech stack — decided

Everything below is $0, verified available for Python 3.13, and runs in Docker Compose on your machine.

### Core

| Layer | Choice | Version | Rationale |
|---|---|---|---|
| Runtime | **Python 3.13** | 3.13.1 ✓ installed | Verified across the whole ML stack. **Not 3.14** — `shap` needs ≥3.12 and full-stack 3.14 wheels are unconfirmed. |
| Packaging | **uv** | 0.12.x | ⚠️ **Not actually installed on your machine** — the `uv` on PATH is a broken Safety shim. Install real uv first. ~10–100× faster than pip; workspaces keep training/serving dependency parity. |
| Config | **pydantic-settings** | 2.14.x | One validation system for config and API schemas. Hydra only if you add training sweeps, and never in the serving path. |
| Lint/format | **ruff** | 0.16.x | |
| Tests | **pytest** + hypothesis + testcontainers | 9.1.x | Hypothesis earns its keep on the SPC engine specifically. |

### ML

| Layer | Choice | Notes |
|---|---|---|
| Vision backbone | **DINOv2 ViT-S/14** via `timm`, frozen | Fallback ResNet-18 if extraction time bites |
| Tabular models | **XGBoost 3.3 + EBM (`interpret`) + TabPFN-2.5** | Three-way comparison is a free results table |
| Uncertainty | **MAPIE** (Mondrian CP) | §3.1 |
| Explainability | `shap` (interventional, grouped) + **ALE** + Anchors | **PDP is banned** — it marginalises over impossible parameter combinations |
| Drift | **Frouros** (tabular) + **Evidently** (embeddings) | |
| Tracking | **MLflow 3.15** | Self-hosts on SQLite/Postgres, `LoggedModel` gives real lineage, $0 on-prem. Rejected W&B/Neptune (SaaS-first), Aim (15-month stale release). |
| Inference | **ONNX Runtime**, optionally **OpenVINO** | ⚠️ You have an **Intel Iris Xe iGPU** — OpenVINO is a genuine production-grade inference story on a "CPU-only" laptop and a nice report angle. Ship ONNX, never a torch checkpoint: drops the serving image from ~4 GB to < 500 MB. |

**Stay FP32.** INT8 quantisation costs 0.5–2.0 pp aggregate accuracy — but aggregate accuracy is the wrong metric under class imbalance, where a 1 pp aggregate drop can be a 10+ pp recall drop on the defect class. You have no latency pressure. Removing this risk is free.

### System

| Layer | Choice | Rationale |
|---|---|---|
| API | **FastAPI 0.141 + Uvicorn** | `app.frontend()` (added 0.138) serves the React SPA from the same container — one fewer process, materially simpler deploy. Rejected Litestar (serialization speed is irrelevant when SHAP dominates latency). |
| DB | **Postgres 18 + TimescaleDB** | One database for telemetry, parts, predictions, audit. Operating one DB instead of three is the biggest reliability win available solo. **Not** Postgres 19 (beta) — Timescale lags major PG releases. |
| Analytics | **DuckDB + Parquet** | Zero servers, embeds in-process, training snapshots and Power BI exports. |
| Broker | **Mosquitto** | Smallest footprint, most-deployed edge broker. |
| Jobs | **Dramatiq + Redis** | Your jobs are CPU-bound (SHAP batches, retraining) — the async-native queues' I/O advantage doesn't apply, and Dramatiq keeps a sync programming model matching your ML code. Rejected Celery (heavy), Taskiq (0.x). |
| Live feed | **SSE, not WebSockets** | Data flows one way (line → browser). SSE gives free browser reconnect with `Last-Event-ID`, no sticky sessions, proxy-friendly. Operator actions are audited POSTs. ⚠️ Serve over HTTP/2 (6-connection cap per origin), disable proxy buffering, heartbeat every 15–20 s. |
| Frontend | **Vite + React 19 + TanStack Router/Query + Mantine + uPlot/ECharts** | **Not Next.js** — no SEO need, and a Node runtime is a process you'd have to operate for zero benefit. Mantine for dense industrial UIs. **uPlot** (~20 KB, canvas) for SPC/high-density traces; ECharts for the rest. |
| Deploy | **Docker Compose** | k3s buys Kubernetes failure modes without HA benefits on one node. Full K8s is indefensible here. |
| Auth | **stub in Week 1, real if time allows** | Keycloak is the right on-prem answer but is genuinely complex to configure. See the de-scope ladder. |

### Rejected, with reasons

Streamlit (full-script rerun model, server-held session state, no reconnect semantics — structurally wrong for an always-on line view), Dash, Kubernetes, KServe, Ray Serve, Triton (all overkill at 1–10 parts/sec), Feast (scale mismatch — solve point-in-time correctness with a feature library + the CI byte-equality test), Iceberg/Delta, ClickHouse/QuestDB/InfluxDB (a second database to operate), Terraform (two containers), Airflow (scheduler + webserver + metadata DB + workers for ~10 jobs), `costcla`, `deepchecks`, `whylogs`, `NannyML`, Git-LFS, SMOTE.

⚠️ **Governance notes worth knowing:** BentoML is now owned by Modular (acquired 2026-02); Astral (uv, ruff) is being acquired by OpenAI (announced 2026-03, closure unverified). Neither changes the recommendation — uv's formats are standards-based (PEP 621, standard wheels), so switching cost is low — but you should know.

---

## 5. Power BI — satisfying the rubric without distorting the system

Power BI is a **hard rubric requirement**, so it ships. But it is genuinely poorly suited to being the live system: its fastest DirectQuery refresh is minutes against your seconds-scale operator view, and per-viewer licensing is $14–24/user/month.

**Resolution — build both, with clear roles:**

| Deliverable | Role |
|---|---|
| **React web app** | The *system*. Live line view, per-part inspector, sandbox, engineer RCA. This is what runs in the demo and what you show in the viva. |
| **`.pbix` file** | The *management reporting layer*, built on `data/exports/*.parquet` and a Postgres DirectQuery connection. Four pages as specified in the original guide (Executive / Process control / Root cause / Part inspector), plus the threshold-slider parameter driving a live COPQ measure. |

Design the export schema **once, in Week 1**, so Power BI is a consumer of a stable contract rather than a rework driver. Build the `.pbix` in Week 4 against frozen models — building a dashboard before models freeze is the original guide's pitfall #10 and it is correct.

**Say this explicitly in the report**: Power BI serves the management reporting layer; the operational layer requires sub-second updates that Power BI's refresh model cannot provide. Justifying a tool boundary is a maturity signal, not a dodge.

---

## 6. Repository layout

```
vision/
├── docs/                      # these design documents; ADRs
├── src/defectlab/
│   ├── config.py              # pydantic-settings
│   ├── twin/                  # THE DIGITAL TWIN
│   │   ├── physics.py         #   solidification, H2 solubility, Fe_crit(Si), sludge
│   │   ├── scm.py             #   causal structure, defect propensity
│   │   ├── dynamics.py        #   stateful line: die thermal state, wear, lots, drift
│   │   └── simulator.py       #   orchestration, propensity logging (for OPE)
│   ├── imaging/
│   │   ├── degrade.py         #   Regime B inline camera model
│   │   └── features.py        #   DINOv2 extraction + caching
│   ├── data/                  # schemas (pandera), lot-aware splits, exports
│   ├── models/                # xgb / ebm / tabpfn, calibration, conformal
│   ├── explain/               # grouped SHAP, ALE, anchors
│   ├── prescribe/             # constrained recommender + validation
│   ├── economics/             # COPQ, thresholds, Taguchi loss, sensitivity
│   ├── spc/                   # X-bar/R, I-MR, EWMA, Nelson rules, frozen limits
│   └── api/                   # FastAPI: routes, SSE, deps
├── services/
│   ├── linesim/               # publishes shots to MQTT (the "factory")
│   └── ingest/                # MQTT -> Postgres; zero business logic
├── web/                       # Vite + React app
├── powerbi/                   # .pbix + export contract docs
├── tests/                     # unit, integration (testcontainers), behavioural
├── notebooks/                 # EDA and ablations, all outputs regenerable
├── figures/                   # every figure produced by a script
├── docker-compose.yml
└── pyproject.toml
```

**Non-negotiable architectural seam:** `twin/` must never import from `models/`. The simulator is ground truth; the model is a consumer. If that dependency ever inverts, you have label leakage and the project is dead. Enforce it with an import-linter test in CI.

---

## 7. SPC — you have to build it

Research checked every Python SPC library. **All are dead:** `pyspc` (2016, GPL-3.0), `spcchart` (2015), `python-spc` (2012, **no licence declared — legally unusable**), `ControlCharts` (unmaintained teaching repo). There is no production-ready open-source Python SPC library in 2026.

**Build it. Scope is smaller than it looks (~1 week solo, tested):**
- Subgroup aggregation X-bar/R, X-bar/S, I-MR — the constants (A2, D3, D4, d2, c4, B3, B4) are a static lookup table. Half a day.
- EWMA + CUSUM — textbook formulas. One day.
- **Nelson rules 1–8** as a windowed evaluator (Western Electric is a subset). ~2 days write, ~2 days test.
- **Phase I / Phase II separation** — control limits **frozen** from a validated baseline period, stored as versioned rows with an effective-date range. **This is the #1 thing homegrown SPC gets wrong.**

Validate against R's `qcc` package: generate fixtures, run both, assert agreement. That comparison is itself a report artefact.

**Run the rule engine server-side, not in the chart library.** An out-of-control signal is a quality *event* that must be persisted and auditable — not a client-side rendering artefact that exists only while a tab is open.

**Do not make SPC limits adaptive or ML-driven.** A chart whose limits move for reasons an engineer cannot reproduce by hand is a chart they stop using.

---

## 8. Production-shaped details that are nearly free

These cost hours and buy disproportionate credibility. All come from real industrial practice.

| Practice | Cost | Value |
|---|---|---|
| **ISA-95 UNS topic tree** — `foundry/plant-01/hpdc/line-03/dc-machine/shot` | 1 h | Real industrial vocabulary; Mosquitto is already in the stack |
| **Never put images in MQTT** — publish object-store key + content hash | free | The correct pattern; multi-MB payloads wreck broker latency |
| **`trigger_seq` association, never timestamps** | 1 h | Timestamp joins are the classic silent-corruption bug |
| **Sparkplug-style sequence numbers** for gap detection | 2 h | Free gap detection on the telemetry stream |
| **Store-and-forward with an explicit backpressure policy** | 3 h | Document it: at 85 % disk, drop OK-classified images but **never** drop NOK evidence. Write it into the FMEA. |
| **Hash-chained append-only audit log** | 1 day | `BEFORE UPDATE OR DELETE` trigger that raises; `prev_hash`/`curr_hash` over a canonical row form; `pg_advisory_xact_lock` so concurrent writes can't fork the chain. Store **the model version and the explanation shown at the time** with every override — otherwise "what did the model actually tell her?" is unanswerable. |
| **Staged autonomy with the right asymmetry** | free (report) | shadow → advisory → **reject-only** → full. Let the AI reject before you let it accept: a false reject costs one casting, a false accept escapes to a customer. |
| **Alert budget from ISA-18.2** | 2 h | 6–12 alarms/hour/operator max; derive the threshold from the budget, don't alarm on single shots, use `k`-of-`n` persistence |
| **EU AI Act positioning** | 2 h (report) | Not Annex III (no category covers industrial QC); not Annex I unless it's a safety component. ⚠️ **Note the trap**: Annex III cat. 4 covers worker management — aggregate override metrics at shift/cell level, **never per operator**. |
| **Operator override log with reason codes** | 3 h | Simultaneously the active-learning source, the drift detector, the trust metric, and the audit trail |
| **Offline demo bundle** (`docker save` tarball) | 2 h | Your viva demo works when the wifi dies |

That last one is not optional. Build it in Week 1.
