# State of play

Written 2026-08-01. Read this first when resuming; it is the only file that needs to be current.

## Verify the build in one command

```
./.venv/Scripts/python.exe -m pytest -q      # 243 tests
./.venv/Scripts/ruff.exe check . && ./.venv/Scripts/ruff.exe format --check .
./.venv/Scripts/lint-imports.exe             # 2 contracts, both KEPT
```

Always call `./.venv/Scripts/python.exe` explicitly. A bare `python` is the system
interpreter and fails with `ModuleNotFoundError: No module named 'defectlab'`; this has
already cost one silent overnight run. `python -m pip` is also broken in this venv — use
`uv pip install`.

## What exists

| Layer | State | Entry point |
|---|---|---|
| `config` | done | `settings` |
| `twin` | done | `run_line`, `score` |
| `data` | done | `defectlab simulate` |
| `imaging` | done, all 12 caches extracted | `defectlab extract` |
| `models` | done | `defectlab ablate` |
| `report` | done | `defectlab figures` |
| `explain` | done — grouped SHAP, ALE, Anchors | `defectlab explain` |
| `economics` | done — prior shift, PAF, Taguchi, sensitivity | `defectlab economics` |
| `spc` | done — X-bar/R, I-MR, EWMA, Nelson 1–8 | no CLI yet |
| `prescribe` | done — interventional surrogate, ramp-limited advice | `defectlab prescribe` |
| `export` | done — validated star schema + generated PBIP | `defectlab export` |
| `api` | done — score, prescribe, SSE stream, hash-chained audit | `defectlab serve` |
| `edge` | done — MQTT line simulator and scoring gate | `defectlab line` |
| `app/` | done — React operator UI | `npm run dev` in `app/` |
| `deploy/` | done — compose stack, two images | `docker compose up --build` |

Not started: offline bundle, report, slides, and the **Power BI `.pbix` report pages**, which are
the last manual step on a hard rubric requirement.

## The results that are settled

Do not re-run these to check them; they are recorded in `docs/04-execution-plan.md`.

- **Fusion beats vision, and the gain grows with camera degradation.** 15 twin seeds,
  ResNet-18: trend slope **+0.0303 per severity step, t = 9.39, p < 1e-5, 15/15 seeds
  positive**.
- **Replicated on DINOv2-S** in direction. Vision loses only 0.084 AUC there against
  ResNet's 0.161, and the fusion gain shrinks in proportion — so the effect tracks
  *headroom*, not architecture. Seed-level test of that relationship: t = −2.66, **p = 0.056**.
  Quote the 0.056, not the pooled 0.0046: both sweeps reuse the same 5 twin seeds and
  therefore share a process channel and label vector, so (backbone, seed) is not an
  independent unit.
- **One open anomaly.** Severity 1.5 gives a smaller gain (+0.0031) than severity 1.0
  (+0.0051), with 8/15 seeds positive and p = 0.291. It survived widening from 5 to 15 seeds,
  so it is real and unexplained. The resolution-boundary idea in the plan is a hypothesis and
  is labelled as one.
- **Vision sd is exactly 0.0000 at every severity.** Images, their order and the label vector
  are identical across twin seeds. All paired variance comes from the fusion side. This is
  correct, not a bug, and it is why the paired test is the right one.
- **Effective sample size is the number of alloy lots, not the number of parts.** This is the
  central methodological finding of the project.

## Measured economics

`defectlab economics --severity 2` on ResNet-18 fusion, corrected to a 3 % line:

- Gate €1.40/shot vs ship-everything €9.00 vs inspect-everything €3.36, per 1,000 shots.
- **Report the saving against 100 % inspection (€1,613–2,289 across M = 10–50×), not against
  ship-everything (€2,529–16,253).** The second is mostly a restatement of the escape
  multiplier, which was guessed. The first is positive across the whole range.
- Alert rate 16.5 % of shots = **9.9 alarms/hour** on a one-minute cycle, inside the ISA-18.2
  6–12/hour band *without the constraint being imposed*. That is a result, and it closes the
  question deferred at `models/pipeline.py:19`.
- **Search and score on the margin, not on probability.** This has now bitten four times —
  including the SPC risk chart, where the raw probability has skew +4.25 and excess kurtosis
  +18.9 and fails the Shewhart normality assumption outright (logit: +1.02 and +1.19). Also
  SHAP attribution, the prescribe search, and the prescribe robustness measure. A risky shot
  sits where the sigmoid is flat, so a real improvement of 16 logits reads as a probability
  change of zero. Any new optimiser or effect measure should default to logits.
- **Taguchi: read `baseline_ratio`, never `mean_loss`.** With Δ₀ = 3σ a parameter drawn at its
  own spread costs A₀/9 whatever A₀ is, so ten parameters sum past the value of the part. The
  ratio divides that artefact out: `pour_temp_c` 2.64 (real drift), four params at 0.98–1.02
  (the control), `die_temp_c` 0.49 (thermal inertia), chemistry 0.43–0.67 (lot-level).

## Measured prescription

`defectlab prescribe --seed 7` on the riskiest of 600 shots raises both plunger velocities and
the pour temperature back toward nominal, each capped at its ramp limit: risk 1.0000 -> 0.0174,
margin +16.6 logits. Stability is 1.000 under ±20/35/50 % weight perturbation.

**Quote that stability with its reason.** It is not vacuous — every recommendation worsens at
least one mechanism, so a bad reweighting could in principle flip it. It is strong because the
improvement dominates the worsening by ~100×, and ±50 % is at most a 3× swing. The claim is
"for shots this far from nominal the advice does not depend on the weights", not "the advice is
verified". Full write-up in `docs/04-execution-plan.md`.

## The export contract

`defectlab export` writes eight CSVs plus a `manifest.json` of row counts and digests. Two
grains, and they must not be merged:

- **`fact_shot`** — the held-out evaluation set. Model metrics live here. **It carries no
  timestamp on purpose**: it is oversampled and grouped by label (lag-1 label autocorrelation
  0.997, one run of 453 identical labels), so it has no time axis and stamping a clock on it
  invents one.
- **`fact_production`** — a contiguous run of the line. Everything time-indexed hangs off this:
  the clock, `dim_date`, and every control chart. Scored on process telemetry alone, which is
  what a continuous monitor actually reads between imaged parts.

Building the SPC page on the evaluation set signalled on 48 % of points, all of it an artefact
of row order. That mistake is the reason the two grains are separate.

## The serving layer

`defectlab serve` runs it; verified live. The model is fitted at startup (deterministic given
the seed, so a restart serves an identical model — which is what makes the audit hashes mean
anything across a redeploy).

- `POST /score` — prior-corrected risk, conformal prediction set, and the audit hash of the
  decision. **Process telemetry only, no images**: a real cell has telemetry for every shot but
  a photograph only for parts that reach the camera, so this is where the process channel earns
  its keep. The fusion model is the offline result; this is the online one.
- `GET /stream` — SSE, not WebSockets: the traffic is one-way, so plain HTTP survives proxies
  and reconnects itself. `?limit=` bounds it; without it the feed runs forever.
- `GET /audit` — walks the hash chain from genesis and reports the first index that fails.
- Requests are validated against the twin's machine limits, so an impossible shot is rejected
  (422) and **never reaches the audit log** — a decision that was never made must not appear to
  have been made.

**The audit chain proves integrity and ordering, not authenticity.** Anyone who can append can
recompute the chain from genesis. Real tamper-evidence needs the head published somewhere the
writer does not control. Say this before an examiner does.

## The MQTT edge

`defectlab line --loopback --limit 120 --cycle 0` runs producer and consumer in one process
against an in-memory broker, so the demo needs no Mosquitto. `--role publish` / `--role gate`
split them across a real broker; `--host`/`--port` point at it.

- **The QoS split is the design.** Telemetry is at-most-once — a reading that needed
  retransmitting is already stale. Verdicts are at-least-once — a lost reject is a shipped
  defect. At-least-once means redelivery, so `Gate` is idempotent on `shot_index` and a
  duplicate is **not** written to the audit chain: the chain records decisions the line made,
  not redeliveries the broker made.
- **The last will is the reason to use MQTT at all.** The broker holds an `offline` status
  message and publishes it *for* the cell if the connection drops without a clean disconnect.
  An SSE stream that stops just stops, and every consumer has to invent its own timeout. The
  will must be armed before `connect()`, which is what `test_the_will_is_registered_before_the_
  connection` pins.
- Status is **retained**, so a dashboard opened mid-shift learns the cell state immediately
  rather than after a cycle.
- The gate scores with `api.scoring.Scorer` — the same object the HTTP endpoint serves. One
  model, one threshold, one audit chain, so a decision cannot depend on the transport it
  arrived over.

## The served threshold was on the wrong scale

Found by the MQTT gate, because it was the first thing to score a whole nominal line and report
an aggregate — the API smoke test only ever scored one deliberately-bad shot. It flagged
**83 %**.

`FittedModel.threshold` is the cost optimum at the **research** prevalence (57 %). `Scorer.risk`
is prior-corrected to the **line** prevalence (3 %). Comparing them put the two sides of the
inequality on different scales. `api/scoring.py` now re-chooses the threshold on corrected
scores at the deployment prior, on a held-out quarter the estimator never saw.

That fix alone gave 66 %, and the reason is a result rather than a bug: **process telemetry
alone barely separates the classes**, so with an escape at 100x an inspection the unconstrained
optimum wants to inspect roughly half the line. Economically right, operationally unusable.
So the served gate also imposes the ISA-18.2 budget (12 alarms/hour on a 60 s cycle = a 20 %
alert rate), and **that constraint binds here although it did not bind offline**.

Quote the price of it, because it is steep and it is the argument for fusion:

| threshold | cost/shot | escape rate | alert rate |
|---|---|---|---|
| cost optimum, 0.0100 | €2.50 | 0.074 | 0.485 |
| budgeted, 0.2122 | €6.12 | 0.654 | 0.022 |

At €6.12 the budgeted process-only gate is **worse than inspecting everything** (€3.36). The
honest statement is that process telemetry alone does not support an economically viable gate at
a usable alarm rate, and that is exactly what the image channel buys.

Measured live: 14.2 % of 3,000 streamed shots = **8.5 alarms/hour**, inside the band. The first
~500 shots of a stream run hotter (31.7 %) because `stream_line` starts from a worn die
(`tool_wear_shots` ≈ 47k, settling to ≈ 27k). That is a real high-risk regime, not a defect —
the budget is a long-run design target, not a per-window cap.

## The operator app

`app/`, React + TypeScript + Vite, Mantine and uPlot. Full write-up in `docs/07-app.md`; the
short version:

- Live line (SSE) with the risk chart **on the logit axis**, alarm rate in **alarms/hour**, and a
  shot list. Click a shot to inspect it.
- Inspector: the anchor rule, the ramp-limited prescription, the readings, and the override.
- Sandbox: sliders bounded by `/parameters`, with lot-level and maintenance parameters locked.
- Four new endpoints back it: `/explain`, `/prescribe`, `/parameters`, `/reasons`, `/override`.

**The stream does not audit; selecting a shot does.** A shot nobody looked at is not a decision
anyone made. An override must carry the `audit_hash` of a decision the log actually contains
(404 otherwise) and the **explanation that was on screen at the time** — re-deriving it later
would record what the model says now, which is the one thing an audit of a past decision must
not do. It is an attestation, not a verified fact, and the docstring says so.

**Not done: operator identity.** The override records what and why, not by whom. A quality record
with no signatory is incomplete; say it before an examiner does.

## Deployment

`docker compose up --build` → UI on **:8080**, API on **:8000**. Full write-up in
`docs/08-deploy.md`. Two things that cost real time to find:

- **`xgboost-cpu`, not `xgboost`.** The Linux wheel pulls `nvidia-nccl-cu12`, 289 MB of CUDA the
  service never calls. Image **2.01 GB → 1.02 GB**. The whole point of the `serve` extra was to
  keep torch out; shipping NCCL instead would have been the same mistake twice.
- **The serving path must not reach OpenCV, and it did.** The first container start crashed with
  `ModuleNotFoundError: No module named 'cv2'`. `models/__init__` re-exported `run_cell`, which
  needed `imaging.Regime`, which imports cv2 — so scoring process telemetry pulled in a vision
  dependency. `AblationResult` and `run_cell` moved to `ablation.py` where they belong, and
  `models/__init__` no longer re-exports them. **Import `ablation` directly, never via the
  package init.** No unit test could catch this locally (the dev venv has cv2), so it is pinned
  by importing each serving module in a subprocess with `sys.modules['cv2'] = None`.
- **The API and the MQTT gate get separate audit volumes.** The hash chain is **single-writer** —
  two processes appending to one file each compute `previous` from their own in-memory head, so
  it would not verify. One log per decision-maker is the honest model anyway.

`postgres` and `redis` are in the `infra` profile and stay down; nothing in the build reads them.

## Verified against a real MQTT broker

Not only the loopback transport:

- 20 telemetry, 20 verdicts, 2 status messages, read by an **independent subscriber**. Verdicts
  carried `audit_hash`, so the MQTT path audits.
- Retained status reached a subscriber that connected **after** the run ended.
- **The last will fired 6.0 s after `kill`** on the publisher. This is the behaviour the loopback
  cannot test and the reason MQTT is here rather than a second SSE feed.

## Conventions that are load-bearing

- **Layer contract in `pyproject.toml` is enforced.** `economics` sits *below* `models`, so it
  takes arrays and cost parameters and never a fitted model. `CostMatrix` lives in
  `economics.costs` for that reason and is re-exported from `models.thresholds`.
- **Style, set by the user and still in force:** no long comments, single line at most; proper
  folder structure; clean modular code; no god functions.
- Comments explain *why*, and are worth most where a reader would otherwise assume a bug —
  the empty anchor, the path-dependent SHAP, the zero vision sd, the Taguchi absolute scale.
- Progress logging goes to **stderr**. Python block-buffers stdout when it is redirected to a
  file, which is why `ablate` once appeared to hang for 17 minutes.
- `.env` holds an HF token. Grep it by **key name only**, with `output_mode: "count"`; never
  read or print the value. It is gitignored at `.gitignore:20`.

## Next, in order

1. **Power BI `.pbix`** — the semantic model is **done and generated**; only the visuals and
   the Save As remain. `defectlab export` writes `data/exports/*.csv` and
   `powerbi/DefectLab.pbip` (9 typed tables, 5 relationships, 15 DAX measures, all generated
   from `export/schema.py` so they cannot drift). **Read `docs/06-powerbi.md`** — it has the
   page-by-page build guide. Open the PBIP in Desktop, lay out four pages, File → Save As.
   This is the last manual step on the hard rubric item and it is maybe an hour.
2. Deploy (`docker compose` already defines postgres, mosquitto and redis), offline bundle,
   report, slides.

A note on ordering, learned the hard way: breadth-first beats depth-first here. Adding seeds to
an already-significant result optimises the thing most recently looked at, while a
rubric-mandated deliverable sits at zero.
