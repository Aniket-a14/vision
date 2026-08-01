# DefectLab

Multi-modal casting defect prediction: a physics-based process twin fused with real industrial
imagery, under realistic inline-camera degradation.

```
322 tests · 2 enforced layer contracts · 12 CLI commands · 14 packages
```

## What this is

A quality-control system for Al-Si high-pressure die casting that predicts impeller defects by
fusing two channels available at different moments in the production cycle:

| Channel | Available | Source |
|---|---|---|
| Process parameters | before and during the shot | physics-based digital twin |
| Part image | after the shot | real Kaggle casting dataset |

The twin is simulated because no foundry releases its process logs. The images are real because a
simulated camera would make the vision result meaningless.

**Claim boundary.** The two channels are coupled only through the shared defect label. No
cross-modal interaction is claimed to be physical. The research question is therefore *how much
predictive performance an uncorrelated process channel recovers as image quality degrades* — a
question this design answers validly, because the channels' independence is what the coupling
scheme guarantees. See [docs/03-architecture-and-stack.md](docs/03-architecture-and-stack.md).

## Quick start

```bash
uv sync --extra ml --extra viz
uv run pytest tests -q
docker compose up -d          # UI on :8080, API on :8000
```

Always call the venv interpreter explicitly — a bare `python` is the system interpreter and fails
with `ModuleNotFoundError: No module named 'defectlab'`. `python -m pip` is also broken in this
venv; use `uv pip install`.

## Getting the dataset

The casting images are not redistributable, so download them yourself:

```bash
# needs ~/.kaggle/kaggle.json from https://www.kaggle.com/settings
uv run kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product
```

Extract so the layout is `data/raw/casting_data/{train,test}/{def_front,ok_front}/`, then:

```bash
uv run defectlab verify     # asserts 3758/2875 train, 453/262 test
uv run defectlab simulate   # twin -> paired parquet tables
uv run defectlab extract --backbone dinov2_s --regime both
```

`defectlab gates` prints the leakage and prevalence diagnostics without training anything,
and needs no dataset at all.

**Measured extraction cost on a 4-core i7-1195G7:** `resnet18` 28 min per regime,
`dinov2_s` 52 min per regime. Caches are keyed by `(split, backbone, regime)` and reused.

## Commands

| Command | Does | Cost |
|---|---|---|
| `verify` | check the casting dataset folder counts | instant |
| `gates` | leakage and prevalence diagnostics, no training | seconds |
| `simulate` | build the paired train and test tables | ~1 min |
| `extract` | cache image embeddings | 28–52 min |
| `ablate` | run the 3x2 modality-by-regime ablation | long |
| `figures` | render the sweep charts from a results table | seconds |
| `explain` | grouped SHAP attribution for one cell | ~1 min |
| `economics` | price one cell in money, not in AUC | ~1 min |
| `prescribe` | recommend setpoint changes for a risky shot | ~1 min |
| `export` | write the dashboard star schema to CSV + the Power BI project | ~1 min |
| `serve` | run the scoring API and live SSE feed | runs |
| `line` | publish and score shot telemetry over MQTT | runs |

## Results

**Fusion beats vision, and the gain grows with camera degradation.** 15 twin seeds on ResNet-18:
trend slope **+0.0303 AUC per severity step, t = 9.39, p < 1e-5, 15/15 seeds positive**.

Replicated in direction on DINOv2-S. Vision loses only 0.084 AUC there against ResNet's 0.161 and
the fusion gain shrinks in proportion, so the effect tracks *headroom*, not architecture. Quote the
seed-level **p = 0.056**, not the pooled 0.0046: both sweeps reuse the same five twin seeds and so
share a process channel and label vector, meaning `(backbone, seed)` is not an independent unit.

Two results that look like bugs and are not:

- **Vision sd is exactly 0.0000 at every severity.** Images, their order and the label vector are
  identical across twin seeds; all paired variance comes from the fusion side. This is why the
  paired test is the right one.
- **Severity 1.5 gives a smaller gain (+0.0031) than severity 1.0 (+0.0051)**, 8/15 seeds
  positive, p = 0.291. It survived widening from 5 to 15 seeds, so it is real and unexplained.

**Effective sample size is the number of alloy lots, not the number of parts.** Chemistry is drawn
per lot and shared by every part in it. This is the central methodological finding of the project
and the reason the splits are lot-disjoint.

### Economics — including the negative result

Corrected from the 57 % research prevalence to a 3 % line, under a PAF cost model:

| Policy | Cost/shot |
|---|---|
| Fused gate | €1.40 |
| Inspect everything | €3.36 |
| Ship everything | €9.00 |

Report the saving against 100 % inspection (**€1,613–2,289 per 1,000 shots** across escape
multipliers M = 10–50×), not against ship-everything (€2,529–16,253). The second is mostly a
restatement of the escape multiplier, which was guessed.

A process-only gate held to the ISA-18.2 alarm budget costs **€6.12/shot — worse than inspecting
everything at €3.36**. Process telemetry alone does not support an economically viable gate at a
usable alarm rate, and that gap is exactly what the image channel buys. It is the strongest
argument in the project and it is a failure.

Measured live: 14.2 % of 3,000 streamed shots = **8.5 alarms/hour**, inside the 6–12 band.

### Prescription

On the riskiest of 600 shots: risk **1.0000 → 0.0174**, margin **+16.6 logits**, all changes
capped at their ramp limits. Stability 1.000 under ±20/35/50 % weight perturbation — which means
"for shots this far from nominal the advice does not depend on the weights", not "the advice is
verified".

## Architecture rules

Two contracts are enforced in CI by `import-linter`:

1. **`twin/` never imports from `models/` or `data/`.** The simulator is ground truth; models are
   consumers. If that dependency inverts, the project has label leakage.
2. **Layered imports** — `api → prescribe → explain → models → imaging → data → twin → config`.

A consequence: `economics` sits *below* `models`, so it takes arrays and cost parameters and never
a fitted model. `CostMatrix` lives in `economics.costs` and is re-exported from `models.thresholds`.

## Gates

Enforced as tests in `tests/test_gates.py`:

| Gate | Threshold | Current |
|---|---|---|
| Process-only ROC-AUC | 0.80 – 0.88 | ~0.85 |
| Max feature-label correlation | < 0.35 | ~0.34 |
| Controllable levers carry signal | std > 0.1 | pass |
| Lots never span a split | disjoint | pass |

An AUC *above* the band means the simulator is too easy to be credible; real foundries cannot
predict defects this well from process data alone.

## Serving

`defectlab serve`, or the whole stack with `docker compose up -d`. The model is fitted at startup
and is deterministic given the seed, so a restart serves an identical model — which is what makes
the audit hashes mean anything across a redeploy.

- `POST /score` — prior-corrected risk, conformal prediction set, audit hash. **Process telemetry
  only, no images:** a real cell has telemetry for every shot but a photograph only for parts that
  reach the camera. The fusion model is the offline result; this is the online one.
- `GET /stream` — SSE, not WebSockets. One-way traffic, so plain HTTP survives proxies and
  reconnects itself.
- `GET /audit` — walks the hash chain from genesis and reports the first index that fails.
- Impossible shots are rejected (422) and **never reach the audit log**. A decision that was never
  made must not appear to have been made.

**The chain proves integrity and ordering, not authenticity.** Anyone who can append can recompute
it from genesis. Real tamper-evidence needs the head published where the writer cannot reach it.

### MQTT edge

`defectlab line --loopback --limit 120 --cycle 0` runs producer and consumer in one process against
an in-memory broker, so a demo needs no Mosquitto. `--role publish` / `--role gate` split them
across a real one.

- **Telemetry is at-most-once; verdicts are at-least-once.** A stale reading is worthless; a lost
  reject is a shipped defect. At-least-once means redelivery, so the gate is idempotent on
  `shot_index` and duplicates are not written to the audit chain.
- **The last will is the reason to use MQTT at all.** Measured: the will fired **6.0 s** after
  `kill` on the host and **2.1 s** after `docker kill` on the container.
- Status is **retained**, so a dashboard opened mid-shift learns the cell state immediately.
- The gate scores with the same `Scorer` the HTTP endpoint serves — one model, one threshold, one
  audit chain.

## Dashboard

`defectlab export` writes `data/exports/*.csv` plus `powerbi/DefectLab.pbip`: 9 typed tables, 5
relationships and 15 DAX measures, all generated from `export/schema.py` so the model cannot drift
from the data. Open the PBIP in Desktop, refresh, lay out the four pages, File → Save As.

**The two grains must not be merged:**

| Table | Use for | Never for |
|---|---|---|
| `fact_shot` | model quality — escape rate, overkill, attribution | anything over time |
| `fact_production` | anything with a clock — trends, shifts, control charts | model accuracy |

`fact_shot` is the held-out evaluation set: oversampled and grouped by label (lag-1 label
autocorrelation 0.997), so it carries **no timestamp on purpose**. Building the SPC page on it
signalled on 48 % of points, every one an artefact of row order.

Measures live on a table called **`Metrics`**, not `Measures` — that name is the MDX measures
dimension and the tabular schema reserves it.

## Deployment

`docker compose up --build` → UI on **:8080**, API on **:8000**. `postgres` and `redis` are in the
`infra` profile and stay down.

- **`xgboost-cpu`, not `xgboost`.** The Linux wheel pulls `nvidia-nccl-cu12`, 289 MB of CUDA the
  service never calls. Image 2.01 GB → 1.02 GB.
- **The serving path must not reach OpenCV.** `models/__init__` used to re-export `run_cell`, which
  needed `imaging.Regime`, which imports cv2. Import `ablation` directly, never via the package
  init. Pinned by a subprocess test with `sys.modules['cv2'] = None`, since the dev venv has cv2.
- **The API and the MQTT gate get separate audit volumes.** The hash chain is single-writer.

`python deploy/bundle.py` produces a 260 MB offline bundle. Verified by deletion, not inspection:
`docker image prune -af` reclaimed 9.875 GB, then the stack came up from the tarball alone. One
flipped byte makes `--verify` report `BUNDLE CORRUPT`.

## Verify the build

```bash
./.venv/Scripts/python.exe -m pytest -q       # 322 tests
./.venv/Scripts/ruff.exe check . && ./.venv/Scripts/ruff.exe format --check .
./.venv/Scripts/lint-imports.exe              # 2 contracts, both KEPT
```

## Known gaps

Named deliberately rather than discovered:

- **No operator identity.** An override records what and why, not by whom.
- **Audit authenticity.** Integrity and ordering only; the head is not externally published.
- **The escape multiplier is a guess**, which is why the headline saving is quoted against 100 %
  inspection instead.
- **The severity-1.5 anomaly is unexplained.**
- **No cross-modal physics** — declared, not discovered.

## Documents

| Document | Contents |
|---|---|
| [01-physics-twin-corrections.md](docs/01-physics-twin-corrections.md) | The six physics corrections the twin implements |
| [02-decision-layer.md](docs/02-decision-layer.md) | Explainability, prescription, thresholds, COPQ, SPC |
| [03-architecture-and-stack.md](docs/03-architecture-and-stack.md) | Architecture, model choices, full tech stack |
| [04-execution-plan.md](docs/04-execution-plan.md) | Four-week plan, gates, de-scope ladder |
| [05-state-of-play.md](docs/05-state-of-play.md) | **Read this first when resuming.** Current state, settled results |
| [06-powerbi.md](docs/06-powerbi.md) | Page-by-page dashboard build guide |
| [07-app.md](docs/07-app.md) | The React operator UI |
| [08-deploy.md](docs/08-deploy.md) | Compose stack and images |
| [09-offline-bundle.md](docs/09-offline-bundle.md) | The air-gapped bundle |
| [10-report.md](docs/10-report.md) | Report draft |
| [11-slides.md](docs/11-slides.md) | Viva deck draft |

## Layout

```
src/defectlab/
├── config.py          typed settings
├── twin/              digital twin - ground truth, imports nothing above it
│   ├── constants.py   alloy and physical constants
│   ├── parameters.py  parameter specs, actionability, sampling
│   ├── physics.py     solidification, H2 solubility, Fe_crit, sludge
│   ├── propensity.py  structural causal model
│   ├── dynamics.py    stateful line: thermal state, wear, lots, drift
│   └── simulator.py   orchestration
├── data/              dataset discovery, twin-to-image pairing, contracts, splits
├── imaging/           backbones, degradation regimes, cached embeddings
├── models/            the 3x2 ablation, thresholds, conformal prediction
├── explain/           grouped SHAP, ALE, Anchors
├── economics/         prior shift, PAF costing, Taguchi, sensitivity
├── spc/               X-bar/R, I-MR, EWMA, Nelson rules 1-8
├── prescribe/         interventional surrogate, ramp-limited advice
├── export/            star schema, CSV export, generated Power BI project
├── api/               FastAPI scoring, SSE stream, hash-chained audit
├── edge/              MQTT line simulator and scoring gate
├── report/            sweep figures
└── cli/               argument parsing and command bodies

app/        React + TypeScript operator UI
deploy/     Dockerfiles, nginx config, offline bundle builder
powerbi/    generated PBIP - the plain-text form of the dashboard
```

## Conventions that are load-bearing

- **Search and score on the margin, not on probability.** This has bitten five times. A risky shot
  sits where the sigmoid is flat, so a real improvement of 16 logits reads as a probability change
  of zero. Raw probability has skew +4.25 and excess kurtosis +18.9 and fails the Shewhart
  normality assumption; the logit is +1.02 and +1.19.
- **Taguchi: read `baseline_ratio`, never `mean_loss`.** With Δ₀ = 3σ a parameter drawn at its own
  spread costs A₀/9 whatever A₀ is, so ten parameters sum past the value of the part.
- Comments explain *why*, and are worth most where a reader would otherwise assume a bug.
- Progress logging goes to **stderr**. Python block-buffers stdout when redirected, which is why
  `ablate` once appeared to hang for 17 minutes.
- `.env` holds an HF token. It is gitignored at `.gitignore:20`.
