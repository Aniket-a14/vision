# Multi-modal casting defect prediction: a physics-grounded digital twin fused with inline vision

Report draft. Every number here is measured and traceable to a command; nothing is estimated.
Sources: `docs/04-execution-plan.md` (full results), `docs/05-state-of-play.md` (current state).

---

## Abstract

High-pressure die casting produces defects whose causes are in the process — pour temperature,
plunger velocity, intensification pressure — but whose detection happens downstream, at a camera
that sees only the finished surface. This project asks whether the process channel adds anything
once a vision model is already good, and finds that the answer depends entirely on how good the
camera is.

Fusing simulated process telemetry from a physics-based twin with embeddings of real casting
images (Kaggle impeller dataset, 7,348 parts) raises AUC over vision alone by **+0.0303 per unit
of camera degradation** (t = 9.39, p < 1e-5, 15/15 twin seeds positive). At an undegraded camera
the gain is +0.0002 and indistinguishable from zero; at severity 3, where the camera costs vision
0.161 AUC, fusion recovers 45 % of that loss on every seed tested.

The result is then priced. Corrected from the 57 % research prevalence to a 3 % line by Elkan
odds rescaling, the fusion gate costs **€1.40 per 1,000 shots against €3.36 for inspecting
everything**, and the saving holds across an escape multiplier from 10× to 50×. The same
machinery applied to a process-only gate shows the opposite: constrained to an alarm rate an
operator can work, it costs **€6.12 against €3.36 — worse than inspecting everything.** That
contrast is the strongest single argument in the project for the image channel.

---

## 1. Problem and contribution

**The gap.** Casting defect datasets are images. Casting *physics* is process telemetry. The two
are almost never available together, so the literature optimises one channel in isolation and the
question "does the process channel earn its instrumentation cost?" goes unanswered.

**The approach.** A physics-grounded digital twin generates process telemetry with known causal
structure; each simulated shot is paired with a real image of the matching label. The twin is
ground truth for the mechanism, the images are ground truth for appearance, and the pairing lets
both be varied independently — which is what makes the degradation sweep possible at all.

**Contributions.**

1. A fusion result with a **stated direction of dependence**: the benefit is a function of camera
   headroom, not a fixed number, and it is reported across the whole degradation ladder rather
   than at the point that flatters it.
2. A **replication on a second backbone** that reproduces the direction and not the significance,
   reported as such.
3. An **economic layer** that prices the model on the line rather than on the test set, and
   reports the conservative comparison.
4. A **methodological finding** that governs every statistic in the report: the effective sample
   size is the number of alloy lots, not the number of parts.
5. A running system — API, MQTT edge, operator UI, containerised and reproducible offline.

---

## 2. Method

### 2.1 The twin

Eleven process parameters with physical units, machine limits, nominal values and an
**actionability class** (immediate, slow, lot-level, maintenance). Defect probability is composed
from named failure mechanisms — gas porosity, shrinkage, cold shut, flash — rather than a single
logistic function, so an attribution can be read back to a mechanism.

Two design decisions matter downstream:

- **Lot-level structure.** Alloy chemistry and die state vary by lot, not by shot. This makes
  shots within a lot correlated, which is the origin of the effective-sample-size finding in §4.5.
- **`stream_line` starts from a worn die.** `tool_wear_shots` begins near 47k and settles toward
  27k, so the first ~500 shots of a run are a genuinely higher-risk regime. This is realistic and
  it is why alarm rates measured over a short window run hot (§4.6).

### 2.2 Pairing and prevalence

Each simulated shot is paired with a real Kaggle image whose label matches the sampled outcome.
The twin's uncalibrated defect probability is near 1.0, so an explicit **target prevalence** is
always supplied: 0.567 for the research datasets (matching the image corpus), 0.03 for the line.
Nothing in the codebase reads a defect label it did not itself set.

### 2.3 Degradation

An `InlineCamera` model applies blur, noise, exposure drift and downscaling at a severity
parameter. The ladder (0.5, 1.0, 1.5, 2.0, 3.0) was **fixed before any result was seen** and the
whole curve is reported.

Image embeddings are cached per (backbone, split, regime, severity). Because images and their
order are identical across twin seeds, widening the study from 5 to 15 seeds cost model fits only
— 13 minutes, no re-extraction.

### 2.4 Models

Gradient-boosted trees on process features and PCA-reduced image embeddings, isotonically
calibrated, with Mondrian (class-conditional) conformal prediction for abstention. Calibration
comes first: cost-optimal thresholds are meaningless on uncalibrated scores.

Mondrian conformal is not a stylistic choice. Class-conditional quantiles are **invariant under
label shift**, which is exactly the shift applied when moving from the 57 % research prevalence to
the 3 % line. A marginal conformal predictor would have needed re-calibrating; this one does not.

### 2.5 Economics

Three inputs kept deliberately separate: what the classifier does (per-class error rates), what
the line looks like (prevalence), and what things cost. Counting confusion-matrix cells would
fuse the first two and price a factory that does not exist.

- **Prior correction** by Elkan odds rescaling from source to target prevalence.
- **PAF accounting** (prevention, appraisal, internal failure, external failure) and COPQ.
- **Escape multiplier M** stated as a range, 10–50×, never a point estimate — it is a guess and
  the report treats it as one.
- **Taguchi quadratic loss** `L = k(y − m)²` with `k = A₀/Δ₀²` for continuous severity.

### 2.6 Decision and monitoring layers

- **SPC**: X-bar/R, I-MR and EWMA with frozen Phase I limits and Nelson rules 1–8.
- **Prescription**: a surrogate fitted on a **randomised interventional** design from the twin, so
  it is unconfounded by construction, searched by greedy coordinate descent under ramp limits,
  actionability constraints and a sparsity cap.
- **Robustness**: the twin's mechanism weights are perturbed ±20/35/50 % and the fraction of
  recommendations whose direction survives is reported.

---

## 3. System

| Layer | What it is |
|---|---|
| `defectlab serve` | FastAPI: `/score`, `/explain`, `/prescribe`, SSE `/stream`, hash-chained `/audit` |
| `defectlab line` | MQTT producer and gate, or both against an in-process broker |
| `app/` | React operator UI: live line, shot inspector, what-if sandbox, override |
| `defectlab export` | validated star schema plus a generated Power BI PBIP |
| `docker compose up` | broker, API, line simulator, gate, UI |
| `deploy/bundle.py` | 260 MB offline bundle for a demo with no network |

**The gate is served on process telemetry only.** This is not a shortcut: a real cell has
telemetry for every shot but a photograph only for parts that reach the camera, so a per-shot
endpoint is exactly where the process channel earns its place. The fusion model is the offline
result; the served one is the online counterpart.

**Audit.** Every scored decision is hash-chained. Stated plainly because it will be asked: the
chain proves **integrity and ordering, not authenticity** — anyone who can append can recompute
it from genesis. Real tamper-evidence needs the head published somewhere the writer does not
control.

---

## 4. Results

### 4.1 Fusion gain grows with camera degradation

ResNet-18, 15 twin seeds, paired within seed.

| severity | vision AUC | fusion gain | sd | seeds + | p |
|---|---|---|---|---|---|
| 0.5 | 0.9985 | +0.0002 | 0.0007 | 9/15 | 0.263 |
| 1.0 | 0.9847 | +0.0051 | 0.0034 | 14/15 | <0.001 |
| 1.5 | 0.9671 | +0.0031 | 0.0111 | 8/15 | 0.291 |
| 2.0 | 0.8984 | +0.0352 | 0.0228 | 14/15 | <0.001 |
| 3.0 | 0.8374 | +0.0727 | 0.0290 | **15/15** | <0.001 |

**Trend: +0.0303 per unit severity, t = 9.39, p < 1e-5, 15/15 seeds positive.**

The headline is not "fusion beats vision". It is that **the benefit is a function of how much
headroom the camera has lost**. At severity 0.5 there is nothing to add and the model adds
nothing. An earlier five-seed run reported a null; that null was correct, and measured at the one
severity where vision has no headroom to lose.

### 4.2 Replication on a second backbone

| backbone | vision 0.5 → 3.0 | AUC lost | slope | t | p | seeds + |
|---|---|---|---|---|---|---|
| `resnet18` | 0.9985 → 0.8374 | 0.161 | +0.0258 | 3.39 | **0.028** | 5/5 |
| `dinov2_s` | 0.9994 → 0.9157 | 0.084 | +0.0076 | 1.84 | 0.140 | 4/5 |

**The direction replicates; the significance does not.** Both sweeps were run, so both are shown.

DINOv2-S is the more robust encoder and loses half as much AUC to the same camera, so there is
half as much for the process channel to recover — consistent with the headroom account rather
than with an architecture effect. The seed-level test of that relationship gives **t = −2.66,
p = 0.056**.

**Quote the 0.056, not the pooled 0.0046.** Both sweeps reuse the same five twin seeds and
therefore share a process channel and a label vector, so (backbone, seed) is not an independent
unit and the pooled test overstates its evidence.

### 4.3 An open anomaly

The gain at severity 1.5 (+0.0031) is **lower** than at severity 1.0 (+0.0051) despite worse
imaging, with 8/15 seeds positive, p = 0.291, and a spread three times wider (sd 0.0111 vs
0.0034). It survived tripling the seed count, so it is not sampling noise.

Vision itself degrades smoothly through that point, so the dip is in the fusion model rather than
the camera. The neighbouring severities cross a resolution boundary (96 px at 1.0, 54 px at 1.5,
31 px at 2.0) and 1.5 is where effective resolution first falls below the backbone's 64 px
receptive field — **but that is a hypothesis, not a finding.** Reported as an open anomaly.

### 4.4 Vision variance is exactly zero

Vision sd is **0.0000** at every severity, because the images, their order and the label vector
are identical across twin seeds. Only the process channel varies. This is correct rather than a
bug, and it is why the paired within-seed test is the right one: all of the paired variance comes
from the fusion side.

### 4.5 The effective sample size is the number of lots

The central methodological finding. Alloy chemistry and die state vary by lot; shots within a lot
are correlated. Treating 7,348 parts as 7,348 independent observations would overstate every
confidence interval in the report. **The unit is the lot, not the shot.**

The same structure appeared again, unprompted, in the SPC work: charting the risk score showed
autocorrelation of 0.23 at lag 1 that was still 0.16 at lag 20. That is not AR(1) memory — it is
a level shift between lots. An AR(1) residual chart was fitted and barely helped, which is how the
structure was identified.

### 4.6 Economics

`defectlab economics --severity 2`, ResNet-18 fusion, corrected to a 3 % line, per 1,000 shots:

| policy | cost/shot |
|---|---|
| **gate** | **€1.40** |
| inspect everything | €3.36 |
| ship everything | €9.00 |

**Report the saving against 100 % inspection: €1,613–2,289 across M = 10–50×.** The saving against
ship-everything (€2,529–16,253) is mostly a restatement of the escape multiplier, which was
guessed. The first is positive across the whole range and does not depend on the guess.

**The alert rate is 16.5 % of shots = 9.9 alarms/hour on a one-minute cycle — inside the ISA-18.2
band of 6–12/hour without the constraint being imposed.** That is a result rather than a design
choice, and it closes a question deferred at `models/pipeline.py:19`.

**Taguchi: read the ratio, never the absolute euros.** With Δ₀ = 3σ a parameter drawn at its own
spread costs A₀/9 whatever A₀ is, so ten parameters sum past the value of the part. The
`baseline_ratio` divides that artefact out:

| parameter | ratio | reading |
|---|---|---|
| `pour_temp_c` | 2.64 | real drift |
| four control params | 0.98–1.02 | the control |
| `die_temp_c` | 0.49 | thermal inertia |
| chemistry | 0.43–0.67 | lot-level, not shot-level |

### 4.7 The process channel cannot run an economic gate alone

The serving layer re-derives its threshold at the deployment prior and then imposes the ISA-18.2
budget (12 alarms/hour on a 60 s cycle = 20 % alert rate). **The budget binds here although it did
not bind in the offline study**, and it is expensive:

| threshold | cost/shot | escape rate | alert rate |
|---|---|---|---|
| cost optimum, 0.0100 | €2.50 | 0.074 | 0.485 |
| budgeted, 0.2122 | €6.12 | 0.654 | 0.022 |

At €6.12 the budgeted process-only gate is **worse than inspecting everything at €3.36**.

Process-only scores bunch at mean 0.54, median 0.53 on a 57 %-defective set — the channel barely
separates the classes — so with an escape at 100× an inspection the unconstrained optimum wants
to inspect roughly half the line: economically correct, operationally unusable.

**This is the strongest argument for fusion in the project, and it is a negative result about the
process channel.** It was found by instrumenting the demo, not by the study.

Measured live: 14.2 % of 3,000 streamed shots = **8.5 alarms/hour**, inside the band. The first
~500 shots run hotter (31.7 %) because the line starts from a worn die (§2.1) — a real high-risk
regime, and the budget is a long-run design target rather than a per-window cap.

### 4.8 Prescription

`defectlab prescribe --seed 7`, riskiest of 600 shots. Three moves, each capped at its ramp limit:
`fast_shot` +0.4 m/s, `slow_shot` +0.06 m/s, `pour_temp` +15 °C. Risk **1.0000 → 0.0174**, margin
**+16.6 logits**. Stability **1.000** under ±20/35/50 % weight perturbation.

**Quote that stability with its reason.** It is not vacuous — every recommendation worsens at
least one mechanism (worst → gas porosity, p99 → shrinkage, median → flash), so a different
weighting could in principle flip it. It passes because the improvement dominates the worsening by
roughly 100× and ±50 % is at most a 3× swing. The claim is *"for shots this far from nominal the
advice does not depend on the weights"*, not *"the advice is verified"*.

### 4.9 Monitoring

Building the SPC page on the evaluation set signalled on **48 % of points**, all artefact: that
set is grouped by label (lag-1 label autocorrelation 0.997, one run of 453 identical labels), so
"nine points on one side" was reading row order rather than the process. The fix was a schema
change — `fact_shot` carries **no timestamp**, and a separate `fact_production` holds a genuine
contiguous run.

Signal rate then fell 34.8 % → **8.3 %** (~5 alarms/hour) through three further corrections, in
order of size:

1. **Nelson run rules were not edge-triggered**, despite the docstring claiming they were. One
   sustained 30-point shift raised 22 alarms. The existing test passed only because its fixture
   was exactly the nine-point window.
2. **The chart was drawn on the probability**, which has skew +4.25 and excess kurtosis +18.9 and
   fails the Shewhart normality assumption outright. On the logit: +1.02 and +1.19.
3. An AR(1) residual chart, which barely helped — and thereby identified the structure as a
   lot-level level shift (§4.5).

### 4.10 System verification

| check | result |
|---|---|
| test suite | 308 passing, ruff clean, 2 import-linter contracts kept |
| API through nginx | `/health` ok, 3 SSE frames, UI 200 on :8080 |
| MQTT, 25 s window | 26 telemetry, 25 verdicts, audit hashes present |
| `docker kill` the line container | broker published `offline` in **2.1 s** |
| offline bundle | stack up from `images.tar` after `prune -af` reclaimed 9.875 GB |

The `docker kill` row is the one worth demonstrating: pulling the plug on the machine container
and having the *broker* announce it, with no consumer-side timeout involved. It is the behaviour
that justifies MQTT rather than a second SSE feed.

---

## 5. Limitations

Written candidly, because a defensive limitations section signals the opposite of what it intends.

**The process channel is simulated.** This is the central limitation and no amount of engineering
around it changes that. The twin encodes plausible HPDC physics with named mechanisms, but the
fusion gain is a gain over *this* process channel. Real telemetry would have sensor drift,
missing values, unmodelled couplings and correlations with the image channel that the twin cannot
have. The honest claim is about the *structure* of the result — that the benefit tracks camera
headroom — not about its magnitude on a real line.

**The images and labels do not vary across twin seeds.** Vision sd is exactly 0.0000 (§4.4). All
the reported variance is process-side, so the confidence intervals describe uncertainty in the
fusion model, not in the vision baseline.

**The escape multiplier is a guess.** M = 25× is a literature convention (the 1-10-100 rule), not
a measurement from this line. Everything downstream of it is reported as a range, and the
headline saving is quoted against 100 % inspection precisely because that comparison does not
lean on it.

**One anomaly is unexplained** (§4.3). The resolution-boundary account is a hypothesis and is
labelled as one.

**The backbone replication is underpowered.** p = 0.056 on five shared seeds. It supports the
headroom account; it does not establish it.

**Effective sample size is small.** Lots, not parts (§4.5). Every interval in this report should
be read against that.

**The audit chain proves integrity and ordering, not authenticity.** Anyone who can append can
recompute it from genesis.

**No operator identity.** The override records what was decided, why, and the explanation shown at
the time — but not by whom. A quality record with no signatory is incomplete.

**No TLS, no broker auth.** `allow_anonymous true`. Fine on localhost; the difference between this
and something facing a plant network.

**The SPC engine has not been validated against R `qcc` fixtures.** Planned in the execution plan
and not done. The X-bar limits are checked against tabulated A2 factors, which is weaker.

---

## 6. Conclusion

Fusing process telemetry with inline vision helps in proportion to how much the camera has lost:
**+0.0303 AUC per unit of degradation, 15/15 seeds positive**, and nothing at all when the camera
is clean. Reported across the whole ladder rather than at the flattering point, replicated in
direction on a second backbone, and priced on the line rather than the test set.

The economic layer sharpens it into something a plant could act on. The fusion gate saves against
100 % inspection across the whole plausible range of escape costs. The process channel alone,
held to an alarm rate an operator can actually work, is **worse than inspecting everything** —
which is the clearest statement of what the image channel is buying, and it is a negative result
about the half of the system this project built from scratch.

---

## Appendix: reproducing this

```
uv sync --extra ml --extra viz
defectlab simulate --root data/raw/casting_data
defectlab extract  --backbone resnet18 --regime both
defectlab ablate   --seeds 1,2,...,15 --severities 0.5,1,1.5,2,3
defectlab figures  --results results/ablation_resnet18.csv
defectlab economics --severity 2
defectlab prescribe --seed 7
defectlab export
docker compose up --build            # the running system
python deploy/bundle.py              # the offline demo
```

Every figure in this report is regenerated by `defectlab figures`. Every number is produced by
one of the commands above; none is transcribed by hand from a notebook.
