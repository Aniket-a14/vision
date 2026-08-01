# State of play

Written 2026-08-01. Read this first when resuming; it is the only file that needs to be current.

## Verify the build in one command

```
./.venv/Scripts/python.exe -m pytest -q      # 166 tests
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
| `prescribe` | **not started** | — |
| `api` | **not started** | — |

Not started beyond the package: React app, MQTT simulator, deploy, offline bundle, report,
slides, and the **Power BI `.pbix`, which is a hard rubric requirement still at zero**.

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
- **Taguchi: read `baseline_ratio`, never `mean_loss`.** With Δ₀ = 3σ a parameter drawn at its
  own spread costs A₀/9 whatever A₀ is, so ten parameters sum past the value of the part. The
  ratio divides that artefact out: `pour_temp_c` 2.64 (real drift), four params at 0.98–1.02
  (the control), `die_temp_c` 0.49 (thermal inertia), chemistry 0.43–0.67 (lot-level).

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

1. **`prescribe`** — recommender fitted on `sample_uniform_envelope` (randomised interventional
   data, already in `twin/parameters.py`), plus a simulator-perturbation robustness test.
   Respect `Actionability`: an operator cannot change `si_content_pct` mid-shift.
2. **Power BI `.pbix`** — hard rubric requirement, still at zero. Needs exports from
   `explain`, `economics` and `spc`, which all now exist. **Do this before the API if time
   gets short**; the API is impressive but not marked, and the `.pbix` is.
3. **`api`** — FastAPI + SSE, MQTT line simulator, hash-chained audit log.
4. React app, deploy, offline bundle, report, slides.

A note on ordering, learned the hard way: breadth-first beats depth-first here. Adding seeds to
an already-significant result optimises the thing most recently looked at, while a
rubric-mandated deliverable sits at zero.
