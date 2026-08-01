# Power BI

## Build it

```
./.venv/Scripts/python.exe -m defectlab.cli.main export --severity 2
```

That writes `data/exports/*.csv` plus `powerbi/DefectLab.pbip`. Then:

1. Open `powerbi/DefectLab.pbip` in Power BI Desktop.
2. If the CSVs are not at the baked-in path, change the **ExportFolder** parameter
   (Transform data → Manage parameters) and refresh. The path is a parameter precisely so the
   model moves between machines without editing nine queries.
3. Lay out the four pages (below).
4. **File → Save As → `.pbix`.**

The `.pbix` is a binary container only Desktop can author, which is why the repo holds the PBIP
text form. `*.pbix` is gitignored: it is a build artefact, not a source file.

## What is already done for you

The whole semantic model is generated from `export/schema.py`, so it cannot drift from the data:

- **9 tables**, every column typed (no "everything is text" import).
- **5 relationships**, each checked by a test against the export contract.
- **`dim_date`** marked for time intelligence, joined via a `date` calculated column on
  `fact_production` — a timestamp will not join to a date table at day grain.
- **15 DAX measures** with format strings, listed in `powerbi.MEASURES`.

The report has four named but empty pages. Visual JSON written blind is how a PBIP ends up
refusing to open, and laying visuals out is the one part Desktop is genuinely good at.

## The two grains — read this before building a visual

| Table | Use it for | Never use it for |
|---|---|---|
| `fact_shot` | model quality: escape rate, overkill rate, confusion outcomes, attribution | anything over time |
| `fact_production` | anything with a clock: trends, shift comparisons, control charts | model accuracy |

`fact_shot` is the held-out evaluation set. It is oversampled and grouped by label, so it has no
time axis and deliberately carries **no timestamp column**. Putting it on a time axis produced a
control chart signalling on 48 % of points, every one an artefact of row order.

## The four pages

**1. Line overview** — `fact_production`.
Cards: `Production Shots`, `Alert Rate`, `Alarms per Hour`. Line chart of `risk` by `timestamp`.
Stacked column of shot counts by `dim_date[shift_label]`. Slicer on `lot_id`.

**2. Why this part** — `fact_shot` + `fact_attribution`.
Bar chart of `SUM(contribution)` by `dim_group[group]` — the grouped SHAP story. Table of shots
with `risk`, `outcome`, `dominant_mechanism`. Slicer on `outcome` so an examiner can jump
straight to the escapes. Add `dim_parameter` as a slicer on `is_lever` to show what is
actionable.

**3. Cost of quality** — `fact_cost_curve`.
Line chart of `per_shot` by `threshold`, with `escape_rate` and `overkill_rate` on a secondary
axis — the scissors plot. Slicer on `threshold` driving the `Cost per Shot`,
`COPQ per 1000 Shots` and `Selected Threshold` cards. This is the interactive what-if the
rubric asks for.

**4. Process control** — `fact_spc` + `fact_production`.
Line chart of `value` by `shot_id` with `centre`, `lower` and `upper` as three more series — the
limits are repeated on every row precisely so this needs no join. Colour points by `signal`.
Bar chart of signal counts by `dim_rule[description]`. Cards: `SPC Signals`, `SPC Signal Rate`.

> The chart is drawn on `risk_residual`, not on the raw score. The raw probability has skew
> +4.25 and fails the Shewhart normality assumption; see `docs/04-execution-plan.md` for why the
> signal rate fell from 34.8 % to 8.3 %. Label the axis "risk (log-odds, AR(1) residual)" so the
> units are not mistaken for probability.

## If a refresh fails

- **"We couldn't find file"** — the `ExportFolder` parameter is stale. Re-point and refresh.
- **A rule slicer shows a blank member** — expected. Most `fact_spc` rows have no rule because
  they did not signal.
- **A measure returns blank** — check the slicer context; `Selected Threshold` needs exactly one
  threshold selected.
