# Physics Twin — Corrections to the Original Build Guide

**Status:** derived from deep research, 2026-07-31. Supersedes §4.2 of `multimodal_defect_prediction_build_guide.md`.
**Why this document exists:** the original `defect_logit()` encodes six physics assumptions that are wrong or unsupported. Since every downstream claim (SHAP recovers the physics, recommendations verify against ground truth, COPQ) inherits the simulator's assumptions, a wrong simulator produces a *confidently wrong* system. These are the fixes.

---

## C1 — The iron threshold is silicon-dependent, and 0.9 % is too high

**Original:** `0.55 * relu((fe_content_pct - 0.9) / 0.15)` — a fixed threshold at 0.9 %.

**Correct:** the critical iron content follows Taylor's relation:

```
Fe_crit ≈ 0.075 × (wt% Si) − 0.05
```

| Alloy | Si | Fe_crit |
|---|---|---|
| A380 | 8.5–9.5 % | **0.59–0.66 %** |
| ADC12 | 9.6–12 % | **0.67–0.85 %** |

At the guide's `si_content_pct ~ N(9.5, 0.7)`, Fe_crit ≈ **0.66 %**, not 0.9 %. The threshold must be **computed from the sampled Si of the same row**, which also creates a genuine, physically-real Si×Fe interaction the model can discover.

**Mechanism matters too.** Above Fe_crit, β-Al₅FeSi platelets form *before* the Al-Si eutectic and **block interdendritic feeding channels → shrinkage porosity**. So Fe is a porosity driver, not merely a ductility/toughness driver. The original guide's comment ("Fe intermetallics") understates the causal path.

## C2 — Iron has a lower bound; the effect is two-sided

**Original:** one-sided — less Fe is always better.

**Correct:** conventional die-cast alloys deliberately hold Fe at **0.7–1.1 %** to suppress **die soldering** (Fe-Al intermetallic bonding of casting to H13 die steel). Only structural alloys (Silafont-36, Aural, EZCast) run Fe < 0.15–0.25 %, and they require vacuum + Mn/Sr countermeasures to compensate.

A simulator that says "less Fe = better" will produce a recommender that prescribes chemistry which destroys the die. **Model Fe as U-shaped**: soldering penalty below ~0.7 %, β-platelet porosity penalty above Fe_crit(Si).

## C3 — Missing: the sludge constraint

Not in the original at all. Sludge factor:

```
SF = 1×%Fe + 2×%Mn + 3×%Cr        keep SF < 1.8   (at ~650 °C holding)
```

Temperature-dependent form for critical sludge deposition at holding temperature T °C:

```
%Fe + {3.34 − (T−630)/714}·%Mn = 2.39 + (T−630)/152
```

Sludge particles are hard primary intermetallics causing **both scrap and tool wear**. Without this term, any optimiser is free to raise Fe/Mn without penalty and will wreck the holding furnace. Adding Mn as a tenth feature makes this real and gives the constrained optimiser something meaningful to respect.

## C4 — The dominant factors are the plunger velocities, not pour temperature

**Original:** `0.95 * t²` on pour temperature is the largest single coefficient — pour temp is the star of the model.

**Correct:** a published L25 Taguchi ANOVA on ADC12 HPDC porosity attributes:

| Factor | Contribution to porosity |
|---|---|
| **1st-phase (slow-shot) plunger velocity** | **34.0 %** |
| **2nd-phase (fast-shot) plunger velocity** | **31.6 %** |
| Temperature and pressure | the remaining ~34 % between them |

Velocities jointly account for **~66 %**. The guide's single lumped `inj_speed_ms` at coefficient 0.50 both under-weights velocity and wrongly collapses two physically distinct phases into one.

**Fix:** split `inj_speed_ms` into `slow_shot_velocity_ms` (0.15–0.5 m/s, optimum ≈ 0.26–0.30) and `fast_shot_velocity_ms` (2.6–3.0 m/s), and re-weight so the velocities dominate. Each is **non-monotonic with its own critical-velocity optimum**:

- Slow shot **too low** → the wave fails to seal the shot sleeve → air entrapment. **Too high** → wave breaking → air entrapment.
- Fast shot **too low** → cold shut / misrun. **Too high** → turbulent fill → air entrapment.

> ⚠️ *Source caveat:* the ANOVA percentages come from a search index; the underlying journal paper was not directly retrievable, and the same source reports injection pressure in `kg/m³`, which is certainly a typo for `kg/cm²`. Cite the primary paper or state the figures as indicative.

## C5 — Intensification pressure saturates

**Original:** `0.80 * relu(-p)` — monotone, unbounded benefit from higher pressure.

**Correct:** the benefit of intensification pressure **saturates around ~67 MPa** in an A380 study, and beyond the knee you incur flash and die stress. A monotone term means any optimiser pushes pressure to the box maximum on every single recommendation — a tell-tale sign of a naive model that a viva examiner will spot immediately.

**Fix:** saturating benefit (e.g. `relu(-p)` capped, or a `tanh`/logistic form) **plus** a rising flash/die-stress penalty above the knee. Also note the mechanism limit: intensification **feeds shrinkage and compresses entrapped air — it cannot remove dissolved-hydrogen gas porosity.** These are different defect families and should not share one lever.

## C6 — Cycle/cooling time is a mediator, not a lever

**Original:** `0.35 * relu(-c)` wires cooling time directly to defect probability.

**Correct:** cooling/cycle time acts *through* die thermal state. Wiring it directly means the recommender will prescribe cycle-time changes as a primary intervention, when the actual lever is the thermolator/spray setting. This is the classic confounded-proxy error: in observational plant data, cycle time correlates strongly with porosity while being the wrong knob.

**Fix:** make `die_temp_c` a *function* of cooling time, spray time and accumulated thermal state, and let defect probability depend on `die_temp_c`. This is also exactly what the stateful line dynamics need (see §Line dynamics below).

---

## Additional physics to encode

| Quantity | Value | Confidence |
|---|---|---|
| **H₂ solubility in liquid Al @ 660 °C** | 0.69 mL/100 g | HIGH |
| **H₂ solubility in solid Al @ 660 °C** | 0.036 mL/100 g | HIGH |
| **Solubility drop at solidification** | **~19–20×** — *this cliff is the gas-porosity mechanism* | HIGH |
| Target dissolved H (structural) | < 0.10–0.15 mL/100 g | MED-HIGH |
| Melt/holding temperature | 660–700 °C (Taguchi range); 700–730 °C general | HIGH |
| Die surface temperature | 180–280 °C by wall thickness; **±5 °C uniformity** | MEDIUM |
| Local cold-spot sensitivity | a spot **30 °C** below average can trigger a downstream cold shut | MEDIUM |
| Runner→gate melt loss | 50–120 °C | MEDIUM |
| Intensification pressure | 60–120 MPa; saturates ~67 MPa | MED-HIGH |
| Typical HPDC porosity | ~0.5 vol% | MEDIUM |
| Pressure-tight spec | < 1 % porosity, max pore Ø 0.2 mm | MEDIUM |
| Leak-test acceptance | ≤ 1×10⁻³ mbar·L/s | MEDIUM |
| Cpk targets | ≥ 1.33 general, ≥ 1.67 special characteristics | HIGH |

**Die soldering:** there is **no universal numeric threshold** (the "480 °C" figure that circulates does not exist as a constant — the critical temperature is the solidus of the local Fe-Al reaction product, which is alloy- and die-steel-specific). Model soldering as an **accumulating thermal-exposure integral** over die surface temperature, not a step function. This also gives tool wear a physically-grounded accumulation law rather than the original guide's arbitrary `(shots/50000)^1.5`.

**Misrun ≠ cold shut.** Distinct defects, label them separately:
- **Misrun** — metal freezes before the cavity fills → missing features.
- **Cold shut** — cavity fills, but two fronts are too cold to fuse → seam.

Both are driven by fill time, gate velocity, die temperature uniformity and superheat. **Final pressure is nearly irrelevant to them** — another reason the pressure term must not be a universal fix-all.

---

## Line dynamics (stateful twin)

Required for the live demo to be meaningful rather than decorative. The twin simulates a *running line*, not IID draws:

- **Die thermal state** — integrates shot-to-shot: heats with each shot, cools with spray/idle. Drives `die_temp_c`, which drives misrun/cold-shut risk. Explains the classic start-of-shift scrap burst.
- **Tool wear** — accumulates by shot count *and* by thermal-exposure integral (C-above). Drives flash and dimensional drift.
- **Alloy lots** — Si/Fe/Mn are **lot-level**, not shot-level. A new furnace charge steps the chemistry. ⚠️ *This is a genuine leakage hazard*: joining lot-level chemistry onto shot-level rows creates grouped structure, so **splits must be by lot, not random**, or the model learns the lot rather than the physics.
- **Sensor drift** — thermocouple drift, pressure-transducer zero drift. Gives the drift monitor something real to catch.
- **Shift effects** — operator changeover, breaks, restart transients.

## Guard rails that must remain from the original guide

The original guide's most important instruction is still correct and non-negotiable:

> **Never generate process parameters from the label.** Causal order is always
> `parameters → defect probability → sampled label → matched image`.

Keep the leakage checks (no single feature correlating > ~0.35 with the label; U-shape visible in the pour-temp histogram split by label). Add one: **split by alloy lot and by die, never randomly.**

---

## The circularity problem — and how to answer it

The original guide validates recommendations by feeding them back into `defect_logit()`. **This is circular**: it proves the recommender successfully inverted the simulator, not that the recommendation is correct. An examiner will say exactly this.

**The fix is cheap and it is the single highest-value robustness test available:**

1. **Simulator-perturbation test.** Perturb the simulator's coefficients by ±20–50 %, add unmodelled noise, shift to an out-of-distribution operating point, then re-run the recommendations. **The *sign* of each recommendation must stay stable.** If flipping one coefficient flips the advice, the recommendation is an artefact.
2. Report the fraction of recommendations whose direction survives perturbation — a far stronger number than "97 % verify against the model that generated them".
3. State plainly in the methodology that stage-0 simulator inversion is *necessary but insufficient*, and that stage-1 perturbation robustness is what is actually being claimed.

Doing this converts the project's biggest methodological weakness into a demonstration of methodological maturity.
