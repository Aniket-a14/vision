# Viva deck

14 slides, ~15 minutes, leaving time for the live demo. Written as content plus speaker notes so
it drops into PowerPoint, Google Slides or reveal.js unchanged.

The deck is built around one argument and refuses to bury it: **the process channel helps in
proportion to how much the camera has lost, and on its own it is not enough.** Every slide either
advances that or is cut.

---

## 1 — Title

> **Multi-modal casting defect prediction**
> A physics-grounded digital twin fused with inline vision
>
> *[name] · [supervisor] · [date]*

**Notes.** Do not open with the architecture. Open with the question on slide 2.

---

## 2 — The question

> Casting defect datasets are **images**.
> Casting physics is **process telemetry**.
> They are almost never available together.
>
> **So: does the process channel earn its instrumentation cost?**

**Notes.** This is the gap. The literature optimises one channel or the other because nobody has
both, so the question does not get asked. Say the question out loud and then answer it in one
sentence: *it depends entirely on how good your camera is, and I can show you the slope.*

---

## 3 — Approach

> A physics twin generates telemetry with **known causal structure**.
> Each simulated shot is paired with a **real** Kaggle casting image of the matching label.
>
> Twin = ground truth for mechanism. Images = ground truth for appearance.
> Varying them independently is what makes the experiment possible.

**Notes.** Expect "why not real telemetry" immediately — answer it here rather than defending
later. Nobody publishes paired process-and-image casting data; if they did, this project would
not exist. The limitation is real and it is on slide 12.

---

## 4 — The headline

> **Fusion gain grows as the camera degrades.**
>
> | severity | vision | gain | seeds + |
> |---|---|---|---|
> | 0.5 | 0.9985 | +0.0002 | 9/15 |
> | 1.0 | 0.9847 | +0.0051 | 14/15 |
> | 1.5 | 0.9671 | +0.0031 | 8/15 |
> | 2.0 | 0.8984 | +0.0352 | 14/15 |
> | 3.0 | 0.8374 | +0.0727 | **15/15** |
>
> **+0.0303 per unit severity · t = 9.39 · p < 1e-5 · 15/15 seeds**

**Notes.** Land two things. First: the whole ladder is shown, and it was fixed before any result
was seen. Second: an earlier five-seed run reported a **null**, and that null was correct — it was
measured at severity 1.0, the one point where vision has no headroom to lose. That story is worth
30 seconds; it is the difference between a result and a finding.

At severity 3 the camera costs vision 0.161 AUC and fusion recovers 45 % of it, on every seed.

---

## 5 — Replication, reported honestly

> | backbone | AUC lost | slope | p | seeds + |
> |---|---|---|---|---|
> | ResNet-18 | 0.161 | +0.0258 | **0.028** | 5/5 |
> | DINOv2-S | 0.084 | +0.0076 | 0.140 | 4/5 |
>
> **Direction replicates. Significance does not.**
> Headroom, not architecture: DINOv2 loses half as much, so there is half as much to recover.
> Seed-level test of that relationship: **p = 0.056**.

**Notes.** Say explicitly: I quote 0.056, not the pooled 0.0046, because both sweeps reuse the
same five twin seeds and share a process channel and label vector — (backbone, seed) is not an
independent unit. Volunteering this is worth more than the result itself.

---

## 6 — One thing I cannot explain

> Severity **1.5** gives a *smaller* gain (+0.0031) than severity 1.0 (+0.0051),
> despite worse imaging. 8/15 seeds, p = 0.291, spread 3× wider.
>
> Survived tripling the seed count. **Not noise.**
> Resolution-boundary account is a **hypothesis**, not a finding.

**Notes.** Put this on a slide of its own. An examiner who finds an unexplained dip you have
hidden will spend the rest of the viva there; one you volunteer costs you a slide. Vision degrades
smoothly through that point, so the dip is in the fusion model, not the camera.

---

## 7 — Pricing it on the line, not the test set

> Corrected 57 % → 3 % prevalence (Elkan odds rescaling), per 1,000 shots:
>
> | policy | cost/shot |
> |---|---|
> | **gate** | **€1.40** |
> | inspect everything | €3.36 |
> | ship everything | €9.00 |
>
> Saving vs 100 % inspection: **€1,613–2,289** across M = 10–50×.

**Notes.** Explain why the *smaller* number is quoted. The saving against ship-everything is
€2,529–16,253 and it is mostly a restatement of the escape multiplier, which was guessed. The
saving against inspection is positive across the whole range and does not lean on the guess.

Then the free result: alert rate 16.5 % = **9.9 alarms/hour**, inside the ISA-18.2 band of 6–12,
without that constraint being imposed.

---

## 8 — The negative result

> Process channel **alone**, held to an alarm rate an operator can work:
>
> | threshold | cost/shot | escape rate | alert rate |
> |---|---|---|---|
> | cost optimum | €2.50 | 0.074 | 0.485 |
> | **budgeted** | **€6.12** | 0.654 | 0.022 |
>
> **€6.12 > €3.36.** Worse than inspecting everything.

**Notes.** This is the most important slide in the deck. It is a negative result about the half of
the system I built from scratch, and it is the clearest possible statement of what the image
channel buys.

The mechanism: process-only scores bunch at mean 0.54 on a 57 %-defective set — the channel barely
separates — so with an escape at 100× an inspection the unconstrained optimum wants to inspect
half the line. Economically correct, operationally unusable.

Found by instrumenting the demo, not by the study.

---

## 9 — What the operator sees

*[screenshot: live line + inspector]*

> Risk on the **logit** axis · alarms/hour, not percent
> Anchor rule · ramp-limited advice · override with reason codes

**Notes.** One sentence on the logit axis, because it recurs: a risky shot sits where the sigmoid
is flat, so on a probability axis every alarm pins to the top and a real 16-logit improvement is
invisible. The same trap appeared four separate times — SHAP attribution, the prescribe search,
the robustness measure and the SPC chart.

Advice is capped at each parameter's ramp limit: one shift's adjustment, not a re-qualification.

---

## 10 — Live demo

> `docker compose up` → **localhost:8080**
>
> 1. Line running, alarms/hour inside the band
> 2. Click a flagged shot → rule, advice, prediction set
> 3. Override with a reason code → audit hash
> 4. **`docker kill` the line container** → broker announces `offline` in **2.1 s**

**Notes.** Step 4 is the one to rehearse. Pulling the plug on the machine container and having the
*broker* announce it — no consumer-side timeout — is the whole reason MQTT is there rather than a
second SSE feed. It is also the only step that can fail live, so have the screenshot ready.

The bundle runs with no network: verified by deleting every image and layer (`prune -af` reclaimed
9.875 GB) and bringing the stack up from the tar alone.

---

## 11 — Engineering, briefly

> 308 tests · ruff clean · 2 architecture contracts enforced
> Hash-chained audit · conformal abstention · frozen SPC limits
> Star-schema export → generated Power BI model
> Image 2.01 GB → **1.02 GB** (`xgboost-cpu`: the Linux wheel pulls 289 MB of CUDA)

**Notes.** Do not dwell. One line: the tests that mattered were the ones that failed for real
reasons — a Nelson rule that was not edge-triggered and raised 22 alarms for one shift, and an
import chain that made scoring process telemetry require OpenCV.

If asked about the audit: **it proves integrity and ordering, not authenticity.** Anyone who can
append can recompute it from genesis. Say that before they do.

---

## 12 — Limitations

> **The process channel is simulated.** The claim is about the *structure* of the result —
> the benefit tracks camera headroom — not its magnitude on a real line.
>
> Vision sd is exactly 0.0000: images and labels do not vary across seeds.
> M = 25× is a convention, not a measurement.
> Effective sample size is **lots, not parts**.
> Backbone replication is underpowered (p = 0.056, five shared seeds).
> No operator identity, no TLS, no broker auth.
> SPC not validated against R `qcc`.

**Notes.** Deliver this without hedging and without apologising. Lead with the one that actually
matters. A defensive limitations section signals the opposite of what it intends.

---

## 13 — The methodological finding

> **The effective sample size is the number of alloy lots, not the number of parts.**
>
> Chemistry and die state vary by lot. Shots within a lot are correlated.
> 7,348 parts are *not* 7,348 independent observations.
>
> It showed up twice, independently: in the fusion statistics, and again in SPC —
> autocorrelation 0.23 at lag 1, still 0.16 at lag 20. Not AR(1) memory. A level shift between lots.

**Notes.** This is the slide a good examiner will most respect, so give it its own place near the
end rather than burying it in method. The SPC recurrence is the strong part: the same structure
appeared unprompted in a different layer, and an AR(1) residual chart was fitted and *barely
helped*, which is how it was identified.

---

## 14 — Conclusion

> Fusion helps **in proportion to what the camera has lost**:
> +0.0303 AUC per unit degradation, 15/15 seeds — and nothing at all when the camera is clean.
>
> The fusion gate saves against 100 % inspection across the whole plausible range of escape costs.
> The process channel alone, at a workable alarm rate, is **worse than inspecting everything**.

**Notes.** End on the contrast, not on a summary of what was built. The system is the evidence,
not the contribution.

---

## Questions to have an answer ready for

| Question | Answer |
|---|---|
| Why not real telemetry? | Nobody publishes paired process-and-image casting data. Limitation, slide 12. |
| Why does severity 1.5 dip? | I don't know. Survived 15 seeds. Hypothesis is a resolution boundary. |
| Is the fusion gain practically significant? | +0.073 AUC at severity 3 is; +0.0002 at severity 0.5 is not. That's the finding. |
| Why is vision sd zero? | Images, order and labels are identical across twin seeds. All paired variance is process-side, which is why the paired test is right. |
| Isn't M = 25 arbitrary? | Yes. Reported as a range, and the headline saving is quoted against inspection so it doesn't depend on M. |
| Could you fake the audit log? | Yes, if you can append. It proves integrity and ordering, not authenticity. Real tamper-evidence needs the head published externally. |
| Why MQTT and not just SSE? | The last will. A crashed cell announces itself; an HTTP stream that stops just stops. |
| Why is your threshold 0.21 and not 0.5? | An escape costs ~100× an inspection, so the cost optimum sits far below 0.5. It is then raised to fit the ISA-18.2 alarm budget. |
