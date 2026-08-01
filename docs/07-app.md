# The operator app

React 19 + TypeScript + Vite, Mantine for chrome, uPlot for the streaming chart. It talks to
`defectlab serve` over REST and SSE and holds no state the server does not.

## Running it

```
./.venv/Scripts/python.exe -m uvicorn --factory defectlab.api.app:create_app --port 8000
cd app && npm install && npm run dev        # http://localhost:5173
```

The API host is `VITE_API`, defaulting to `http://127.0.0.1:8000`. CORS allows only the Vite dev
origins; a production build served from the same origin needs none.

## Layout

| File | Job |
|---|---|
| `api/client.ts` | typed fetch wrappers; unwraps the 422 detail so the twin's own message reaches the user |
| `api/useStream.ts` | SSE subscription with a capped buffer |
| `api/useReference.ts` | machine limits, reason codes, model version — fetched once |
| `api/useInspection.ts` | score + explain + prescribe for the selected shot, fired together |
| `panels/RiskChart.tsx` | uPlot, created once and fed with `setData` |
| `panels/LineView.tsx` | chart, alarm rate, shot list |
| `panels/ShotInspector.tsx` | why / what to change / readings / disagree |
| `panels/Sandbox.tsx` | what-if sliders bounded by `/parameters` |

## Decisions worth defending

- **The chart plots the logit.** On a probability axis every alarm pins to the top and a real
  20-logit improvement is invisible, because a risky shot sits where the sigmoid is flat. Same
  trap as the SPC risk chart and the prescribe search.
- **The prescription headline is the margin, not the probability drop.** Measured live: the
  riskiest shot goes 1.0000 to 0.99999999999 while gaining **20.8 logits**. Rendering that as
  "100 % to 100 %" beside a large gain looks like a bug, so the saturated case says what is
  happening rather than hiding it.
- **The alarm rate is shown in alarms/hour, not percent.** ISA-18.2 is written in those units and
  an operator can say yes or no to a count per hour.
- **`/parameters` serves the machine limits.** The sandbox cannot offer a setpoint the machine
  cannot reach, and the physics is not copied into the frontend where it would drift.
- **Lot-level and maintenance parameters are shown but locked.** Nobody changes alloy chemistry
  mid-shift; a sandbox that pretends otherwise teaches the wrong lesson.
- **The stream does not audit; selecting a shot does.** A shot nobody looked at is not a decision
  anyone made. Clicking one re-scores it through `/score`, which writes the audit entry and
  returns the hash an override is signed against.
- **The buffer is capped at 240 shots.** A live view is a window, not an archive. The archive is
  Power BI.
- **An empty anchor is reported as "no rule needed", not as a failure.** At the served threshold
  most of the line passes, so the empty rule already beats the precision target. Inventing
  conditions to fill the panel would be worse than saying so.

## The override, and why it carries its own explanation

Gate 3 requires an override to record the model version *and the explanation shown at the time*.
`explanation_shown` therefore comes from the client, because only the client knows what was
rendered. Re-deriving it server-side would log the explanation the model gives **now**, which is
the one thing an audit of a past decision must not do.

That makes it an attestation rather than a verified fact, and the docstring on `OverrideRequest`
says so. Reason codes come from `/reasons` so the vocabulary exists once; `other` demands a note,
and a rising share of `other` is the signal that the vocabulary needs extending.

An override against an `audit_hash` the log does not contain is refused with 404 — otherwise the
chain would record a dissent from a call nobody can show was taken.

## Two things a reader will notice

- **`Predicate.upper` arrives as `null`, not `Infinity`.** JSON has no infinity. Anchors have
  genuinely one-sided predicates, so `bound()` renders "at least 3.09" rather than a broken
  interval. The TypeScript type says `number | null` because saying `number` would be a lie that
  happened to render correctly.
- **The conformal set and the gate can disagree** — a shot at risk 0.10 can carry
  `prediction_set: [1]` while the gate passes it. They answer different questions: the Mondrian
  set is a class-conditional coverage guarantee, which is exactly why it is invariant to the
  prior shift, while the gate is a cost decision at the deployment prior. The set is therefore
  shown as an abstention state, never as a competing verdict.

## Not done

Operator identity. The override records what was decided and why, not by whom; there is no auth
layer. Say that before an examiner does — a quality record with no signatory is an incomplete
one, and adding real identity is a deployment concern rather than a modelling one.
