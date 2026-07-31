# DefectLab

Multi-modal casting defect prediction: a physics-based process twin fused with real industrial
imagery, under realistic inline-camera degradation.

## What this is

A quality-control system for Al-Si high-pressure die casting that predicts impeller defects by
fusing two channels available at different moments in the production cycle:

| Channel | Available | Source |
|---|---|---|
| Process parameters | before and during the shot | physics-based digital twin |
| Part image | after the shot | real Kaggle casting dataset |

**Claim boundary.** The two channels are coupled only through the shared defect label. No
cross-modal interaction is claimed to be physical. The research question is therefore *how much
predictive performance an uncorrelated process channel recovers as image quality degrades* — a
question this design answers validly, because the channels' independence is what the coupling
scheme guarantees. See [docs/03-architecture-and-stack.md](docs/03-architecture-and-stack.md).

## Quick start

```bash
uv sync --extra ml --extra viz
uv run pytest tests -q
docker compose up -d
```

## Design documents

| Document | Contents |
|---|---|
| [01-physics-twin-corrections.md](docs/01-physics-twin-corrections.md) | The six physics corrections the twin implements |
| [02-decision-layer.md](docs/02-decision-layer.md) | Explainability, prescription, thresholds, COPQ, SPC |
| [03-architecture-and-stack.md](docs/03-architecture-and-stack.md) | Architecture, model choices, full tech stack |
| [04-execution-plan.md](docs/04-execution-plan.md) | Four-week plan, gates, de-scope ladder |

## Architecture rules

Two contracts are enforced in CI by `import-linter`:

1. **`twin/` never imports from `models/` or `data/`.** The simulator is ground truth; models are
   consumers. If that dependency inverts, the project has label leakage.
2. **Layered imports** — `api → prescribe → explain → models → imaging → data → twin → config`.

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
└── data/              dataset discovery, twin-to-image pairing, contracts, splits
```
