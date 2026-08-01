"""One function per CLI command. Each delegates to the library and prints a summary."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from ..data import build as build_data
from ..data.images import verify_counts
from ..imaging import Regime
from ..imaging.degrade import InlineCamera
from ..imaging.features import cache_path, extract_cached
from ..twin import FEATURES, TwinConfig, run_line, score

LOG = logging.getLogger("defectlab.cli")


def verify(args: argparse.Namespace) -> int:
    """Check the extracted Kaggle dataset against the published folder counts."""
    counts = verify_counts(Path(args.root), strict=not args.lenient)
    for (split, folder), count in sorted(counts.items()):
        print(f"{split:<6} {folder:<10} {count}")
    print(f"total {sum(counts.values())}")
    return 0


def simulate(args: argparse.Namespace) -> int:
    """Run the twin and pair each shot with a real image of the matching label."""
    config = TwinConfig(seed=args.seed, noise_sd=args.noise_sd, signal_gain=args.signal_gain)
    dataset = build_data.build(Path(args.root), config, oversample=args.oversample)
    written = build_data.write(dataset, Path(args.out))
    print(dataset.summary().to_string(index=False))
    for name, path in written.items():
        print(f"wrote {name}: {path}")
    return 0


def extract(args: argparse.Namespace) -> int:
    """Extract and cache image embeddings for one backbone and regime."""
    processed = Path(args.processed)
    for split in build_data.SPLITS:
        frame = pd.read_parquet(processed / f"{split}_paired.parquet")
        for regime in _regimes(args.regime):
            for severity in _severities(args, regime):
                destination = cache_path(processed, args.backbone, split, regime, severity)
                features = extract_cached(
                    build_data.image_paths(frame),
                    regime,
                    args.backbone,
                    destination,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    camera=InlineCamera(severity=severity),
                )
                print(f"{split:<6} {regime.value:<7} {features.shape} -> {destination.name}")
    return 0


def _severities(args: argparse.Namespace, regime: Regime) -> tuple[float, ...]:
    """The lab regime is undegraded, so sweeping severity there would repeat one pass."""
    if regime is Regime.LAB:
        return (1.0,)
    return tuple(float(value) for value in args.severities.split(","))


def ablate(args: argparse.Namespace) -> int:
    """Run the 3x2 ablation across twin seeds; one seed cannot estimate the spread."""
    from ..models.ablation import fusion_gain, summarise

    seeds = [int(value) for value in args.seeds.split(",")]
    severities = [float(value) for value in args.severities.split(",")]
    clock = _Clock(len(seeds) * len(severities))
    frames = [_ablate_seed(seed, severities, args, clock) for seed in seeds]
    results = pd.concat(frames, ignore_index=True)
    print("\n" + summarise(results).round(4).to_string(index=False))
    print("\nfusion - vision, paired within seed:")
    print(fusion_gain(results).round(4).to_string(index=False))
    return _write_results(results, Path(args.out), args.backbone)


@dataclass(slots=True)
class _Clock:
    """Cell-level progress; a whole seed is 15 model fits and far too coarse to watch."""

    total: int
    done: int = 0
    started: float = field(default_factory=time.perf_counter)

    def tick(self, label: str) -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.started
        remaining = elapsed / self.done * (self.total - self.done)
        LOG.info("%s  %d/%d  eta %s", label, self.done, self.total, _mmss(remaining))


def _mmss(seconds: float) -> str:
    minutes, remainder = divmod(int(max(seconds, 0.0)), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _ablate_seed(
    seed: int, severities: list[float], args: argparse.Namespace, clock: _Clock
) -> pd.DataFrame:
    """Rebuild the twin once per seed; severity only changes the image side."""
    LOG.info("seed %d: building twin", seed)
    config = TwinConfig(seed=seed, signal_gain=args.signal_gain)
    dataset = build_data.build(Path(args.root), config, oversample=args.oversample)
    frames = []
    for severity in severities:
        frames.append(_ablate_severity(dataset, severity, args))
        clock.tick(f"seed {seed} severity {severity:g}")
    results = pd.concat(frames, ignore_index=True)
    results.insert(0, "seed", seed)
    return results


def _ablate_severity(dataset, severity: float, args: argparse.Namespace) -> pd.DataFrame:
    """Image order is fixed across twin seeds, so the caches apply unchanged."""
    from ..models.ablation import AblationInputs, run
    from ..models.pipeline import FitConfig

    processed = Path(args.processed)
    regimes = [Regime(name) for name in args.regimes.split(",")]
    inputs = AblationInputs(
        train_frame=dataset.train,
        test_frame=dataset.test,
        regimes=[_regime_data(processed, args.backbone, r, severity) for r in regimes],
        n_components=args.components,
        fit_config=FitConfig(estimator=args.estimator, seed=args.fit_seed),
    )
    results = run(inputs)
    results.insert(0, "severity", severity)
    return results


def figures(args: argparse.Namespace) -> int:
    """Render the sweep charts from one or more results tables written by `ablate`."""
    from ..report import write_comparison_figure, write_sweep_figures

    tables = [Path(value) for value in args.results.split(",")]
    written = list(write_sweep_figures(pd.read_csv(tables[0]), Path(args.out)))
    if len(tables) > 1:
        written.append(write_comparison_figure(_labelled(tables), Path(args.out)))
    for path in written:
        print(f"wrote {path}")
    return 0


def _labelled(tables: list[Path]) -> pd.DataFrame:
    """The backbone is not a column in the results; it is in the file name."""
    frames = [pd.read_csv(path).assign(backbone=_backbone_of(path)) for path in tables]
    return pd.concat(frames, ignore_index=True)


def _backbone_of(path: Path) -> str:
    return path.stem.removeprefix("ablation_")


def explain(args: argparse.Namespace) -> int:
    """Grouped SHAP attribution for one fitted cell, global and per part."""
    from ..explain import explain as attribute
    from ..models.pipeline import FitConfig, fit

    dataset, blocks = _explain_blocks(args)
    model = fit(
        blocks.train, dataset.train["label"].to_numpy(), FitConfig(estimator=args.estimator)
    )
    attribution = attribute(model, blocks.test, blocks.names)
    print("grouped SHAP (mean |log-odds|):")
    print(attribution.importance().round(4).to_string())
    print(f"\nrow {args.row}, label {dataset.test['label'].iloc[args.row]}:")
    print(attribution.explain_row(args.row).round(4).to_string())
    _print_effects(model, blocks, args)
    _print_anchor(model, blocks, args)
    return _write_attribution(attribution, dataset.test, Path(args.out), args.backbone)


def _process_indices(names: list[str]) -> list[int]:
    """Anchors and effect curves stay on process columns; a PC is not a shop-floor action."""
    return [position for position, name in enumerate(names) if name in FEATURES]


def _print_effects(model, blocks, args: argparse.Namespace) -> None:
    """Accumulated local effects, ranked by how far each feature moves the model."""
    from ..explain import ale

    curves = [
        ale(model.score, blocks.test, index, blocks.names[index])
        for index in _process_indices(blocks.names)
    ]
    spans = pd.Series({curve.feature: curve.span() for curve in curves})
    print("\nALE span, main effect only, in probability (not comparable to the log-odds above):")
    print(spans.sort_values(ascending=False).round(4).to_string())


def _print_anchor(model, blocks, args: argparse.Namespace) -> None:
    from ..explain import anchor

    rule = anchor(
        lambda features: (model.score(features) >= model.threshold).astype(int),
        blocks.test[args.row],
        blocks.train,
        blocks.names,
        candidates=_process_indices(blocks.names),
        seed=args.seed,
    )
    print(f"\nanchor for row {args.row}:")
    print(rule.describe())


def _explain_blocks(args: argparse.Namespace):
    """The twin is rebuilt so the explanation lines up with the rows it describes."""
    from ..models.features import Modality, build_blocks

    config = TwinConfig(seed=args.seed, signal_gain=args.signal_gain)
    dataset = build_data.build(Path(args.root), config, oversample=args.oversample)
    regime = Regime(args.regime)
    data = _regime_data(Path(args.processed), args.backbone, regime, args.severity)
    blocks = build_blocks(
        Modality(args.modality),
        dataset.train,
        dataset.test,
        data.train_embeddings,
        data.test_embeddings,
        args.components,
    )
    return dataset, blocks


def _write_attribution(attribution, test: pd.DataFrame, out: Path, backbone: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    frame = attribution.frame()
    frame.insert(0, "label", test["label"].to_numpy())
    destination = out / f"attribution_{backbone}.csv"
    frame.to_csv(destination, index=False)
    print(f"\nwrote {destination}")
    return 0


def economics(args: argparse.Namespace) -> int:
    """Price one fitted cell: corrected priors, PAF ledger, and the sensitivity band."""
    from ..economics import CostModel, operate, prevalence, shift
    from ..models.pipeline import FitConfig, fit

    dataset, blocks = _explain_blocks(args)
    labels = dataset.test["label"].to_numpy()
    model = fit(
        blocks.train, dataset.train["label"].to_numpy(), FitConfig(estimator=args.estimator)
    )
    source = prevalence(dataset.train["label"].to_numpy())
    scores = shift(model.score(blocks.test), source, args.prevalence)
    costs = CostModel(args.scrap, args.inspection, args.escape_multiplier)
    point = operate(labels, scores, args.prevalence, costs, args.shots)
    _print_economics(point, source, args)
    return _write_economics(labels, scores, point, costs, args)


def _print_economics(point, source: float, args: argparse.Namespace) -> None:
    print(f"prior correction {source:.3f} -> {args.prevalence:.3f}")
    print(
        f"threshold {point.threshold:.3f}  escape {point.gate.outcome.escape_rate:.3f}  "
        f"overkill {point.gate.outcome.overkill_rate:.3f}"
    )
    print(f"alert rate {point.gate.alert_rate:.3f}  ({_alarms_per_hour(point.gate):.1f}/hour)")
    print(f"\ncost per {args.shots} shots:")
    print(point.frame().round(2).to_string())
    print(
        f"\nsaving vs ship-all {point.savings_vs_ship:,.0f}  "
        f"vs inspect-all {point.savings_vs_inspect:,.0f}"
    )


SHOTS_PER_HOUR = 60.0


def _alarms_per_hour(gate) -> float:
    """ISA-18.2 caps an operator at 6-12 alarms/hour; a one-minute cycle makes that comparable."""
    return gate.alert_rate * SHOTS_PER_HOUR


def _write_economics(labels, scores, point, costs, args: argparse.Namespace) -> int:
    """The sweeps are the honest headline; a single saving figure hides the guessed multiplier."""
    from ..economics import multiplier_sweep

    band = multiplier_sweep(labels, scores, args.prevalence, costs, shots=args.shots)
    print("\nsensitivity to the escape multiplier:")
    print(band.round(2).to_string(index=False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"economics_{args.backbone}.csv"
    band.to_csv(destination, index=False)
    print(f"\nwrote {destination}")
    return 0


def prescribe(args: argparse.Namespace) -> int:
    """Recommend setpoint changes for the riskiest shot in a run, then stress-test the advice."""
    from ..prescribe import fit as fit_surrogate
    from ..prescribe import recommend

    config = TwinConfig(seed=args.seed, signal_gain=args.signal_gain)
    LOG.info("fitting surrogate on %d interventional shots", args.design)
    surrogate = fit_surrogate(shots=args.design, seed=args.seed, config=config)
    frame = run_line(args.shots, config)
    reading = frame.iloc[int(surrogate.risk(frame).argmax())].to_dict()
    advice = recommend(surrogate, reading, max_actions=args.max_actions)
    print(f"riskiest of {args.shots} shots:\n")
    print(advice.describe())
    _print_stability(advice, reading, args)
    return 0


def _print_stability(advice, reading: dict, args: argparse.Namespace) -> None:
    """Advice scored against the twin that produced it is circular; perturb the weights instead."""
    from ..prescribe import scale_sweep

    if not advice.actions:
        return
    table = scale_sweep(advice, reading, trials=args.trials)
    print("\nstability under perturbed mechanism weights:")
    print(table.round(4).to_string(index=False))


def export(args: argparse.Namespace) -> int:
    """Write the dashboard star schema: one coherent run, scored, attributed and charted."""
    from ..economics import CostModel, cost_curve, optimal_threshold, prevalence, shift
    from ..export import ExportInputs, risk_chart, write
    from ..models.pipeline import FitConfig, fit

    dataset, blocks = _explain_blocks(args)
    model = fit(
        blocks.train, dataset.train["label"].to_numpy(), FitConfig(estimator=args.estimator)
    )
    source = prevalence(dataset.train["label"].to_numpy())
    risk = shift(model.score(blocks.test), source, args.prevalence)
    costs = CostModel()
    threshold = optimal_threshold(dataset.test["label"].to_numpy(), risk, args.prevalence, costs)
    line, line_risk = _production_run(dataset, args)
    inputs = ExportInputs(
        shots=dataset.test,
        risk=risk,
        threshold=threshold,
        cost_curve=cost_curve(dataset.test["label"].to_numpy(), risk, args.prevalence, costs),
        attribution=_attribution_frame(model, blocks),
        spc=risk_chart(line_risk),
        production=line,
        production_risk=line_risk,
    )
    written = write(inputs, Path(args.out))
    for name, path in written.items():
        print(f"{name:<18} {path}")
    return _write_powerbi_project(Path(args.out), args)


def _write_powerbi_project(exports: Path, args: argparse.Namespace) -> int:
    """A .pbix is binary and only Desktop can author it; PBIP is its plain-text equivalent."""
    from ..export import write_project

    if args.no_pbip:
        return 0
    project = Path(args.pbip)
    write_project(project, exports.resolve())
    print(f"\npower bi project  {project / 'DefectLab.pbip'}")
    print("open it in Power BI Desktop, then File > Save As to produce the .pbix")
    return 0


def _production_run(dataset, args: argparse.Namespace):
    """A contiguous run for the monitoring page, scored on process telemetry alone.

    The evaluation set is oversampled and grouped by label, so it has no usable time axis. A
    real line also has telemetry for every shot but an image only for imaged parts, so the
    process channel is what a continuous monitor actually reads.
    """
    from ..economics import prevalence, shift
    from ..models.pipeline import FitConfig, fit

    columns = list(FEATURES)
    labels = dataset.train["label"].to_numpy()
    model = fit(dataset.train[columns].to_numpy(), labels, FitConfig(estimator=args.estimator))
    config = TwinConfig(seed=args.seed + 1, signal_gain=args.signal_gain)
    line = score(run_line(args.line_shots, config), config, target_prevalence=args.prevalence)
    risk = shift(model.score(line[columns].to_numpy()), prevalence(labels), args.prevalence)
    return line, risk


def _attribution_frame(model, blocks):
    """Grouped SHAP over the exported rows; the dashboard shows why, not just what."""
    from ..explain import explain as attribute

    return attribute(model, blocks.test, blocks.names).frame()


def serve(args: argparse.Namespace) -> int:
    """Run the scoring API. The model is fitted at startup, so the first request is not slow."""
    import uvicorn

    from ..api.app import create_app

    uvicorn.run(
        create_app(seed=args.seed, estimator=args.estimator),
        host=args.host,
        port=args.port,
        log_level="warning" if args.quiet else "info",
    )
    return 0


def gates(args: argparse.Namespace) -> int:
    """Report the Gate 1 and Gate 2 diagnostics without training anything heavy."""
    config = TwinConfig(seed=args.seed, noise_sd=args.noise_sd, signal_gain=args.signal_gain)
    frame = score(run_line(args.shots, config), config, target_prevalence=args.prevalence)
    correlations = frame[list(FEATURES)].corrwith(frame["label"]).abs().sort_values(ascending=False)
    print(f"prevalence {frame['label'].mean():.4f}   lots {frame['lot_id'].nunique()}")
    print(correlations.round(3).to_string())
    print(f"\nmax |corr| {correlations.max():.3f}  (gate < 0.35)")
    return 0


def _regimes(name: str) -> tuple[Regime, ...]:
    return tuple(Regime) if name == "both" else (Regime(name),)


def _paired(processed: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(processed / f"{split}_paired.parquet")


def _regime_data(processed: Path, backbone: str, regime: Regime, severity: float = 1.0):
    """Embeddings must already be cached; extraction is a separate, explicit step."""
    from ..models.ablation import RegimeData

    paths = {
        split: cache_path(processed, backbone, split, regime, severity)
        for split in build_data.SPLITS
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run `defectlab extract` first; missing: {', '.join(missing)}")
    return RegimeData(regime, np.load(paths["train"]), np.load(paths["test"]))


def _write_results(results: pd.DataFrame, out: Path, backbone: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"ablation_{backbone}.csv"
    results.to_csv(destination, index=False)
    print(f"\nwrote {destination}")
    return 0


def default_root() -> Path:
    return settings.paths.casting_root


def default_processed() -> Path:
    return settings.paths.processed
