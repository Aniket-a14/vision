"""One function per CLI command. Each delegates to the library and prints a summary."""

from __future__ import annotations

import argparse
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
    results = pd.concat([_ablate_seed(seed, severities, args) for seed in seeds], ignore_index=True)
    print("\n" + summarise(results).round(4).to_string(index=False))
    print("\nfusion - vision, paired within seed:")
    print(fusion_gain(results).round(4).to_string(index=False))
    return _write_results(results, Path(args.out), args.backbone)


def _ablate_seed(seed: int, severities: list[float], args: argparse.Namespace) -> pd.DataFrame:
    """Rebuild the twin once per seed; severity only changes the image side."""
    config = TwinConfig(seed=seed, signal_gain=args.signal_gain)
    dataset = build_data.build(Path(args.root), config, oversample=args.oversample)
    frames = [_ablate_severity(dataset, severity, args) for severity in severities]
    results = pd.concat(frames, ignore_index=True)
    results.insert(0, "seed", seed)
    print(f"seed {seed} done")
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
