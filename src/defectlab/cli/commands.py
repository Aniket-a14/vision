"""One function per CLI command. Each delegates to the library and prints a summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import settings
from ..data import build as build_data
from ..data.images import verify_counts
from ..imaging import Regime
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
            destination = cache_path(processed, args.backbone, split, regime)
            features = extract_cached(
                build_data.image_paths(frame),
                regime,
                args.backbone,
                destination,
                seed=args.seed,
                batch_size=args.batch_size,
            )
            print(f"{split:<6} {regime.value:<7} {features.shape} -> {destination.name}")
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


def default_root() -> Path:
    return settings.paths.casting_root


def default_processed() -> Path:
    return settings.paths.processed
