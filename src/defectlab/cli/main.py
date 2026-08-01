"""Argument parsing and dispatch. Command bodies live in commands.py."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from ..config import settings
from . import commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="defectlab", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_verify(subparsers)
    _add_simulate(subparsers)
    _add_extract(subparsers)
    _add_ablate(subparsers)
    _add_gates(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(getattr(args, "quiet", False))
    return args.handler(args)


def configure_logging(quiet: bool = False) -> None:
    """Progress goes to stderr so piped stdout stays machine-readable."""
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    for noisy in ("httpx", "urllib3", "filelock", "timm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _add_verify(subparsers) -> None:
    parser = subparsers.add_parser("verify", help="check the casting dataset folder counts")
    parser.add_argument("--root", default=str(commands.default_root()))
    parser.add_argument("--lenient", action="store_true", help="report counts without asserting")
    parser.set_defaults(handler=commands.verify)


def _add_simulate(subparsers) -> None:
    parser = subparsers.add_parser("simulate", help="build the paired train and test tables")
    parser.add_argument("--root", default=str(commands.default_root()))
    parser.add_argument("--out", default=str(commands.default_processed()))
    parser.add_argument("--oversample", type=int, default=4)
    _add_twin_arguments(parser)
    parser.set_defaults(handler=commands.simulate)


def _add_extract(subparsers) -> None:
    parser = subparsers.add_parser("extract", help="cache image embeddings")
    parser.add_argument("--processed", default=str(commands.default_processed()))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--regime", default="both", choices=["lab", "inline", "both"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--quiet", action="store_true", help="suppress per-batch progress")
    parser.set_defaults(handler=commands.extract)


def _add_ablate(subparsers) -> None:
    parser = subparsers.add_parser("ablate", help="run the 3x2 modality-by-regime ablation")
    parser.add_argument("--root", default=str(commands.default_root()))
    parser.add_argument("--processed", default=str(commands.default_processed()))
    parser.add_argument("--out", default="results")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--estimator", default="xgboost")
    parser.add_argument("--components", type=int, default=settings.pca_components)
    parser.add_argument("--seeds", default="42,7,99,123,2024", help="comma-separated twin seeds")
    parser.add_argument("--signal-gain", type=float, default=settings.signal_gain)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=settings.seed)
    parser.set_defaults(handler=commands.ablate)


def _add_gates(subparsers) -> None:
    parser = subparsers.add_parser("gates", help="print the leakage and prevalence diagnostics")
    parser.add_argument("--shots", type=int, default=12000)
    parser.add_argument("--prevalence", type=float, default=0.567)
    _add_twin_arguments(parser)
    parser.set_defaults(handler=commands.gates)


def _add_twin_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--noise-sd", type=float, default=settings.noise_sd)
    parser.add_argument("--signal-gain", type=float, default=settings.signal_gain)


if __name__ == "__main__":
    raise SystemExit(main())
