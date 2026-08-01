"""The degradation sweep charts.

Every curve shows the spread across twin seeds, never a single seed. The effective
sample size is the number of alloy lots, so a lone seed is not a measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ..models.ablation import fusion_gain

matplotlib.use("Agg")

COLOURS = {"vision": "#c0392b", "process": "#2980b9", "fusion": "#27ae60"}
ORDER = ("vision", "process", "fusion")
FIGSIZE = (7.0, 4.5)
DPI = 200


def write_sweep_figures(results: pd.DataFrame, out: Path) -> list[Path]:
    """Render both sweep charts and return what was written."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, draw in (("degradation_curve", degradation_curve), ("fusion_gain", gain_curve)):
        written.append(_render(draw, results, out / f"{name}.png"))
    return written


def _render(draw, results: pd.DataFrame, destination: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGSIZE)
    draw(results, axes)
    figure.tight_layout()
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def degradation_curve(results: pd.DataFrame, axes) -> None:
    """ROC-AUC against camera severity, one band per modality."""
    for modality in ORDER:
        stats = _by_severity(results, modality)
        _band(axes, stats, COLOURS[modality], modality)
    axes.set_xlabel("inline camera severity")
    axes.set_ylabel("ROC-AUC")
    axes.set_title("Fusion degrades more slowly than vision alone")
    axes.legend(frameon=False)
    axes.grid(alpha=0.25)


def _by_severity(results: pd.DataFrame, modality: str) -> pd.DataFrame:
    frame = results[results["modality"] == modality]
    return frame.groupby("severity")["roc_auc"].agg(["mean", "std"]).fillna(0.0).reset_index()


def _band(axes, stats: pd.DataFrame, colour: str, label: str) -> None:
    """Mean with a +/- 1 sd ribbon; the ribbon is the seed spread, not a confidence band."""
    axes.plot(stats["severity"], stats["mean"], marker="o", color=colour, label=label)
    low = stats["mean"] - stats["std"]
    high = stats["mean"] + stats["std"]
    axes.fill_between(stats["severity"], low, high, color=colour, alpha=0.15, linewidth=0)


def gain_curve(results: pd.DataFrame, axes) -> None:
    """Fusion minus vision per severity, with each seed shown so the spread is visible."""
    gains = fusion_gain(results)
    _seed_points(results, axes)
    axes.errorbar(
        gains["severity"],
        gains["mean"],
        yerr=gains["std"],
        marker="o",
        color=COLOURS["fusion"],
        capsize=4,
        label="mean +/- 1 sd",
    )
    axes.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axes.set_xlabel("inline camera severity")
    axes.set_ylabel("fusion - vision (ROC-AUC)")
    axes.set_title("The fusion advantage grows as imaging degrades")
    axes.legend(frameon=False)
    axes.grid(alpha=0.25)


def _seed_points(results: pd.DataFrame, axes) -> None:
    """Individual seeds, so a positive mean cannot hide a seed that disagrees."""
    wide = results.pivot_table(index=["seed", "severity"], columns="modality", values="roc_auc")
    delta = (wide["fusion"] - wide["vision"]).reset_index()
    axes.scatter(delta["severity"], delta[0], color="#888888", s=14, zorder=3, label="per seed")
