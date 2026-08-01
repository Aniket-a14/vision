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
    delta = _deltas(results)
    axes.scatter(
        delta["severity"], delta["delta"], color="#888888", s=14, zorder=3, label="per seed"
    )


def _deltas(results: pd.DataFrame) -> pd.DataFrame:
    """Fusion minus vision per seed and severity, keeping any backbone label."""
    index = [name for name in ("backbone", "seed", "severity") if name in results.columns]
    wide = results.pivot_table(index=index, columns="modality", values="roc_auc")
    return (wide["fusion"] - wide["vision"]).rename("delta").reset_index()


BACKBONE_MARKERS = {"resnet18": "o", "dinov2_s": "s"}
BACKBONE_COLOURS = {"resnet18": "#8e44ad", "dinov2_s": "#d35400"}


def write_comparison_figure(results: pd.DataFrame, out: Path) -> Path:
    """The headroom chart, which needs a `backbone` column spanning both sweeps."""
    out.mkdir(parents=True, exist_ok=True)
    return _render(headroom_curve, results, out / "headroom_curve.png")


def headroom_curve(results: pd.DataFrame, axes) -> None:
    """Gain against vision AUC. Severity is a dial; what fusion answers to is headroom."""
    deltas = _deltas(results)
    for backbone, frame in deltas.groupby("backbone"):
        _headroom_points(axes, results, frame, str(backbone))
    axes.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axes.invert_xaxis()
    axes.set_xlabel("vision-only ROC-AUC (worse camera to the right)")
    axes.set_ylabel("fusion - vision (ROC-AUC)")
    axes.set_title("The gain tracks what the camera cost vision, not the backbone")
    axes.legend(frameon=False)
    axes.grid(alpha=0.25)


def _headroom_points(axes, results: pd.DataFrame, frame: pd.DataFrame, backbone: str) -> None:
    vision = _vision_auc(results, backbone)
    merged = frame.assign(vision=frame["severity"].map(vision))
    axes.scatter(
        merged["vision"],
        merged["delta"],
        marker=BACKBONE_MARKERS.get(backbone, "o"),
        color=BACKBONE_COLOURS.get(backbone, "#333333"),
        alpha=0.7,
        s=28,
        label=backbone,
    )


def _vision_auc(results: pd.DataFrame, backbone: str) -> pd.Series:
    """Vision is identical across seeds, so one value per severity is the whole story."""
    frame = results[(results["backbone"] == backbone) & (results["modality"] == "vision")]
    return frame.groupby("severity")["roc_auc"].mean()
