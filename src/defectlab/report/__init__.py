"""Figures rendered from results tables. Presentation only; nothing is fitted here."""

from .figures import (
    degradation_curve,
    gain_curve,
    headroom_curve,
    write_comparison_figure,
    write_sweep_figures,
)

__all__ = [
    "degradation_curve",
    "gain_curve",
    "headroom_curve",
    "write_comparison_figure",
    "write_sweep_figures",
]
