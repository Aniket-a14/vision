"""Estimators, feature assembly, calibration, conformal prediction and thresholds."""

from .conformal import MondrianConformal, PredictionSets, coverage_by_class
from .estimators import BUILDERS, build
from .evaluation import Scores, evaluate
from .features import Modality, build_blocks, fit_image_reducer
from .pipeline import AblationResult, CellData, FitConfig, FittedModel, fit, run_cell
from .thresholds import CostMatrix, choose, cost_optimal, sweep

__all__ = [
    "BUILDERS",
    "AblationResult",
    "CellData",
    "CostMatrix",
    "FitConfig",
    "FittedModel",
    "Modality",
    "MondrianConformal",
    "PredictionSets",
    "Scores",
    "build",
    "build_blocks",
    "choose",
    "cost_optimal",
    "coverage_by_class",
    "evaluate",
    "fit",
    "fit_image_reducer",
    "run_cell",
    "sweep",
]
