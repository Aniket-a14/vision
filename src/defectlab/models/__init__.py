"""Estimators, feature assembly, calibration, conformal prediction and thresholds.

`ablation` is deliberately not re-exported here. It imports `imaging`, and therefore OpenCV, and
a package init that pulls it in would put a vision dependency on the path of anything that merely
scores process telemetry -- which is how the serving container first failed. Import it directly:
`from defectlab.models.ablation import run`.
"""

from .conformal import MondrianConformal, PredictionSets, coverage_by_class
from .estimators import BUILDERS, build
from .evaluation import Scores, evaluate
from .features import Modality, build_blocks, fit_image_reducer
from .pipeline import CellData, FitConfig, FittedModel, fit
from .thresholds import CostMatrix, choose, cost_optimal, sweep

__all__ = [
    "BUILDERS",
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
    "sweep",
]
