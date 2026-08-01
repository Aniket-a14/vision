"""Physics-based digital twin of an Al-Si high-pressure die-casting line."""

from .parameters import CONTROLLABLE, FEATURES, LOT_LEVEL, ParameterSpec, spec
from .propensity import MECHANISM_WEIGHTS
from .simulator import SETPOINTS, TwinConfig, run_line, score, stream_line

__all__ = [
    "CONTROLLABLE",
    "FEATURES",
    "LOT_LEVEL",
    "MECHANISM_WEIGHTS",
    "SETPOINTS",
    "ParameterSpec",
    "TwinConfig",
    "run_line",
    "score",
    "spec",
    "stream_line",
]
