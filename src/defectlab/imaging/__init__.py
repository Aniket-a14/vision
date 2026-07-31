"""Inline camera simulation and frozen-backbone feature extraction."""

from .backbones import BACKBONES, DEFAULT_BACKBONE, BackboneSpec, spec_for
from .degrade import InlineCamera, Regime, apply_regime, degrade

__all__ = [
    "BACKBONES",
    "DEFAULT_BACKBONE",
    "BackboneSpec",
    "InlineCamera",
    "Regime",
    "apply_regime",
    "degrade",
    "spec_for",
]
