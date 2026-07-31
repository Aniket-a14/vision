"""Dataset discovery, twin-to-image pairing, contracts and group-aware splits."""

from .images import ImageSplit, load_split, verify_counts
from .pairing import build_paired_frame
from .schemas import LabelledShotSchema, PairedShotSchema, ShotSchema
from .splits import assert_disjoint, grouped_folds, grouped_holdout

__all__ = [
    "ImageSplit",
    "LabelledShotSchema",
    "PairedShotSchema",
    "ShotSchema",
    "assert_disjoint",
    "build_paired_frame",
    "grouped_folds",
    "grouped_holdout",
    "load_split",
    "verify_counts",
]
