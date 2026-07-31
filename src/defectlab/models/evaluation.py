"""Metrics for a quality gate.

Escape rate and overkill rate are reported separately; a single accuracy figure is
not an acceptable specification for a scrap-or-ship decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class Scores:
    roc_auc: float
    pr_auc: float
    brier: float
    recall: float
    precision: float
    f1: float
    escape_rate: float
    overkill_rate: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Scores:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return Scores(
        roc_auc=float(roc_auc_score(labels, scores)),
        pr_auc=float(average_precision_score(labels, scores)),
        brier=float(brier_score_loss(labels, scores)),
        recall=_ratio(tp, tp + fn),
        precision=_ratio(tp, tp + fp),
        f1=_f1(tp, fp, fn),
        escape_rate=_ratio(fn, fn + tp),
        overkill_rate=_ratio(fp, fp + tn),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0
