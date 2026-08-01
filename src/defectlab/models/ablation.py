"""The 3x2 ablation and the degradation sweep.

Three modalities by two imaging regimes is the experiment. The sweep over degradation
severity is the headline figure: process signal is uncorrelated with image quality, so
fusion should hold up as the camera gets worse.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from ..imaging import Regime
from .features import Modality, build_blocks
from .pipeline import AblationResult, CellData, FitConfig, run_cell

MODALITIES: tuple[Modality, ...] = (Modality.VISION, Modality.PROCESS, Modality.FUSION)
REGIMES: tuple[Regime, ...] = (Regime.LAB, Regime.INLINE)


@dataclass(frozen=True, slots=True)
class RegimeData:
    """Cached embeddings for one imaging regime."""

    regime: Regime
    train_embeddings: np.ndarray
    test_embeddings: np.ndarray


@dataclass(frozen=True, slots=True)
class AblationInputs:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    regimes: Sequence[RegimeData]
    n_components: int = 30
    fit_config: FitConfig = field(default_factory=FitConfig)

    @property
    def train_labels(self) -> np.ndarray:
        return self.train_frame["label"].to_numpy()

    @property
    def test_labels(self) -> np.ndarray:
        return self.test_frame["label"].to_numpy()


def run(inputs: AblationInputs) -> pd.DataFrame:
    """Every modality against every regime, as one tidy results table."""
    return pd.DataFrame([result.as_row() for result in _iter_results(inputs)])


EmbeddingSource = Callable[[float], tuple[np.ndarray, np.ndarray]]


def degradation_sweep(
    inputs: AblationInputs, severities: Sequence[float], embeddings_for: EmbeddingSource
) -> pd.DataFrame:
    """ROC-AUC against camera severity for all three modalities."""
    rows = []
    for severity in severities:
        train_embeddings, test_embeddings = embeddings_for(severity)
        data = RegimeData(Regime.INLINE, train_embeddings, test_embeddings)
        for result in _iter_modalities(inputs, data):
            rows.append({"severity": severity, **result.as_row()})
    return pd.DataFrame(rows)


SPREAD = ("mean", "std", "min", "max")


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Spread across seeds; the number of alloy lots, not parts, sets the precision."""
    keys = _present(results, "modality", "regime", "severity")
    return results.groupby(keys)["roc_auc"].agg(list(SPREAD)).reset_index()


def fusion_gain(results: pd.DataFrame) -> pd.DataFrame:
    """Fusion minus vision, paired within seed so the image channel cancels out."""
    keys = _present(results, "regime", "severity")
    deltas = _paired_deltas(results)
    grouped = deltas.groupby(keys)["delta"]
    tests = grouped.apply(_paired_test).unstack()
    return grouped.agg(list(SPREAD)).join(tests).reset_index()


def _paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    """One delta per seed; the seed is the experimental unit, not the part."""
    index = _present(results, "seed", "regime", "severity")
    wide = results.pivot_table(index=index, columns="modality", values="roc_auc")
    delta = wide[Modality.FUSION.value] - wide[Modality.VISION.value]
    return delta.rename("delta").reset_index()


def _paired_test(deltas: pd.Series) -> pd.Series:
    """Seeds are few, so report t and p rather than leaning on a bootstrap."""
    statistic, pvalue = stats.ttest_1samp(deltas, 0.0)
    wins = int((deltas > 0.0).sum())
    return pd.Series({"t": statistic, "p": pvalue, "wins": wins, "n": len(deltas)})


def _present(results: pd.DataFrame, *names: str) -> list[str]:
    """A single-severity run has no severity column; the same helpers still apply."""
    return [name for name in names if name in results.columns]


def _iter_results(inputs: AblationInputs) -> Iterator[AblationResult]:
    for data in inputs.regimes:
        yield from _iter_modalities(inputs, data)


def _iter_modalities(inputs: AblationInputs, data: RegimeData) -> Iterator[AblationResult]:
    for modality in MODALITIES:
        yield _run_one(inputs, data, modality)


def _run_one(inputs: AblationInputs, data: RegimeData, modality: Modality) -> AblationResult:
    blocks = build_blocks(
        modality,
        inputs.train_frame,
        inputs.test_frame,
        data.train_embeddings,
        data.test_embeddings,
        inputs.n_components,
    )
    cell = CellData(blocks.train, inputs.train_labels, blocks.test, inputs.test_labels)
    return run_cell(modality, data.regime, cell, inputs.fit_config)


def component_ablation(
    inputs: AblationInputs, data: RegimeData, component_counts: Sequence[int]
) -> pd.DataFrame:
    """Wider image blocks suppress the tabular signal; this finds the balance point."""
    rows = []
    for count in component_counts:
        narrowed = replace(inputs, n_components=count)
        result = _run_one(narrowed, data, Modality.FUSION)
        rows.append({"n_components": count, **result.as_row()})
    return pd.DataFrame(rows)
